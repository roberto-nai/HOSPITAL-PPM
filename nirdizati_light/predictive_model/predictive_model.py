import os
import logging
import numpy as np
import torch
from typing import Union, Optional, Type
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from hyperopt import STATUS_OK, STATUS_FAIL
from pandas import DataFrame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier,XGBRegressor
from nirdizati_light.evaluation.common import evaluate_classifier, evaluate_regressor
from nirdizati_light.predictive_model.common import ClassificationMethods, RegressionMethods, get_tensor, shape_label_df, LambdaModule, EarlyStopper

logger = logging.getLogger(__name__)


def drop_columns(df: DataFrame) -> DataFrame:
    df = df.drop(['trace_id', 'label'],axis=1)
    return df

class PredictiveModel:
    """
    A class representing a predictive model.

    Args:
        model_type (Union[ClassificationMethods, RegressionMethods]): Type of predictive model.
        train_df (DataFrame): Training data to train model.
        validate_df (DataFrame): Validation data to evaluate model.
        test_df (DataFrame): Test data to evaluate model.
        prefix_length (int): Length of prefix to consider.
        hyperopt_space (Optional[dict]): Space to perform hyperparameter optimization on; if not provided, fallbacks to default values. Defaults to None.
        custom_model_class (Optional[Type[Module]]): Class of a custom PyTorch module. Defaults to None.
    """

    def __init__(
        self,
        model_type: Union[ClassificationMethods, RegressionMethods],
        train_df: DataFrame,
        validate_df: DataFrame,
        test_df: DataFrame,
        prefix_length: int,
        hyperopt_space: Optional[dict]=None,
        custom_model_class: Optional[Type[Module]]=None
    ):
        self.model_type = model_type
        self.config = None
        self.model = None
        self.full_train_df = train_df
        self.train_df = drop_columns(train_df)
        self.train_df_shaped = None
        self.full_validate_df = validate_df
        self.validate_df = drop_columns(validate_df)
        self.validate_df_shaped = None
        self.full_test_df = test_df
        self.test_df = drop_columns(test_df)
        self.test_df_shaped = None

        self.hyperopt_space = hyperopt_space
        self.custom_model_class = custom_model_class

        if model_type in [ClassificationMethods.LSTM.value, ClassificationMethods.CUSTOM_PYTORCH.value]:
            self.train_tensor = get_tensor(self.train_df, prefix_length)
            self.validate_tensor = get_tensor(self.validate_df, prefix_length)
            self.test_tensor = get_tensor(self.test_df, prefix_length)

            self.train_label = shape_label_df(self.full_train_df)
            self.validate_label = shape_label_df(self.full_validate_df)
            self.test_label = shape_label_df(self.full_test_df)

        elif model_type is ClassificationMethods.MLP.value:
            self.train_label = self.full_train_df['label'].nunique()
            self.validate_label = self.full_validate_df['label'].nunique()
            self.test_label = self.full_test_df['label'].unique()
    
    def train_and_evaluate_configuration(self, config, target):
        try:
            self.model = self._instantiate_model(config)
            self._fit_model(self.model, config)
            actual = self.full_validate_df['label']
            
            if self.model_type in [ClassificationMethods.LSTM.value, ClassificationMethods.CUSTOM_PYTORCH.value]:
                actual = np.array(actual.to_list())

            if self.model_type in [item.value for item in ClassificationMethods]:
                predicted, scores = self.predict(test=False)
                result = evaluate_classifier(actual, predicted, scores, loss=target)
            elif self.model_type in [item.value for item in RegressionMethods]:
                predicted = self.model.predict(self.validate_df)
                result = evaluate_regressor(actual, predicted, loss=target)
            else:
                raise Exception('Unsupported model_type')

            return {
                'status': STATUS_OK,
                'loss': - result['loss'],  # we are using fmin for hyperopt
                'exception': None,
                'config': config,
                'model': self.model,
                'result': result,
            }
        except Exception as e:
            return {
                'status': STATUS_FAIL,
                'loss': 0,
                'exception': str(e)
            }

    def _instantiate_model(self, config):
        if self.model_type is ClassificationMethods.RANDOM_FOREST.value:
            model = RandomForestClassifier(**config)
        elif self.model_type is ClassificationMethods.DT.value:
            model = DecisionTreeClassifier(**config)
        elif self.model_type == ClassificationMethods.KNN.value:
            model = KNeighborsClassifier(**config)
        elif self.model_type == ClassificationMethods.XGBOOST.value:
            model = XGBClassifier(**config)
        elif self.model_type == ClassificationMethods.SGDCLASSIFIER.value:
            model = SGDClassifier(**config)
        elif self.model_type == ClassificationMethods.PERCEPTRON.value:
            # added CalibratedClassifier to get predict_proba from perceptron model
            model = Perceptron(**config)
            model = CalibratedClassifierCV(model, cv=10, method='isotonic')
        elif self.model_type is ClassificationMethods.MLP.value:
            model = MLPClassifier(**config)
            #model = CalibratedClassifierCV(model, cv=10, method='isotonic')
        elif self.model_type == RegressionMethods.RANDOM_FOREST.value:
            model = RandomForestRegressor(**config)
        elif self.model_type == ClassificationMethods.SVM.value:
            model = SVC(**config,probability=True)
        elif self.model_type is ClassificationMethods.LSTM.value:
            model = torch.nn.Sequential(
                torch.nn.LSTM(
                    input_size=self.train_tensor.shape[2],
                    hidden_size=int(config['lstm_hidden_size']),
                    num_layers=int(config['lstm_num_layers']),
                    batch_first=True
                ),
                LambdaModule(lambda x: x[0][:,-1,:]),
                torch.nn.Linear(int(config['lstm_hidden_size']), self.train_label.shape[1]),
                torch.nn.Softmax(dim=1),
            ).to(torch.float32)
        elif self.model_type is ClassificationMethods.CUSTOM_PYTORCH.value:
            model = self.custom_model_class(
                input_dim=self.train_tensor.shape[2],
                output_dim=self.train_label.shape[1],
                config=config,
            ).to(torch.float32)
        else:
            raise Exception('unsupported model_type')
        
        return model

    def _fit_model(self, model, config=None):
        if self.model_type in [ClassificationMethods.LSTM.value, ClassificationMethods.CUSTOM_PYTORCH.value]:
            MAX_NUM_EPOCHS = config['max_num_epochs']

            train_dataset = TensorDataset(torch.tensor(self.train_tensor, dtype=torch.float32), torch.tensor(self.train_label, dtype=torch.float32))
            validate_dataset = TensorDataset(torch.tensor(self.validate_tensor, dtype=torch.float32), torch.tensor(self.validate_label, dtype=torch.float32))

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            validate_loader = DataLoader(validate_dataset, batch_size=config['batch_size'], shuffle=False)

            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            criterion = torch.nn.CrossEntropyLoss()
            
            early_stopper = EarlyStopper(patience=config['early_stop_patience'], min_delta=config['early_stop_min_delta'])

            for _ in range(MAX_NUM_EPOCHS):
                # training
                model.train()

                for inputs, labels in train_loader:
                    output = model(inputs)
                    loss = criterion(output, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # validation
                model.eval()
                validate_loss = 0
                
                with torch.no_grad():
                    for inputs, labels in validate_loader:
                        output = model(inputs)
                        validate_loss += criterion(output, labels).item()
                
                validate_loss /= len(validate_loader)

                if early_stopper.early_stop(validate_loss):             
                    break

        else:
            model.fit(self.train_df, self.full_train_df['label'])

    def predict(self, test: bool=True) -> str:
        """
        Performs predictions with the model and returns them.

        Args:
            test (bool): Whether to perform predictions on test set (`True`) or on validation set (`False`).

        Returns:
            tuple: A tuple with predicted values and scores for predictions.
        """

        data = self.test_df if test else self.validate_df

        if self.model_type in [ClassificationMethods.LSTM.value, ClassificationMethods.CUSTOM_PYTORCH.value]:
            data_tensor = torch.tensor(self.test_tensor if test else self.validate_tensor, dtype=torch.float32)

            probabilities = self.model(data_tensor).detach().numpy()
            predicted = np.argmax(probabilities, axis=1)
            scores = np.amax(probabilities, axis=1)
        else:
            predicted = self.model.predict(data)
            
            if hasattr(self.model, 'predict_proba'):
                scores = self.model.predict_proba(data)[:, 1]
            else:
                # Handle the case where predict_proba is not available
                # For example, this may be the case for SGDClassifier trained with certain losses
                scores = None

        return predicted, scores
    

    def save(self, path: str, name: str):
        """
        Save the model to the given path.

        Args:
            path (str): Path to save the model.
            name (str): Name of the model.

        Returns:
            str: Path to the saved model.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        path_with_name = os.path.join(path, name)
        
        if self.model_type in [ClassificationMethods.LSTM.value, ClassificationMethods.CUSTOM_PYTORCH.value]:
            # save pytorch model
            path_with_name += '.pt'
            torch.save(self.model.state_dict(), path_with_name)
        else:
            # save scikit-learn model
            path_with_name += '.joblib'
            import joblib
            joblib.dump(self.model, path_with_name)

        return path_with_name