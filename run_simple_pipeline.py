### IMPORTS ###
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

### LOCAL IMPORTS ###
from nirdizati_light.log.common import get_log, split_train_val_test
from nirdizati_light.encoding.common import get_encoded_df, EncodingType
from nirdizati_light.encoding.constants import TaskGenerationType, PrefixLengthStrategy, EncodingTypeAttribute
from nirdizati_light.encoding.time_encoding import TimeEncodingType
from nirdizati_light.labeling.common import LabelTypes
from nirdizati_light.predictive_model.common import ClassificationMethods
from nirdizati_light.predictive_model.predictive_model import PredictiveModel
from nirdizati_light.hyperparameter_optimisation.common import retrieve_best_model, HyperoptTarget
from nirdizati_light.evaluation.common import evaluate_classifier,evaluate_classifiers,plot_model_comparison
from nirdizati_light.explanation.common import ExplainerType, explain
from functions import create_output_directory

### GLOBASL VARIABLES ###
SEED = 1234
LOG_DIR = 'logs'            # directory with input logs
DATA_DIR = 'data'           # directory to save ML data (prefix end encodings)
RESULTS_DIR = 'results'     # directory to save ML results

### INPUT PARAMETERS ###
MIN_PREFIX_LENGTH = 1       # minimum prefix length (included)
MAX_PREFIX_LENGTH = 8       # maximum prefix length (included)
INPUT_LOG = 'eventlog_anonymous.csv'  # log file name
# INPUT_LOG = 'BPIC11_f1.csv'  # log file name
CSV_SEP = ';'                # separator used in the log file

### MAIN ###
if __name__ == '__main__':

    print()
    print("*** PROGRAM START ***")
    print()

    start_time = datetime.now().replace(microsecond=0)
    print("Start process:", str(start_time))
    print()

    random.seed(SEED)
    np.random.seed(SEED)

    ### INPUT LOG ###
    path_log = Path(LOG_DIR) / INPUT_LOG
    print(f'Input log: {path_log}')
    print()

    ### ENCODING LIST ###
    print(f'Encoding list:')
    encoding_list = list(EncodingType)
    print(encoding_list)
    print()
    # for i, encoding in enumerate(encoding_list):
    #     print(f'  - {i}: {encoding.value}')

    ### PREFIX LENGTH LIST ###
    print(f'Prefix length list:')
    list_prefix_length = list(range(MIN_PREFIX_LENGTH, MAX_PREFIX_LENGTH + 1))
    print(list_prefix_length)
    print()

    ### OUTPUT DIRECTORIES ###
    print(f'Output directories:')
    print(f'  - {DATA_DIR}')
    print(f'  - {RESULTS_DIR}')
    create_output_directory(DATA_DIR)
    create_output_directory(RESULTS_DIR)
    
    ### RUN PIPELINE ###
    print(f'Running pipeline...')
    print(f'Running pipeline for {len(list_prefix_length)} prefix lengths and {len(encoding_list)} encodings')
    print()

    list_results = [] # all the results fro every prefix length and encoding

    for i, prefix_len in enumerate(list_prefix_length, start=1):        

        print(f'[{i}]')
        print()

        for encoding in encoding_list:
            
            print(f'Prefix length: {prefix_len}')
            print(f'Encoding: {encoding.value}')

            CONF = {
                # path to log
                'data': path_log.as_posix(), # path to log file as string
                # train-validation-test set split percentages
                'train_val_test_split': [0.7, 0.1, 0.2],

                # path to output folder
                'output': 'output_data',

                'prefix_length_strategy': PrefixLengthStrategy.FIXED.value,
                'prefix_length': prefix_len,

                # whether to use padding or not in encoding
                'padding': True,
                # which encoding to use
                # 'feature_selection': EncodingType.SIMPLE.value,
                'feature_selection': encoding.value,

                # which attribute encoding to use
                'attribute_encoding': EncodingTypeAttribute.LABEL.value,
                # which time encoding to use
                'time_encoding': TimeEncodingType.DATE_AND_DURATION.value,

                # the label to be predicted (e.g. outcome, next activity)
                'labeling_type': LabelTypes.ATTRIBUTE_STRING.value,
                # whether the model should be trained on the specified prefix length (ONLY_THIS) or to every prefix in range [1, prefix_length] (ALL_IN_ONE)
                'task_generation_type': TaskGenerationType.ONLY_THIS.value,
                
                # list of predictive models and their respective hyperparameter optimization space
                # if it is None, then the default hyperopt space will be used; otherwise, the provided space will be used
                'predictive_models': [
                    #ClassificationMethods.RANDOM_FOREST.value,
                    #ClassificationMethods.KNN.value,
                    #ClassificationMethods.LSTM.value,
                    #ClassificationMethods.MLP.value,
                    # ClassificationMethods.PERCEPTRON.value,
                    # ClassificationMethods.SGDCLASSIFIER.value,
                    #ClassificationMethods.SVM.value,
                    ClassificationMethods.XGBOOST.value,
                ],
                
                # which metric to optimize hyperparameters for
                'hyperparameter_optimisation_target': HyperoptTarget.F1.value,
                # number of hyperparameter configurations to try
                'hyperparameter_optimisation_evaluations': 3,

                # explainability method to use
                'explanator': ExplainerType.DICE.value,
                
                'target_event': None,
                'seed': SEED,
            }

            print('Loading log...')
            log = get_log(filepath=CONF['data'], separator=CSV_SEP)

            print('Encoding traces...')
            encoder, full_df = get_encoded_df(
            log=log,
            feature_encoding_type=CONF['feature_selection'],
            prefix_length=CONF['prefix_length'],
            prefix_length_strategy=CONF['prefix_length_strategy'],
            time_encoding_type=CONF['time_encoding'],
            attribute_encoding=CONF['attribute_encoding'],
            padding=CONF['padding'],
            labeling_type=CONF['labeling_type'],
            task_generation_type=CONF['task_generation_type'],
            target_event=CONF['target_event'],
            )

            print('Saving encoded data...')
            path_data = Path(DATA_DIR) /f'{Path(INPUT_LOG).stem}_P_{prefix_len}_E_{encoding.value}.csv'
            print(f'Saving encoded data to: {path_data}')
            full_df.to_csv(path_data, index=False)
            print()

            print('Splitting in train, validation and test...')
            train_size, val_size, test_size = CONF['train_val_test_split']
            train_df, val_df, test_df = split_train_val_test(full_df, train_size, val_size, test_size, shuffle=False, seed=CONF['seed'])

            label_distribution_train = train_df['label'].value_counts()
            print("Label distribution in train_df:")
            print(label_distribution_train)

            label_distribution_val = val_df['label'].value_counts()
            print("Label distribution in val_size:")
            print(label_distribution_val)

            label_distribution_test = test_df['label'].value_counts()
            print("Label distribution in test_df:")
            print(label_distribution_test)
            
            print('Instantiating predictive models...')
            predictive_models = [PredictiveModel(predictive_model, train_df, val_df, test_df, prefix_length=CONF['prefix_length']) for predictive_model in CONF['predictive_models']]

            print('Running hyperparameter optimization...')
            best_candidates,best_model_idx, best_model_model, best_model_config = retrieve_best_model(
                predictive_models,
                max_evaluations=CONF['hyperparameter_optimisation_evaluations'],
                target=CONF['hyperparameter_optimisation_target']
            )

            best_model = predictive_models[best_model_idx]
            best_model.model = best_model_model
            best_model.config = best_model_config
            print(f'Best model is {best_model.model_type}')

            print('Evaluating best model...')
            predicted, scores = best_model.predict(test=True)
            actual = test_df['label']

            initial_result = evaluate_classifier(actual, predicted, scores)
            results = evaluate_classifiers(predictive_models,actual)
            # plot_model_comparison(results)
            print(f'Evaluation: {initial_result}')

            # CReate a DataFrame with the results
            # Extract label distributions
            label_0_count = label_distribution_train.get(0, 0)  # Get count for label 0, default to 0 if not present
            label_1_count = label_distribution_train.get(1, 0)  # Get count for label 1, default to 0 if not present

            initial_result_df = pd.DataFrame([initial_result])
            initial_result_df.insert(0, 'prefix_length', prefix_len)
            initial_result_df.insert(1, 'encoding', encoding.value)
            initial_result_df.insert(2, 'label_0_train', label_0_count)
            initial_result_df.insert(3, 'label_1_train', label_1_count)
            initial_result_df.insert(4, 'model', ClassificationMethods.XGBOOST.value)
            
            list_results.append(initial_result_df)
            
            print(f"--- End of cycle ({i}) ---")
            print()

            '''
            print('Computing explanation...')
            test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]
            cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
            full_df = pd.concat([train_df, val_df, test_df])
            cf_dataset.loc[len(cf_dataset)] = 0


            cf_result = explain(CONF, best_model, encoder=encoder, df=full_df.iloc[:, 1:],
                    query_instances=test_df_correct.iloc[:, 1:],
                    method='genetic_conformance', optimization='baseline',
                    heuristic='heuristic_2', support=0.95,
                    timestamp_col_name='Complete Timestamp', # name of the timestamp column in the log
                    model_path='./experiments/process_models/process_models',
                    random_seed=CONF['seed'], adapted=False, filtering=False)

            test_df_correct = test_df[(test_df['label'] == predicted) & (test_df['label'] == 0)]
            cf_dataset = pd.concat([train_df, val_df], ignore_index=True)
            full_df = pd.concat([train_df, val_df, test_df])


            cf_result = explain(CONF, best_model, encoder=encoder, df=train_df.iloc[:, 1:],
                    query_instances=test_df_correct,target_trace_id=test_df_correct.iloc[0,0],
                    method='genetic_conformance', optimization='baseline',
                    heuristic='heuristic_2', support=0.95,
                    timestamp_col_name='Complete Timestamp', # name of the timestamp column in the log
                    model_path='./experiments/process_models/process_models',
                    random_seed=CONF['seed'], adapted=False, filtering=False)
            counterfactuals = cf_results.cf_examples_list[0].final_cfs_df.copy()

            encoder.decode(counterfactuals)

            encoder.decode(test_df_correct)

            cf_results.cf_examples_list[0].final_cfs_df = counterfactuals
            cf_results.cf_examples_list[0].final_cfs_df_sparse = counterfactuals
            cf_results.cf_examples_list[0].test_instance_df = test_df_correct.iloc[:1,1:].copy()
            print(cf_results.visualize_as_dataframe())
            print(cf_results.visualize_as_dataframe(show_only_changes=True))
            exp = explain(CONF, best_model, encoder=encoder,df=train_df, test_df=test_df, target_trace_id=test_df_correct.iloc[0,0])
            import shap
            shap.plots.bar(exp[0])



            exp = explain(CONF, best_model, encoder=encoder,df=train_df, test_df=test_df)
            shap.plots.bar(exp)
            '''

    print("Pipeline completed successfully")
    print()

    ### SAVE RESULTS ###
    print('Saving results...')
    results_df = pd.concat(list_results, ignore_index=True)
    path_results = Path(RESULTS_DIR) / f'{Path(INPUT_LOG).stem}_{CONF["predictive_models"]}_results.csv'
    print(f'Saving results to: {path_results}')
    results_df.to_csv(path_results, index=False)
    print()

    end_time = datetime.now().replace(microsecond=0)
    print("End process:", str(end_time))
    print()

    print()
    print("*** PROGRAM END ***")
    print()