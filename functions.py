import os
def create_output_directory(dir_name: str) -> None:
    """
    Creates the results directory if it does not exist and adds a .gitkeep file.

    Args:
        dir_name (str): Path to the directory.
    """
    print('> Creating directory...')
    print('Directory name:', dir_name)
    if os.path.exists(dir_name):
        print('Directory already exists')
    else:
        os.makedirs(dir_name)
        print('Directory created')
        # Create an empty .gitkeep file
        gitkeep_path = os.path.join(dir_name, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            pass
        print('.gitkeep file created')
    print()