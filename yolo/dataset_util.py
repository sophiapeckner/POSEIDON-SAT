import shutil
from pathlib import Path

from ultralytics.utils import DATASETS_DIR


def check_dataset(dataset_name: str):
    if not (DATASETS_DIR / dataset_name).is_dir():
        if not Path(dataset_name).is_dir():
            raise FileNotFoundError(f'Could not find dataset directory {dataset_name}')
        print('Preparing dataset for conversion to YOLO format...')
        shutil.copytree(dataset_name, DATASETS_DIR / dataset_name)
    
    config_file = Path(__file__).parent / f'{dataset_name}.yaml'
    if not config_file.is_file():
        raise FileNotFoundError(f'Could not find a YOLO dataset config file for {dataset_name} in {config_file.parent}')
    
    return config_file.resolve()
