# NOTE: This script is not quite resulting in the same validation process that is performed at the very end of training, leading to some variations in results.
# There are some parameters from training that are not being passed to the validation function here, such as the seed and a few others, which are likely resulting
# in different validation results. This is not so much a problem for performance benchmarking, but it may be a problem for consistency in reported metrics.
# As such, the results reported automatically at the end of training should be considered the most accurate and consistent results as far as classification and detection
# metrics are concerned as variations my be slight, or potentially more significant.

import yaml
from argparse import ArgumentParser
from pathlib import Path
from ultralytics import YOLO
from yolov5.val import run as val_yolov5

from yolo_dataset_cfg.dataset_util import check_dataset


IMG_SZ = 960 # 930x930 is the most common resolution of our train images, but we need imgsz to be a multiple of the batch size


def main():
    parser = ArgumentParser(description='Validate a trained YOLOv8 or YOLOv5 model')
    parser.add_argument('-d', '--dataset', type=str, help='The name of the dataset directory, in the root of the repository, to validate the model on. Defaults to the dataset used for training if not specified')
    parser.add_argument('-m', '--model', type=str, default='yolov8n', help='The YOLO model to validate. Can be a path to a custom model or a name of one of the built-in models. Default is yolov8n')
    parser.add_argument('-b', '--benchmark', action='store_true', help='Run the validation in benchmark mode, which will set the batch size to 1')
    parser.add_argument('-n', '--run-name', type=str, default=None, help='The name of the run to use for outputs in the project directory')
    parser.add_argument('-p', '--project', type=str, default=None, help='The name of the project directory under runs to use for outputs')
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='Overwrite the project/run name directory if it already exists rather than appending a number to the end of the name')
    parser.add_argument('-v', '--model-version', type=int, default=None, help='Use to specify the YOLO version of the model. Required if it cannot be inferred based on the model name', choices=[5, 8])
    parser.add_argument('--plots', action='store_true', help='Generate plots and visuals of validation results')
    parser.add_argument('--device', type=str, default='0', help='The GPU device or devices, given as a comma-separated list, to use for validation. Default is 0')

    args = parser.parse_args()

    dataset_config = check_dataset(args.dataset) if args.dataset is not None else None
    
    model : str = args.model
    model_version : int | None = args.model_version
    
    if model_version is None:
        if model.startswith('yolov5'):
            model_version = 5
        elif model.startswith('yolov8'):
            model_version = 8
        else:
            raise ValueError('Model version must be specified with --model-version: Cannot infer model version from model name.')
    
    if not model.endswith('.pt'):
        model = f'{model}.pt'
    
    yolo_args = {} if dataset_config is None else {'data': dataset_config}
    
    if args.benchmark:
        yolo_args['batch_size'] = 1

    if model_version == 8:
        if 'batch_size' in yolo_args:
            yolo_args['batch'] = yolo_args['batch_size']
            yolo_args.pop('batch_size')
        yolo = YOLO(model=model, task='detect')
        yolo.val(device=args.device,
                 imgsz=IMG_SZ,
                 name=args.run_name,
                 project=None if args.project is None else str(Path('runs') / args.project),
                 exist_ok=args.force_overwrite,
                 plots=args.plots,
                **yolo_args)
    elif model_version == 5:
        weights_path = Path(model).resolve()
        if weights_path.parent.name == 'weights':
            try:
                opts = yaml.load((weights_path.parents[1] / 'opt.yaml').read_text(), Loader=yaml.SafeLoader)
                yolo_args['data'] = yolo_args['data'] if 'data' in yolo_args else check_dataset(Path(opts['data']).stem)
                yolo_args['batch_size'] = opts['batch_size'] if 'batch_size' not in yolo_args else yolo_args['batch_size']
            except FileNotFoundError:
                pass
        
        if not 'data' in yolo_args:
            raise ValueError('Could not infer dataset from weights path. Please specify the dataset with --dataset')
        
        val_yolov5(weights=model,
                   imgsz=IMG_SZ,
                   device=args.device,
                   project='runs/detect' if args.project is None else str(Path('runs') / args.project),
                   name=args.run_name if args.run_name is not None else 'val',
                   exist_ok=args.force_overwrite,
                   plots=args.plots,
                   **yolo_args)
    else:
        # Should typically be caught by the argument parser, but just in case
        raise ValueError('Model version must be 5 or 8')


if __name__ == '__main__':
    main()
