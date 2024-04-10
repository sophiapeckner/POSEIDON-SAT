import re
from argparse import ArgumentParser
from pathlib import Path
from ultralytics import YOLO
from yolov5.train import run as train_yolov5

from yolo_dataset_cfg.dataset_util import check_dataset
from yolov8.class_weighted_trainer import ClassWeightedDetectionTrainer


SEED = 2378110213
IMG_SZ = 960 # 930x930 is the most common resolution of our train images, but we need imgsz to be a multiple of the batch size


def main():
    parser = ArgumentParser(description='Train YOLOv8 or YOLOv5 on a POSEIDON-augmented dataset, or the original dataset')
    parser.add_argument('dataset', type=str, help='The name of the dataset directory containing the data to train on')
    parser.add_argument('-m', '--model', type=str, default='yolov8n', help='The YOLO model to train. Can be a custom model name or one of the built-in models. Default is yolov8n.')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='The number of epochs to train for. Defaults to 100')
    parser.add_argument('-c', '--use-class-weights', action='store_true', help='Weight each class to adjust for class imbalance in classification loss')
    parser.add_argument('-n', '--run-name', type=str, default=None, help='The name of the run to use for outputs in the project directory')
    parser.add_argument('-p', '--project', type=str, default=None, help='The name of the project directory under runs to use for outputs')
    parser.add_argument('-f', '--force-overwrite', action='store_true', help='Overwrite the project/run name directory if it already exists rather than appending a number to the end of the name.')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume training session from a previous run using the weights file specified with --model')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='The batch size to use for training. Set to -1 to use AutoBatch. Defaults to 32')
    parser.add_argument('-v', '--model-version', type=int, default=None, help='Use to specify the YOLO version of the model. Required if it cannot be inferred based on the model name.', choices=[5, 8])
    parser.add_argument('-d', '--device', type=str, default='0', help='The GPU device or devices, given as a comma-separated list, to use for training. Default is 0')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use pretrained yolo model weights as a starting point for training. Default is to use pretrained weights.')
    parser.add_argument('--no-training-plots', action='store_true', help='Do not generate plots and visuals of training progress')

    args = parser.parse_args()

    dataset_config = check_dataset(args.dataset)

    model : str = args.model
    model_version : int | None = args.model_version
    
    if model_version is None:
        if model.startswith('yolov5'):
            model_version = 5
        elif model.startswith('yolov8'):
            model_version = 8
        else:
            raise ValueError('Model version must be specified with --model-version: Cannot infer model version from model name.')

    model = re.sub(r'\.pt$', '', model) if model.endswith('.pt') else model
    model = re.sub(r'\.yaml$', '', model) if model.endswith('.yaml') else model
    
    model_config = f'{model}.yaml' if args.no_pretrained else f'{model}.pt'
    
    if model_version == 8:
        yolo = YOLO(model=model_config, task='detect')
        yolo.train(data=dataset_config,
                   trainer=ClassWeightedDetectionTrainer if args.use_class_weights else None,
                   device=args.device,
                   imgsz=IMG_SZ,
                   seed=SEED,
                   epochs=args.epochs,
                   batch=args.batch_size,
                   name=args.run_name,
                   project=None if args.project is None else str(Path('runs') / args.project),
                   exist_ok=args.force_overwrite,
                   resume=args.resume,
                   plots=not args.no_training_plots)
    elif model_version == 5:
        # TODO: Implement class weighting in loss function for YOLOv5
        train_yolov5(
            data=dataset_config,
            weights='' if args.no_pretrained else model_config,
            cfg=model_config if args.no_pretrained else '',
            device=args.device,
            imgsz=IMG_SZ,
            seed=SEED,
            epochs=args.epochs,
            batch_size=args.batch_size,
            project='runs/detect' if args.project is None else str(Path('runs') / args.project),
            name=args.run_name if args.run_name is not None else 'train',
            exist_ok=args.force_overwrite,
            resume=args.resume,
            noplots=args.no_training_plots,
        )
    else:
        # Should typically be caught by the argument parser, but just in case
        raise ValueError('Model version must be 5 or 8')


if __name__ == '__main__':
    main()
