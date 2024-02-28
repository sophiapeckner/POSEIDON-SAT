import argparse
from pathlib import Path
from ultralytics import YOLO

from yolo.dataset_util import check_dataset


SEED = 2378110213


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on a POSEIDON-augmented dataset, or the original dataset')
    parser.add_argument('dataset', type=str, help='The name of the dataset directory containing the data to train on')
    parser.add_argument('-m', '--model', type=str, default='yolov8n', help='The YOLO model to train. Can be a custom model name or one of the built-in models. Default is yolov8n.')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='The number of epochs to train for. Defaults to 300')
    parser.add_argument('-n', '--run-name', type=str, default='default-project', help='The name of the run to use for outputs')
    parser.add_argument('-p', '--project', type=str, default=None, help='The name of the project directory to use for outputs')
    parser.add_argument('-r', '--resume', action='store_true', help='Resume training session from a previous run using the weights file specified with --model')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='The batch size to use for training. Set to -1 to use AutoBatch. Defaults to 16')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not use pretrained yolo model weights as a starting point for training. Default is to use pretrained weights.')
    parser.add_argument('--no-training-plots', action='store_true', help='Do not generate plots and visuals of training progress')
    
    args = parser.parse_args()
    
    dataset_config = check_dataset(args.dataset)
    
    model_config = f'{args.model}.yaml' if args.no_pretrained else f'{args.model}.pt'
    
    yolo = YOLO(model=model_config, task='detect')
    yolo.train(data=dataset_config,
               imgsz=930,
               seed=SEED,
               epochs=args.epochs,
               batch=args.batch_size,
               name=args.run_name,
               project=str(Path('runs') / args.project),
               resume=args.resume,
               plots=not args.no_training_plots)
    

if __name__ == '__main__':
    main()
