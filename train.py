import argparse
import torch
import sys
import json
import boto3
import os
import logging
import traceback

from sagemaker_training import environment
from ultralytics import YOLO  # Ensure this import works in your SageMaker environment

# Initialize logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def upload_to_s3(bucket,prefix,local_path):
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_key = os.path.join(prefix, relative_path)
            s3_client.upload_file(local_file, bucket, s3_key)

# Construct the absolute path to data.yaml
data_yaml_path = os.path.join(os.path.dirname(__file__), 'data.yaml')

# List contents of /opt/ml/input/data
logging.info("Listing contents of /opt/ml/input/data/:")
try:
    for item in os.listdir('/opt/ml/input/data/'):
        logging.info(item)
except Exception as e:
    logging.error("Error listing contents of /opt/ml/input/data/: %s", e)
    traceback.print_exc()  # Print full traceback
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define your hyperparameters here
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--optimizer', help='optimizer to use')
    parser.add_argument('--lr0', type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, help='final learning rate')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--weight_decay', type=float, help='optimizer weight decay')
    parser.add_argument('--warmup_epochs', type=float, help='number of warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, help='warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, help='warmup initial bias learning rate')
    parser.add_argument('--box', type=float, help='box loss gain')
    parser.add_argument('--cls', type=float, help='class loss gain')
    parser.add_argument('--hsv_h', type=float, help='HSV Hue augmentation')
    parser.add_argument('--hsv_s', type=float, help='HSV Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, help='HSV Value augmentation')
    parser.add_argument('--degrees', type=float, help='rotation degrees')
    parser.add_argument('--translate', type=float, help='translation factor')
    parser.add_argument('--scale', type=float, help='scaling factor')
    parser.add_argument('--shear', type=float, help='shearing degrees')
    parser.add_argument('--perspective', type=float, help='perspective distortion')
    parser.add_argument('--flipud', type=float, help='probability of flipping the image upside down')
    parser.add_argument('--fliplr', type=float, help='probability of flipping the image left-right')
    parser.add_argument('--mosaic', type=float, help='mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, help='mixup augmentation probability')
    parser.add_argument('--copy_paste', type=float, help='copy-paste augmentation probability')
    parser.add_argument('--auto_augment', help='auto augmentation policy')
    parser.add_argument('--job_name', type=str, help='Name of the hyperparameter tuning job')
    parser.add_argument('--sm-hps', type=json.loads, default=os.environ['SM_HPS'])

    args = parser.parse_args()

    # Check if any required training parameters are missing
    required_params = [
        args.epochs, args.optimizer, args.lr0, args.lrf, args.momentum,
        args.weight_decay, args.warmup_epochs, args.warmup_momentum, args.warmup_bias_lr,
        args.box, args.cls, args.hsv_h, args.hsv_s, args.hsv_v, args.degrees,
        args.translate, args.scale, args.shear, args.perspective, args.flipud,
        args.fliplr, args.mosaic, args.mixup, args.copy_paste, args.auto_augment]
    if any(param is None for param in required_params):
        logging.error("One or more required training parameters are missing.")
        sys.exit(1)

    logging.info('Training parameters: %s', vars(args))

    try:
        # Assume YOLO is a class from your YOLO module that has a train method
        model = YOLO("yolov8l-seg.pt")  # Ensure this model file is accessible
        results = model.train(
            data=data_yaml_path, 
            epochs=args.epochs,
            batch=16,  # You might want to also make this an argument
            single_cls=True,
            patience=200, #change it later right now testing
            cos_lr=True,
            nbs=16,
            plots=True,
            optimizer=args.optimizer, 
            lr0=args.lr0, 
            lrf=args.lrf, 
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            warmup_momentum=args.warmup_momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            box=args.box,
            cls=args.cls,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            auto_augment=args.auto_augment,
        )

        # Access metrics directly if results is an object with attributes
        print(dir(results))
        if hasattr(results, 'results_dict'):
        # Explore the contents of results_dict
            print(results.results_dict)
        # Extract specific metrics from results_dict

        # if hasattr(results, 'maps'):
        #     # Assuming maps hold mAP scores
        #     mAP_scores = results.maps
        #     print(mAP_scores)
        env = environment.Environment()
        job_name = env.job_name
        tuning_job_name = args.job_name
        if hasattr(results, 'results_dict'):
            precision = results.results_dict.get('metrics/precision(B)', None)
            recall = results.results_dict.get('metrics/recall(B)', None)
            mAP50 = results.results_dict.get('metrics/mAP50(B)', None)
            mAP50_95 = results.results_dict.get('metrics/mAP50-95(B)', None)

        if precision is not None:
            print(f"YOLO Metric metrics/precision(B): {precision:.4f}")
        if recall is not None:
            print(f"YOLO Metric metrics/recall(B): {recall:.4f}")
        if mAP50 is not None:
            print(f"YOLO Metric metrics/mAP50(B): {mAP50:.4f}")
        if mAP50_95 is not None:
            print(f"YOLO Metric metrics/mAP50-95(B): {mAP50_95:.4f}")


        # Print metrics in the format expected by SageMaker
        # if precision is not None:
        #     print(f"YOLO Metric metrics/precision(B): {precision:.4f}")
        # if recall is not None:
        #     print(f"YOLO Metric metrics/recall(B): {recall:.4f}")
        # if mAP50 is not None:
        #     print(f"YOLO Metric metrics/mAP50(B): {mAP50:.4f}")
        # if mAP50_95 is not None:
        #     print(f"YOLO Metric metrics/mAP50-95(B): {mAP50_95:.4f}")
        # if box_loss is not None:
        #     print(f"YOLO Metric val/box_loss: {box_loss:.4f}")
        # if cls_loss is not None:
        #     print(f"YOLO Metric val/cls_loss: {cls_loss:.4f}")
        # if dfl_loss is not None:
        #     print(f"YOLO Metric val/dfl_loss: {dfl_loss:.4f}")
        
        # Save the model to the output directory
        output_model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'model.pt')
        torch.save(model.state_dict(), output_model_path)

        # Upload plots to S3 bucket
        # Get the training job name from the environment variable
        # Get the training job name from the environment variable
        training_job_name = os.environ.get('SM_TRAINING_JOB_NAME')

        # If the training job name is not found, then try to get the hyperparameter tuning job name
        if not training_job_name:
            training_job_name = os.environ.get('SM_HP_JOB_NAME', job_name)

        # Path where YOLO saves plots
        plot_path = 'runs/segment/train'
        # Unique S3 path for each training run
        s3_bucket = 'hyperparameter-boundaries'
        s3_prefix = f'runs/{tuning_job_name}/{training_job_name}'
        upload_to_s3(s3_bucket, s3_prefix, plot_path)


    except Exception as e:
        logging.error("Error during model training: %s", e)
        traceback.print_exc()  # Print full traceback
        sys.exit(1)
