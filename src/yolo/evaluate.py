import argparse
import torch
from ultralytics import YOLO
import mlflow
from urllib.parse import urlparse
import os
import yaml
import shutil
from dotenv import load_dotenv
import time  # For inference time
import numpy as np  # For array handling

load_dotenv()

params = yaml.safe_load(open('params.yaml'))['YOLO']
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

def evaluate(model_path, data_path):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    with mlflow.start_run(run_name="Evaluating on Held-Out Test Set (v8m) v3"):
        # Load model
        fine_tuned_model_path = params['optimized_model']
        model = YOLO(fine_tuned_model_path)
        
        # Log basic info
        mlflow.log_param("model_used", model_path)
        mlflow.log_param("dataset", data_path)
        mlflow.log_param("num_classes", len(model.names))
        
        # Load and log test set details
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)
        test_images = data_config.get('test', [])
        if not test_images:
            raise ValueError("No test set defined in data YAML!")
        mlflow.log_param("num_test_images", len(test_images))
        
        # Evaluate on test set
        results = model.val(data=data_path, split='test')  # Specify test split
        
        # Log core metrics (averages for "all" class)
        mlflow.log_metric("mAP50", results.box.map50)  # 0.914
        mlflow.log_metric("mAP50-95", results.box.map)  # 0.792
        mlflow.log_metric("precision", results.box.p[results.box.p != 0].mean())  # Compute mean for "all" class
        mlflow.log_metric("recall", results.box.r[results.box.r != 0].mean())  # Compute mean for "all" class
        
        # Log class-wise metrics
        class_names = model.names
        for i, name in enumerate(class_names):
            mlflow.log_metric(f"mAP50_{name}", results.box.ap50[i])
            mlflow.log_metric(f"mAP50-95_{name}", results.box.ap[i])
            mlflow.log_metric(f"precision_{name}", results.box.p[i])
            mlflow.log_metric(f"recall_{name}", results.box.r[i])
        
        # Measure inference time
        total_time = 0
        for img in test_images[:-1]:  # Test on a few images
            start = time.time()
            model.predict(source=img, save=False)
            total_time += time.time() - start
        avg_inference_time = total_time / min(10, len(test_images))
        mlflow.log_metric("avg_inference_time_ms", avg_inference_time * 1000)
        
        # Save sample predictions
        sample_images = test_images[:5]
        for img in sample_images:
            model.predict(source=img, save=True, save_dir='predictions')
        shutil.make_archive('predictions', 'zip', 'predictions')
        mlflow.log_artifact('predictions.zip')
        
        # Write results to file
        with open("results.txt", "w") as f:
            f.write(f"mAP50: {results.box.map50}\n")
            f.write(f"mAP50-95: {results.box.map}\n")
            f.write(f"Precision: {results.box.p[results.box.p != 0].mean()}\n")
            f.write(f"Recall: {results.box.r[results.box.r != 0].mean()}\n")
            f.write(f"Avg Inference Time (ms): {avg_inference_time * 1000:.2f}\n")
        
        print(f"Evaluation completed. Results saved to results.txt and MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data.yaml with test set")
    args = parser.parse_args()
    print(f"Evaluating model: {args.model_path} on test set: {args.data_path}")
    evaluate(args.model_path, args.data_path)