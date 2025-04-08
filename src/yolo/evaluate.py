import argparse
import torch
from ultralytics import YOLO
from mlflow.models import infer_signature
import mlflow
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
import yaml
import shutil
from dotenv import load_dotenv
load_dotenv()


params = yaml.safe_load(open('params.yaml'))['train']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")


def evaluate(model_path,data_path):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    with mlflow.start_run(run_name="Evaluating on Unified dataset (v8m)"):
        model = YOLO(model_path)
        results = model.val(data=data_path)

        mlflow.log_param("model_used", model_path)
        mlflow.log_param("dataset", data_path)
        print(f"mAP50: {results.box.map50}")
        print(f"mAP50-95: {results.box.map}")
        print(f"Precision: {results.box.p}")
        print(f"Recall: {results.box.r}")
        
    with open("results.txt", "w") as f:
        f.write(f"mAP50: {results.box.map50}\n")
        f.write(f"mAP50-95: {results.box.map}\n")
        f.write(f"Precision: {results.box.p}\n")
        f.write(f"Recall: {results.box.r}\n")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_path", type=str,required=True, help="Path to data")
    args = parser.parse_args()
    print(f"Evaluating model: {args.model_path} on data: {args.data_path}")
    evaluate(args.model, args.data)