import argparse
import torch
from ultralytics import YOLO
import pickle
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
# params = yaml.safe_load(open('params.yaml'))['preprocess']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

            
            
def train_model(model_path,data_path, mode, config_path=None):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    run_name = f"YOLO Training ({mode}) on Custiom Dataset"
    with mlflow.start_run(run_name=run_name):
        # using default hyperparameters
        train_params = { 
            "data": data_path,
            "epochs": 50,
            "batch": 16,
            "imgsz": 640,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        } 
        
        if mode == "initial":
            print("Training with default hyperparameters")
        elif mode == "fine-tune":
            # freezing the backbone and part of neck for fine-tuning
            train_params.update({
                "freeze": 10,
                "epochs": 30
            })
            print("Layers Frozen , Fine-tuning with custom hyperparameters")
        elif mode == "optimized":
            if not config_path:
                raise ValueError("config_path is required for optimized mode")
            with open(config_path, "r") as f:
                optimized_params = yaml.safe_load(f)
            train_params.update(optimized_params)
            print("Training with optimized hyperparameters")
                
        mlflow.log_param(train_params)
        mlflow.log_param("model_used", model_path)
        mlflow.log_param("dataset", data_path)
        
        model = YOLO(model_path)
        results = model.train(**train_params)
        
        mlflow.log_metric("mAP", results.box.mAP50)
        mlflow.log_metric("mAP50-95", results.metrics.mAP50_95)
        mlflow.log_metric("Precision", results.metrics.precision)
        mlflow.log_metric("Recall", results.metrics.recall)
        mlflow.log_metric("F1", results.results["F1"])
        mlflow.log_metric("val_loss", results.results["val_loss"])
        mlflow.log_metric("train_loss", results.results["train_loss"])
        

        output_path = f"models/{mode}_model.pt"
        model.save(output_path)
        mlflow.log_artifact(output_path, artifact_path="best_model")


        tracking_uri_type_store = urlparse(mlflow_tracking_uri).scheme
        
        if tracking_uri_type_store != "file":
            try:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
                registered_model = mlflow.register_model(model_uri, f"{mode}_model")
                print("Model registered: ", registered_model)
            except mlflow.MlflowException as e:
                print(f"Model registration failed: {e}")
        else:
            print("Model not registered")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Train YOLO model in different odes")
    parser.add_argument("--mode", options=["initial", "fine-tune", "optimized"], required="True")
    parser.add_argument("--model_path", default=params['model'], help="Path to the model to be used")
    parser.add_argument("--data_path", default=params['data'], help="Path to the data to be used")
    parser.add_argument("--config_path", default=None, help="Path to the config file for optimized mode")
    args = parser.parse_args()
    
    print(f"starting training in {args.mode} mode")
    train_model(args.model_path, args.data_path, args.mode, args.config_path)

    
    
    
    
    
    
    
    
    
    
    
    
    
    

