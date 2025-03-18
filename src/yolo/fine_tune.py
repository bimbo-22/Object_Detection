import torch
from ultralytics import YOLO
import mlflow
from dotenv import load_dotenv
from urllib.parse import urlparse
import os
import yaml


params = yaml.safe_load(open('params.yaml'))['train']


load_dotenv()
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")


mlflow.set_tracking_uri(mlflow_tracking_uri)

def fine_tune(data_path, model_path, epochs, batch):
    with mlflow.start_run(run_name="Fine-tuning on custom dataset (v8m)"):
        mlflow.log_param("dataset", "unified_dataset")
        mlflow.log_param("model_used", model_path)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("img_size", 640)
        mlflow.log_param("batch_size", batch)
    
        model = YOLO(model_path)
   
        train_params = {
            "data": data_path,
            "epochs": epochs,
            "batch": batch,
            "imgsz": 640,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lr0": 0.01,
            "optimizer": "SGD",
            "freeze": 10,  # Freezing the first 10 layers (backbone and part of neck)
        }


        print("Backbone and neck layers frozen (first 10 layers). Training now...")
        best_model = model.train(**train_params)
        print("Fine-tuning completed!")

        metrics = best_model.metrics
        mlflow.log_metric("mAP", metrics.box.map50)  
        mlflow.log_metric("mAP50-95", metrics.box.map) 
        mlflow.log_metric("Precision", metrics.box.p)
        mlflow.log_metric("Recall", metrics.box.r)

        fine_tuned_model_path = "models/fine_tuned_model.pt"
        best_model.save(fine_tuned_model_path)
        mlflow.log_artifact(fine_tuned_model_path, artifact_path="best_model")
    
        tracking_uri_type_store = urlparse(mlflow_tracking_uri).scheme
        if tracking_uri_type_store != "file":
            try:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
                registered_model = mlflow.register_model(model_uri, "fine_tuned_model")
                print("Model registered: ", registered_model)
            except mlflow.MlflowException as e:
                print(f"Model registration failed: {e}")
        else:
            print("Model not registered (tracking URI is a local file store).")

if __name__ == "__main__":
    fine_tune(params["data"], params["model"], 50, 16)