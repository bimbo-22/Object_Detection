import torch
from ultralytics import YOLO
import mlflow
from dotenv import load_dotenv
from urllib.parse import urlparse
import os

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")


def fine_tune(data_path, model_path, epochs, batch):
    with mlflow.start_run(run_name="Fine-tuning on custom dataset"):
        mlflow.log_param("dataset", "unified_dataset")
        mlflow.log_param("model_used", model_path)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("img_size", 640)
        mlflow.log_param("batch_size", batch)

    model = YOLO(model_path)

    for name, param in model.model.named_parameters():
        if "backbone" in name or "neck" in name:
            param.requires_grad = False

    print("Backbone and neck layers frozen. Training only the detection head.")

    train_params = {
        "data": data_path,
        "epochs": epochs,
        "batch": batch,
        "imgsz": 640,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr0": 0.01,
        "optimizer": "SGD",
    }

    best_model = model.train(**train_params)
    
    mlflow.log_metric("mAP", best_model.metrics.mAP50)
    mlflow.log_metric("mAP50-95", best_model.metrics.mAP50_95)
    mlflow.log_metric("Precision", best_model.metrics.precision)
    mlflow.log_metric("Recall", best_model.metrics.recall)
    mlflow.log_metric("F1", best_model.results["F1"])
    mlflow.log_metric("val_loss", best_model.results["val_loss"])
    mlflow.log_metric("train_loss", best_model.results["train_loss"])

    best_model.save(model_path)
    mlflow.log_artifact(model_path, artifact_path="best_model")


    tracking_uri_type_store = urlparse(mlflow_tracking_uri).scheme
        
    if tracking_uri_type_store != "file":
        try:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
            registered_model = mlflow.register_model(model_uri, "fine_tuned_model")
            print("Model registered: ", registered_model)
        except mlflow.MlflowException as e:
            print(f"Model registration failed: {e}")
    else:
        mlflow.register_model(best_model, "fine_tuned_model")
        print("Model not registered as tracking uri is file")





