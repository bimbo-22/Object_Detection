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

            
            
def train_model(model_path, epochs, batch):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(run_name= "Training v8m on COCO Dataset"):
        mlflow.log_param("dataset", "COCO")
        mlflow.log_param("model_used", model_path)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("img_size", 640) 
        mlflow.log_param("batch_size", batch)
        
        model = YOLO(model_path)
        best_model = model.train(data="coco128.yaml", epochs=epochs, imgsz=640, batch=batch)
        
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
                registered_model = mlflow.register_model(model_uri, "COCO Trained Model")
                print("Model registered: ", registered_model)
            except mlflow.MlflowException as e:
                print(f"Model registration failed: {e}")
        else:
            mlflow.register_model(best_model, "COCO Trained Model")
            print("Model not registered as tracking uri is file")


if __name__ == "__main__":
    print("Script is executing..........")
    results = train_model(params['model'], 50, 16) 
    print("results: ",results)

    
    
    
    
    
    
    
    
    
    
    
    
    
    

