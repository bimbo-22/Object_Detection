from ultralytics import YOLO
import mlflow
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
import yaml


load_dotenv()
params = yaml.safe_load(open('params.yaml'))['train']
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

def train_model(model_path, data_yaml, train_params):

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(run_name="Training on Combined Dataset with best params"):
        
        mlflow.log_params(train_params)
        mlflow.log_param("dataset", data_yaml)
        mlflow.log_param("model_used", model_path)
        
        
        model = YOLO(model_path)
        results = model.train(data=data_yaml, **train_params)
        
       
        mlflow.log_metric("mAP50", results.box.map50)
        mlflow.log_metric("mAP50-95", results.box.map)
        mlflow.log_metric("Precision", results.box.p)
        mlflow.log_metric("Recall", results.box.r)
        
        
        model_path = "best_model.pt"
        model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="best_model")
        
        
        tracking_uri_type_store = urlparse(mlflow_tracking_uri).scheme
        if tracking_uri_type_store != "file":
            try:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
                registered_model = mlflow.register_model(model_uri, "Combined Dataset Model")
                print("Model registered: ", registered_model)
            except mlflow.MlflowException as e:
                print(f"Model registration failed: {e}")
        else:
            print("Model not registered as tracking URI is file")

if __name__ == "__main__":
    print("Script is executing..........")
    
    
    best_params = {
        'lr0': 0.0002958159924565695,
        'lrf': 0.0009854257909620932,
        'epochs': 50,  
        'batch': 64,   
        'optimizer': 'SGD',
        'imgsz': 640,
        'warmup_epochs': 4,
        'momentum': 0.8922755763073302,
        'weight_decay': 0.0033793257099669896,
        'mosaic': 0.1261029783981223,
        'mixup': 0.13469933725660385,
        'hsv_h': 0.020334689099403707,
        'hsv_s': 0.1408971815973312,
        'hsv_v': 0.2389650117953785,
    }
    
    

    

    train_model(params['model'], params['data'], best_params)
    print("Training completed.")