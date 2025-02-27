from ultralytics import YOLO
import pickle
from mlflow.models import infer_signature
import mlflow
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
import yaml
import shutil
from mlflow.exceptions import MlflowExceptions 

params = yaml.safe_load(open('params.yaml'))['train']
# params = yaml.safe_load(open('params.yaml'))['preprocess']


            
            
def train_model(model_path, epochs, batch):
    with mlflow.start_run():
        mlflow.log_param("dataset", "COCO")
        mlflow.log_param("model_used", model_path)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("img_size", 640) 
        mlflow.log_param("batch_size", batch)
        
        model = YOLO(model_path)
        model.train(data=coco128.yaml, epochs=epochs, imgsz=640, batch=batch)


   

if __name__ == "__main__":
    print("Script is executing..........")
    results = train_model(params['model'], 50, 16) 
    print("results: ",results)

    
    
    
    
    
    
    
    
    
    
    
    
    
    

