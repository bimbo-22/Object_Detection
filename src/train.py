from ultralytics import YOLO
from mlflow.models import infer_signature
import mlflow
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
import yaml

params = yaml.safe_load(open('params.yaml'))['train']
# params = yaml.safe_load(open('params.yaml'))['preprocess']




def train(data_path,model_path):
    model = YOLO(model_path)
    print("model info: ",model.info())
    results = model.train(data=data_path, epochs=1, imgsz=640)
    return results
    

if __name__ == "__main__":
    print("Script is executing..........")
    results = train(params['data'],params['model']) 
    print("results: ",results)

    
