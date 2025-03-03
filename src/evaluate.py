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

# params = yaml.safe_load(open('params.yaml'))['preprocess']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")


def evaluate(model_path,data_path):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    with mlflow.start_run(run_name="Evaluation on Unified dataset"):
        model = YOLO(model_path)
        evaluation = model.val(data=data_path)

        return evaluation
        
        
        
if __name__ == "__main__":
    print("Script is executing..........")
    results = evaluate(params['model'],params['data']) 
    print("results: ",results)