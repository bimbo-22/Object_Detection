import mlflow
import yaml 
import ultralytics


def evaluate(model_path,data_path):
    
    with mlflow.start_run(run_name="Evaluation on CCTV datase"):
        model = YOLO(model_path)
        model.evaluate(data=data_path,)
    