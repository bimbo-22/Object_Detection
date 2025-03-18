from ultralytics import YOLO
from mlflow.models import infer_signature
import mlflow
import os
from dotenv import load_dotenv
import yaml
import optuna
import torch
from dotenv import load_dotenv
from optuna.integration import MLflowCallback
load_dotenv()

params = yaml.safe_load(open('params.yaml'))['train']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(mlflow_tracking_uri)

def objective(trial):
    
    lr0 = trial.suggest_float("lr0", 0.0003, 0.0007, log=True)
    lrf = trial.suggest_float("lrf", 0.00005, 0.001, log=True)
    epochs = trial.suggest_categorical("epochs", [50, 100])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    optimizer = trial.suggest_categorical("optimizer", ["SGD",  "AdamW"])
    imgsz = trial.suggest_categorical("imgsz", [640, 768, 896])
    warmup_epochs = trial.suggest_int("warmup_epochs", 3, 5)
    momentum = trial.suggest_float("momentum", 0.9, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 0.0005, 0.001)
    
        # augmentation hyperparameters
    mosaic = trial.suggest_float("mosaic", 0.3, 0.7)  
    mixup = trial.suggest_float("mixup", 0.3, 0.7)  
    hsv_h = trial.suggest_float("hsv_h", 0.01, 0.03)  
    hsv_s = trial.suggest_float("hsv_s", 0.1, 0.3)  
    hsv_v = trial.suggest_float("hsv_v", 0.03, 0.1)
    
    
    train_params = {
        "data": params['data'],
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "lr0": lr0,
        "lrf": lrf,
        "optimizer": optimizer,
        "warmup_epochs": warmup_epochs,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "mosaic": mosaic,
        "mixup": mixup,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "project": "optimizing_cctv_model_v2",
        "name": f'trial_{trial.number}',
        "save": True,
        "exist_ok": True
    }
    model = YOLO(params['model'])
    results = model.train(**train_params)      
    mAP50 = results.box.map50
    return mAP50
    

if __name__ == "__main__":
    experiment_name = "optimizing_cctv_model_v2"
    study = optuna.create_study(direction="maximize", study_name =experiment_name, pruner = optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    with open("results/best_params.yaml", "w") as f:
        yaml.dump(best_params, f)
    print("Best parameters: ", best_params)


