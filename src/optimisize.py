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


def objective(trial):
    
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 1e-5, 1e-1, log=True)
    epochs = trial.suggest_categorical("epochs", [10, 15, 25, 50])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    imgsz = trial.suggest_categorical("imgsz", [512, 640, 768, 896])
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 5)
    momentum = trial.suggest_float("momentum", 0.85, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2)
    
        # augmentation hyperparameters
    mosaic = trial.suggest_float("mosaic", 0.0, 0.5) 
    mixup = trial.suggest_float("mixup", 0.0, 0.5)
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.05) 
    hsv_s = trial.suggest_float("hsv_s", 0.0, 0.3)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 0.3)
    
    model = YOLO(params['model'])

    for param in model.model.parameters():
        param.requires_grad = True
    
    results = model.train(
        data = params['data'],
        epochs = epochs,
        batch = batch_size,
        imgsz = imgsz,
        lr0 = lr0,
        lrf = lrf,
        optimizer = optimizer,
        warmup_epochs = warmup_epochs,
        momentum = momentum,
        weight_decay = weight_decay,
        mosaic = mosaic,
        mixup = mixup,
        hsv_h = hsv_h,
        hsv_s = hsv_s,
        hsv_v = hsv_v,
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        project = "optimizing_cctv_model",
        name = f'trial_{trial.number}',
        save = True,
        exist_ok = True
    )
        
    mAP50 = results.box.map50
    return mAP50
    
def optimize(n_trials=20):
    study = optuna.create_study(direction="maximize", study_name="yolov8_hyperparameter_optimization",
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    
    mlflow_callback =  MLflowCallback(
        tracking_uri = mlflow_tracking_uri,
        metric_name = "mAP50",
    )
    
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_callback])
    
    print("Best trial:", study.best_trial.number)
    print("Best parameters:", study.best_params)
    print("Best mAP50:", study.best_value)
    
    return  study.best_params

if __name__ == "__main__":
    print("Script is executing..........")
    best_params = optimize(n_trials=20)
    print("Best hyperparameters: ", best_params)


