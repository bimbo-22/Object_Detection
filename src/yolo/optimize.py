from ultralytics import YOLO
import mlflow
import os
from dotenv import load_dotenv
import yaml
import optuna
import torch

load_dotenv()

params = yaml.safe_load(open('params.yaml'))["YOLO"]['train']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_tracking_uri)

def objective(trial):
    # Training hyperparameters
    lr0 = trial.suggest_float("lr0", 0.0003, 0.0007, log=True)
    lrf = trial.suggest_float("lrf", 0.00005, 0.001, log=True)
    epochs = trial.suggest_categorical("epochs", [50, 100])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    optimizer = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    imgsz = trial.suggest_categorical("imgsz", [640])
    warmup_epochs = trial.suggest_int("warmup_epochs", 3, 5)
    momentum = trial.suggest_float("momentum", 0.9, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 0.0005, 0.001)
    freeze = trial.suggest_int("freeze", 0, 10)  # Add layer freezing

    # Enhanced augmentation hyperparameters
    mosaic = trial.suggest_float("mosaic", 0.5, 1.0)
    mixup = trial.suggest_float("mixup", 0.5, 1.0)
    hsv_h = trial.suggest_float("hsv_h", 0.015, 0.05)
    hsv_s = trial.suggest_float("hsv_s", 0.2, 0.5)
    hsv_v = trial.suggest_float("hsv_v", 0.2, 0.5)
    degrees = trial.suggest_float("degrees", 0.0, 15.0)  # Add rotation
    translate = trial.suggest_float("translate", 0.0, 0.2)  # Add translation

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
        "freeze": freeze,
        "mosaic": mosaic,
        "mixup": mixup,
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "translate": translate,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "project": "optimizing_cctv_model_v2",
        "name": f'trial_{trial.number}',
        "save": True,
        "exist_ok": True
    }

    # Load fine-tuned model
    fine_tuned_model_path = params['fine_tuned_model']
    model = YOLO(fine_tuned_model_path)

    # Train the model
    results = model.train(**train_params)

    mAP50 = results.box.map50
    precision = results.box.p
    recall = results.box.r
    f1_score = results.box.f1
    iou = results.box.iou  # Assuming IoU is available

    # Log metrics to MLflow
    mlflow.log_metric("mAP50", mAP50)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("iou", iou)

    # Combined objective: balance mAP and recall
    objective_value = 0.7 * mAP50 + 0.3 * recall
    return objective_value

if __name__ == "__main__":
    experiment_name = "optimizing_cctv_model_v2"
    study = optuna.create_study(direction="maximize", study_name=experiment_name, pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    with open("results/best_yolo_params.yaml", "w") as f:
        yaml.dump(best_params, f)
    print("Best parameters:", best_params)