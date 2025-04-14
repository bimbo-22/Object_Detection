import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import mlflow
import optuna
import yaml
from ssd_dataset import SSDDataset, train_transform, val_transform
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from functools import partial
import os
import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Load SSD training parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['SSD']['train']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(batch):
    return tuple(zip(*batch))

def log_images(images, targets, preds, prefix, num_images=3):
    """Utility to log images with ground truth and predictions"""
    for i in range(min(num_images, len(images))):
        # Convert tensor to numpy (CHW -> HWC)
        img = images[i].cpu().numpy().transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)
        # Ground truth bounding boxes
        gt_boxes = targets[i]['boxes'].cpu().numpy()
        gt_labels = targets[i]['labels'].cpu().numpy()
        gt_img = img.copy()
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(gt_img, f"Class {label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        # Predicted bounding boxes (if available)
        if preds is not None and i < len(preds):
            pred_boxes = preds[i]['boxes'].cpu().numpy()
            pred_labels = preds[i]['labels'].cpu().numpy()
            pred_scores = preds[i]['scores'].cpu().numpy()
            pred_img = img.copy()
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(pred_img, f"Class {label}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        else:
            pred_img = gt_img.copy()
        gt_path = f"{prefix}_gt_{i}.png"
        pred_path = f"{prefix}_pred_{i}.png"
        combined_path = f"{prefix}_combined_{i}.png"
        cv2.imwrite(gt_path, cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(pred_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(combined_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        mlflow.log_artifact(gt_path, artifact_path=f"{prefix}_images")
        mlflow.log_artifact(pred_path, artifact_path=f"{prefix}_images")
        mlflow.log_artifact(combined_path, artifact_path=f"{prefix}_images")

def objective(trial):
    # Hyperparameters search space for this trial
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 10, 50, step=10)
    momentum = trial.suggest_float("momentum", 0.85, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])

    with mlflow.start_run(run_name=f"SSD_Optuna_Trial_{trial.number}"):
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("optimizer", optimizer_name)
        mlflow.log_param("model_name", "ssdlite320_mobilenet_v3_large")
        mlflow.log_param("num_classes", params["num_classes"])
        mlflow.log_param("data_yaml", params["data"])
        mlflow.log_param("device", device)

        # Build model without default weights to guarantee consistency.
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=params["num_classes"],
            norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03)
        )
        # Load the fine-tuned checkpoint from training.
        fine_tuned_path = "models/ssd/best_ssd_model.pth"
        model.load_state_dict(torch.load(fine_tuned_path, map_location=device))
        model.to(device)
        
        # Prepare datasets and loaders.
        train_dataset = SSDDataset(params["data"], split='train', transform=train_transform)
        val_dataset = SSDDataset(params["data"], split='val', transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=16)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=16)
        
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        total_batches = len(train_loader)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            avg_loss = total_loss / total_batches
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
        model.eval()
        metric = MeanAveragePrecision(max_detection_thresholds=[1, 100, 500])
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)
                preds = [{'boxes': o['boxes'].cpu(), 'scores': o['scores'].cpu(), 'labels': o['labels'].cpu()} for o in outputs]
                targets = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets]
                if i == 0:
                    log_images(images, targets, preds, "val_final", num_images=3)
                metric.update(preds, targets)
            map_dict = metric.compute()
            mAP = map_dict['map'].item()
            mAP50 = map_dict['map_50'].item()
        mlflow.log_metric("mAP", mAP)
        mlflow.log_metric("mAP50", mAP50)
        trial_model_path = f"models/ssd/trial_{trial.number}_model.pth"
        os.makedirs(os.path.dirname(trial_model_path), exist_ok=True)
        torch.save(model.state_dict(), trial_model_path)
        mlflow.log_artifact(trial_model_path)
    return mAP50

if __name__ == "__main__":
    mlflow.set_experiment("SSD_Optimization_Experiment")
    study = optuna.create_study(
        study_name="ssd_optimization",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=10)
    with mlflow.start_run(run_name="SSD_Optuna_Best"):
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial
        mlflow.log_params(best_params)
        mlflow.log_metric("best_mAP50", best_value)
        best_params_file = "results/best_ssd_params.yaml"
        os.makedirs("results", exist_ok=True)
        with open(best_params_file, "w") as f:
            yaml.dump(best_params, f)
        mlflow.log_artifact(best_params_file)
        best_model_path = "models/ssd/best_optimized_ssd_model.pth"
        trial_model_path = f"models/ssd/trial_{best_trial.number}_model.pth"
        if os.path.exists(trial_model_path):
            model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
            in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head.classification_head = SSDLiteClassificationHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=params["num_classes"],
                norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03)
            )
            model.load_state_dict(torch.load(trial_model_path, map_location=device))
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(best_model_path)
    print("Best parameters:", study.best_params)
    print("Best mAP50:", study.best_value)
