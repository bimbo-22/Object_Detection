import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import mlflow
import optuna
import yaml
from ssd_dataset import SSDDataset, train_transform, val_transform
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
from functools import partial
import torchmetrics
import cv2
import numpy as np

print(f"Using torchmetrics from: {torchmetrics.__file__}")
print(f"torchmetrics version: {torchmetrics.__version__}")

from dotenv import load_dotenv

load_dotenv()
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(mlflow_tracking_uri)

params = yaml.safe_load(open('params.yaml'))['SSD']['train']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collate_fn(batch):
    return tuple(zip(*batch))

def draw_boxes(image, boxes, labels, scores=None, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on an image with optional scores."""
    img = image.copy()
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label_text = f"Class {label}"
        if scores is not None:
            label_text += f": {scores[i]:.2f}"
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img

def log_images(images, targets, preds, prefix, num_images=3):
    """Log images with ground truth and predicted boxes to MLflow."""
    for i in range(min(num_images, len(images))):
        # Convert tensor to numpy (CHW -> HWC, denormalize)
        img = images[i].cpu().numpy().transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)

        # Ground truth boxes
        gt_boxes = targets[i]['boxes'].cpu().numpy()
        gt_labels = targets[i]['labels'].cpu().numpy()
        gt_img = draw_boxes(img, gt_boxes, gt_labels, color=(0, 255, 0))  # Green for GT

        # Predicted boxes 
        if preds and i < len(preds):
            pred_boxes = preds[i]['boxes'].cpu().numpy()
            pred_labels = preds[i]['labels'].cpu().numpy()
            pred_scores = preds[i]['scores'].cpu().numpy()
            pred_img = draw_boxes(img, pred_boxes, pred_labels, pred_scores, color=(255, 0, 0))  # Red for preds
            # Combine GT and preds on one image
            combined_img = draw_boxes(gt_img, pred_boxes, pred_labels, pred_scores, color=(255, 0, 0))
        else:
            combined_img = gt_img

        # Save and log images
        gt_path = f"{prefix}_gt_{i}.png"
        pred_path = f"{prefix}_pred_{i}.png" if preds else None
        combined_path = f"{prefix}_combined_{i}.png"
        
        cv2.imwrite(gt_path, cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
        if pred_path:
            cv2.imwrite(pred_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(combined_path, cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        
        mlflow.log_artifact(gt_path, f"{prefix}_images")
        if pred_path:
            mlflow.log_artifact(pred_path, f"{prefix}_images")
        mlflow.log_artifact(combined_path, f"{prefix}_images")

def objective(trial):
   
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 10, 50, step=10)
    momentum = trial.suggest_float("momentum", 0.85, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["SGD",  "AdamW"])

    with mlflow.start_run(run_name=f"SSD_Optuna_Trial_{trial.number}"):
        
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("model_name", "ssdlite320_mobilenet_v3_large")
        mlflow.log_param("num_classes", params["num_classes"])
        mlflow.log_param("data_yaml", params["data"])
        mlflow.log_param("device", device)
        mlflow.log_param("train_transform", str(train_transform))
        mlflow.log_param("val_transform", str(val_transform))
        mlflow.log_param("norm_layer_eps", 1e-3)
        mlflow.log_param("norm_layer_momentum", 0.03)

        # Model setup
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=params["num_classes"],
            norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03)
        )
        model.to(device)

        # Dataset and DataLoader
        train_dataset = SSDDataset(params["data"], split='train', transform=train_transform)
        val_dataset = SSDDataset(params["data"], split='val', transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Log dataset sizes
        mlflow.log_param("train_dataset_size", len(train_dataset))
        mlflow.log_param("val_dataset_size", len(val_dataset))
        mlflow.log_param("num_train_batches", len(train_loader))
        mlflow.log_param("num_val_batches", len(val_loader))

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Training loop with image logging
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            if epoch == 0:  # Log images from first epoch
                for i, (images, targets) in enumerate(train_loader):
                    if i >= 1:  # Log only the first batch
                        break
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    log_images(images, targets, None, "train_epoch_0", num_images=3)
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += losses.item()
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
            else:
                for images, targets in train_loader:
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += losses.item()
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Validation with image logging
        model.eval()
        metric = MeanAveragePrecision(max_detection_thresholds=[1, 100, 500])
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)
                preds = [{'boxes': o['boxes'].cpu(), 'scores': o['scores'].cpu(), 'labels': o['labels'].cpu()} for o in outputs]
                targets = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets]
                if i == 0:  # Log images from first validation batch
                    log_images(images, targets, preds, "val_final", num_images=3)
                metric.update(preds, targets)
            map_dict = metric.compute()
            mAP50 = map_dict['map_50'].item()
            mAP = map_dict['map'].item()

        # Log additional metrics
        mlflow.log_metric("mAP50", mAP50)
        mlflow.log_metric("mAP", mAP)

    return mAP50

if __name__ == "__main__":
    # Set an experiment name in MLflow
    mlflow.set_experiment("SSD_Optimization_Experiment")

    # Create Optuna study with a pruner
    study = optuna.create_study(
        study_name="ssd_optimization",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=10)

    with mlflow.start_run(run_name="SSD_Optuna_Best"):
        best_params = study.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_mAP50", study.best_value)
        
        best_params_file = "results/best_ssd_params.yaml"
        os.makedirs("results", exist_ok=True)
        with open(best_params_file, "w") as f:
            yaml.dump(best_params, f)
        mlflow.log_artifact(best_params_file)

    print("Best parameters:", best_params)
    print("Best mAP50:", study.best_value)