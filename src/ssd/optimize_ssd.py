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
        img = images[i].cpu().numpy().transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)

        gt_boxes = targets[i]['boxes'].cpu().numpy()
        gt_labels = targets[i]['labels'].cpu().numpy()
        gt_img = draw_boxes(img, gt_boxes, gt_labels, color=(0, 255, 0))

        if preds and i < len(preds):
            pred_boxes = preds[i]['boxes'].cpu().numpy()
            pred_labels = preds[i]['labels'].cpu().numpy()
            pred_scores = preds[i]['scores'].cpu().numpy()
            pred_img = draw_boxes(img, pred_boxes, pred_labels, pred_scores, color=(255, 0, 0))
            combined_img = draw_boxes(gt_img, pred_boxes, pred_labels, pred_scores, color=(255, 0, 0))
        else:
            combined_img = gt_img

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
    # Existing hyperparameters
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 10, 50, step=10)
    momentum = trial.suggest_float("momentum", 0.85, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    
    # New hyperparameters
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
    lr_decay_gamma = trial.suggest_float("lr_decay_gamma", 0.1, 0.5)
    lr_step_size = trial.suggest_int("lr_step_size", 5, 15)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    aug_prob = trial.suggest_float("augmentation_probability", 0.3, 0.7)

    with mlflow.start_run(run_name=f"SSD_Optuna_Trial_{trial.number}"):
        # Log all hyperparameters
        mlflow.log_params({
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "optimizer": optimizer_name,
            "warmup_epochs": warmup_epochs,
            "lr_decay_gamma": lr_decay_gamma,
            "lr_step_size": lr_step_size,
            "dropout_rate": dropout_rate,
            "augmentation_probability": aug_prob,
            "model_name": "ssdlite320_mobilenet_v3_large",
            "num_classes": params["num_classes"],
            "data_yaml": params["data"],
            "device": device,
            "train_transform": str(train_transform),
            "val_transform": str(val_transform),
            "norm_layer_eps": 1e-3,
            "norm_layer_momentum": 0.03
        })

        # Model setup with dropout in classification head
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # Custom classification head with dropout
        class CustomSSDLiteClassificationHead(SSDLiteClassificationHead):
            def __init__(self, in_channels, num_anchors, num_classes, norm_layer, dropout_rate):
                super().__init__(in_channels, num_anchors, num_classes, norm_layer)
                self.dropout = torch.nn.Dropout(p=dropout_rate)
            
            def forward(self, x):
                x = [self.dropout(xi) for xi in x]
                return super().forward(x)
        
        model.head.classification_head = CustomSSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=params["num_classes"],
            norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03),
            dropout_rate=dropout_rate
        )
        model.to(device)

        # Update transforms with dynamic augmentation probability
        dynamic_train_transform = A.Compose([
            A.Resize(320, 320),
            A.HorizontalFlip(p=aug_prob),
            A.RandomBrightnessContrast(p=aug_prob),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=aug_prob),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))

        # Dataset and DataLoader
        train_dataset = SSDDataset(params["data"], split='train', transform=dynamic_train_transform)
        val_dataset = SSDDataset(params["data"], split='val', transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Log dataset sizes
        mlflow.log_params({
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset),
            "num_train_batches": len(train_loader),
            "num_val_batches": len(val_loader)
        })

        # Optimizer setup
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        # Learning rate scheduler with warmup
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay_gamma)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )

        # Training loop with detailed loss tracking
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_cls_loss = 0
            total_bbox_loss = 0
            if epoch == 0:
                for i, (images, targets) in enumerate(train_loader):
                    if i >= 1: break
                    images = list(img.to(device) for img in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    log_images(images, targets, None, "train_epoch_0", num_images=3)
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    total_loss += losses.item()
                    total_cls_loss += loss_dict['classification'].item()
                    total_bbox_loss += loss_dict['bbox_regression'].item()
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
                    total_cls_loss += loss_dict['classification'].item()
                    total_bbox_loss += loss_dict['bbox_regression'].item()
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
            # Apply warmup and scheduler
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()

            # Log average losses per epoch
            avg_loss = total_loss / len(train_loader)
            avg_cls_loss = total_cls_loss / len(train_loader)
            avg_bbox_loss = total_bbox_loss / len(train_loader)
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_cls_loss": avg_cls_loss,
                "train_bbox_loss": avg_bbox_loss
            }, step=epoch)

        # Validation with detailed metrics
        model.eval()
        metric = MeanAveragePrecision(max_detection_thresholds=[1, 100, 500])
        val_total_loss = 0
        val_total_cls_loss = 0
        val_total_bbox_loss = 0
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                model.train()
                loss_dict = model(images, targets)
                val_losses = sum(loss for loss in loss_dict.values())
                val_total_loss += val_losses.item()
                val_total_cls_loss += loss_dict['classification'].item()
                val_total_bbox_loss += loss_dict['bbox_regression'].item()
                model.eval()
                outputs = model(images)
                preds = [{'boxes': o['boxes'].cpu(), 'scores': o['scores'].cpu(), 'labels': o['labels'].cpu()} for o in outputs]
                targets = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets]
                if i == 0:
                    log_images(images, targets, preds, "val_final", num_images=3)
                metric.update(preds, targets)
            map_dict = metric.compute()
            mAP = map_dict['map'].item()
            mAP50 = map_dict['map_50'].item()
            avg_val_loss = val_total_loss / len(val_loader)
            avg_val_cls_loss = val_total_cls_loss / len(val_loader)
            avg_val_bbox_loss = val_total_bbox_loss / len(val_loader)

        # Log all metrics
        mlflow.log_metrics({
            "mAP": mAP,
            "mAP50": mAP50,
            "val_loss": avg_val_loss,
            "val_cls_loss": avg_val_cls_loss,
            "val_bbox_loss": avg_val_bbox_loss
        })

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
        mlflow.log_params(best_params)
        mlflow.log_metric("best_mAP50", study.best_value)
        
        best_params_file = "results/best_ssd_params.yaml"
        os.makedirs("results", exist_ok=True)
        with open(best_params_file, "w") as f:
            yaml.dump(best_params, f)
        mlflow.log_artifact(best_params_file)

    print("Best parameters:", best_params)
    print("Best mAP50:", study.best_value)