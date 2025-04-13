import argparse
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import mlflow
from urllib.parse import urlparse
import os
import yaml
from ssd_dataset import SSDDataset, train_transform, val_transform
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from functools import partial
import torchmetrics
import time

print(f"Using torchmetrics from: {torchmetrics.__file__}")
print(f"torchmetrics version: {torchmetrics.__version__}")

from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

params = yaml.safe_load(open('params.yaml'))['SSD']['train']

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(data_yaml, model_name, num_classes, mode, config_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    run_name = f"SSD Training ({mode}) on Custom Dataset"
    with mlflow.start_run(run_name=run_name):
        train_params = {
            "epochs": params.get('epochs', 50),
            "batch_size": params.get('batch', 16),
            "lr": params.get('lr', 0.001),
            "optimizer": "SGD",  # Default optimizer
            "momentum": 0.9,  # Default for SGD
            "weight_decay": 0.0005  # Default
        }
        if mode == "fine-tune":
            train_params["freeze_backbone"] = True
            train_params["epochs"] = 30
        elif mode == "optimized" and config_path:
            with open(config_path, 'r') as f:
                optimized_params = yaml.safe_load(f)
            train_params.update(optimized_params)

        for key, value in train_params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_yaml", data_yaml)
        mlflow.log_param("device", device)

        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03)
        )
        if mode == "fine-tune" and train_params.get("freeze_backbone", False):
            for param in model.backbone.parameters():
                param.requires_grad = False
        model.to(device)

        train_dataset = SSDDataset(data_yaml, split='train', transform=train_transform)
        val_dataset = SSDDataset(data_yaml, split='val', transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, collate_fn=collate_fn,num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"], shuffle=False, collate_fn=collate_fn,num_workers=16)

        # Choose optimizer dynamically based on config
        if train_params["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=train_params["lr"], weight_decay=train_params.get("weight_decay", 0.0005))
        else:  # Default to SGD
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_params["lr"],
                momentum=train_params.get("momentum", 0.9),
                weight_decay=train_params.get("weight_decay", 0.0005)
            )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_map = 0
        num_batches = len(train_loader)
        print(f"Starting training for {train_params['epochs']} epochs with {num_batches} batches per epoch...")
        for epoch in range(train_params["epochs"]):
            model.train()
            total_loss = 0
            epoch_start_time = time.time()
            for i, (images, targets) in enumerate(train_loader):
                batch_start_time = time.time()
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Verbose output per batch
                batch_time = time.time() - batch_start_time
                print(f"Epoch [{epoch+1}/{train_params['epochs']}], "
                      f"Batch [{i+1}/{num_batches}], "
                      f"Loss: {losses.item():.4f} (Cls: {loss_dict['classification'].item():.4f}, "
                      f"BBox: {loss_dict['bbox_regression'].item():.4f}), "
                      f"Time: {batch_time:.2f}s")

            scheduler.step()
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch [{epoch+1}/{train_params['epochs']}] completed, "
                  f"Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

            model.eval()
            metric = MeanAveragePrecision(max_detection_thresholds=[1, 100, 500])
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(img.to(device) for img in images)
                    outputs = model(images)
                    preds = [{'boxes': o['boxes'].cpu(), 'scores': o['scores'].cpu(), 'labels': o['labels'].cpu()} for o in outputs]
                    targets = [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets]
                    metric.update(preds, targets)
                map_dict = metric.compute()
                mAP = map_dict['map'].item()
                mAP50 = map_dict['map_50'].item()
            mlflow.log_metric("mAP", mAP, step=epoch)
            mlflow.log_metric("mAP50", mAP50, step=epoch)
            print(f"Validation - mAP: {mAP:.4f}, mAP@50: {mAP50:.4f}")

            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), "models/ssd/best_ssd_model.pth")
                mlflow.log_artifact("models/ssd/best_ssd_model.pth")
                print(f"New best mAP: {best_map:.4f}, model saved.")

        tracking_uri_type = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_uri_type != "file":
            mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSD model")
    parser.add_argument("--data_yaml", default=params['data'], help="Path to data.yaml")
    parser.add_argument("--model_name", default=params['model'], help="Model name")
    parser.add_argument("--num_classes", type=int, default=params['num_classes'], help="Number of classes including background")
    parser.add_argument("--mode", choices=["initial", "fine-tune", "optimized"], required=True)
    parser.add_argument("--config_path", default=None, help="Path to optimized params")
    args = parser.parse_args()
    train_model(args.data_yaml, args.model_name, args.num_classes, args.mode, args.config_path)