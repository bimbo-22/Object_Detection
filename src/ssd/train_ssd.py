import argparse
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import mlflow
from urllib.parse import urlparse
import os
import yaml
from ssd_dataset import SSDDataset, train_transform, val_transform
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Load parameters
params = yaml.safe_load(open('params.yaml'))['SSD']['train']

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(data_yaml, model_name, num_classes, mode, config_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    run_name = f"SSD Training ({mode}) on Custom Dataset"
    with mlflow.start_run(run_name=run_name):
        # training parameters
        train_params = {
            "epochs": params.get('epochs', 50),
            "batch_size": params.get('batch', 16),
            "lr": params.get('lr', 0.001)
        }
        if mode == "fine-tune":
            train_params["freeze_backbone"] = True
            train_params["epochs"] = 30
        elif mode == "optimized" and config_path:
            with open(config_path, 'r') as f:
                optimized_params = yaml.safe_load(f)
            train_params.update(optimized_params)

        # Log parameters
        for key, value in train_params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_yaml", data_yaml)
        mlflow.log_param("device", device)

        # Load model
        model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        in_channels = model.head.classification_head.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        if mode == "fine-tune":
            for param in model.backbone.parameters():
                param.requires_grad = False
        model.to(device)

        # Datasets and dataloaders
        train_dataset = SSDDataset(data_yaml, split='train', transform=train_transform)
        val_dataset = SSDDataset(data_yaml, split='val', transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=train_params["batch_size"], shuffle=False, collate_fn=collate_fn)

        # Optimizer and scheduler
        optimizer = torch.optim.SGD(model.parameters(), lr=train_params["lr"], momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Training loop
        best_map = 0
        for epoch in range(train_params["epochs"]):
            model.train()
            total_loss = 0
            for images, targets in train_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Validation
            model.eval()
            metric = MeanAveragePrecision()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = list(img.to(device) for img in images)
                    outputs = model(images)
                    preds = [{'boxes': o['boxes'], 'scores': o['scores'], 'labels': o['labels']} for o in outputs]
                    targets = [{'boxes': t['boxes'], 'labels': t['labels']} for t in targets]
                    metric.update(preds, targets)
                map_dict = metric.compute()
                mAP = map_dict['map'].item()
                mAP50 = map_dict['map_50'].item()
            mlflow.log_metric("mAP", mAP, step=epoch)
            mlflow.log_metric("mAP50", mAP50, step=epoch)

            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), "models/best_ssd_model.pth")
                mlflow.log_artifact("models/best_ssd_model.pth")

        # Register model if remote tracking
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