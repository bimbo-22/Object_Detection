import argparse
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
import mlflow
from urllib.parse import urlparse
import os
import yaml
from ssd_dataset import SSDDataset, val_transform
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

params = yaml.safe_load(open('params.yaml'))['SSD']['train']

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model_path, data_yaml, num_classes):
    with mlflow.start_run(run_name="SSD Evaluation on Custom Dataset"):
        model = ssd300_vgg16(weights=None)
        in_channels = model.head.classification_head.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        model.load_state_dict(torch.load(model_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        val_dataset = SSDDataset(data_yaml, split='val', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

        metric = MeanAveragePrecision()
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                outputs = model(images)
                preds = [{'boxes': o['boxes'], 'scores': o['scores'], 'labels': o['labels']} for o in outputs]
                targets = [{'boxes': t['boxes'], 'labels': t['labels']} for t in targets]
                metric.update(preds, targets)
        metrics = metric.compute()

        mlflow.log_param("model_path", model_path)
        mlflow.log_param("data_yaml", data_yaml)
        mlflow.log_metric("mAP", metrics['map'].item())
        mlflow.log_metric("mAP50", metrics['map_50'].item())

        with open("results_ssd.txt", "w") as f:
            f.write(f"mAP: {metrics['map'].item()}\n")
            f.write(f"mAP50: {metrics['map_50'].item()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SSD model")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--data_yaml", default=params['data'], help="Path to data.yaml")
    parser.add_argument("--num_classes", type=int, default=params['num_classes'], help="Number of classes")
    args = parser.parse_args()
    evaluate(args.model_path, args.data_yaml, args.num_classes)