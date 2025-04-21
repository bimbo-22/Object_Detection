import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import mlflow
from ssd_dataset import SSDDataset, val_transform
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from functools import partial
from dotenv import load_dotenv
import os
import torchmetrics
print(f"Using torchmetrics from: {torchmetrics.__file__}")
print(f"torchmetrics version: {torchmetrics.__version__}")
import sys
print(f"Python executable: {sys.executable}")

load_dotenv()
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(mlflow_tracking_uri)

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_ssd(model_path, data_yaml, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    with mlflow.start_run(run_name="SSD_Evaluation"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("data_yaml", data_yaml)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("device", device)
        mlflow.log_param("model_name", "ssdlite320_mobilenet_v3_large")

        model = ssdlite320_mobilenet_v3_large(weights=None)
        in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03)
        )
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        val_dataset = SSDDataset(data_yaml, split='val', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

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

        mlflow.log_metric("mAP", mAP)
        mlflow.log_metric("mAP50", mAP50)
        mlflow.pytorch.log_model(model, "evaluated_model")

        print(f"mAP: {mAP}, mAP50: {mAP50}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SSD model")
    parser.add_argument("--model_path", required=True, help="Path to trained model weights")
    parser.add_argument("--data_yaml", required=True, help="Path to data.yaml")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes including background")
    args = parser.parse_args()
    evaluate_ssd(args.model_path, args.data_yaml, args.num_classes)