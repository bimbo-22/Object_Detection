import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
import mlflow
import optuna
import yaml
from ssd_dataset import SSDDataset, train_transform, val_transform
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

params = yaml.safe_load(open('params.yaml'))['SSD']['train']

def collate_fn(batch):
    return tuple(zip(*batch))

def objective(trial):
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 10, 20, 30, 40, 50)  # Reduced for optimization speed

    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    in_channels = model.head.classification_head.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, params["num_classes"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_dataset = SSDDataset(params["data"], split='train', transform=train_transform)
    val_dataset = SSDDataset(params["data"], split='val', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            preds = [{'boxes': o['boxes'], 'scores': o['scores'], 'labels': o['labels']} for o in outputs]
            targets = [{'boxes': t['boxes'], 'labels': t['labels']} for t in targets]
            metric.update(preds, targets)
    mAP50 = metric.compute()['map_50'].item()

    return mAP50

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    with open("results/best_ssd_params.yaml", "w") as f:
        yaml.dump(best_params, f)
    print("Best parameters:", best_params)