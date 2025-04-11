import os
import yaml
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SSDDataset(Dataset):
    def __init__(self, data_yaml, split='train', transform=None):
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        self.image_dir = data_config[split]
        self.label_dir = self.image_dir.replace('images', 'labels')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform
        self.classes = data_config['names']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes, labels = self.load_labels(label_path, width, height)

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

    def load_labels(self, label_path, img_width, img_height):
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0]) + 1  # SSD labels start from 1 (0 is background)
                    x_center, y_center, w, h = map(float, parts[1:])
                    # Converting YOLO normalized to absolute [xmin, ymin, xmax, ymax] for ssd format
                    xmin = (x_center - w / 2) * img_width
                    ymin = (y_center - h / 2) * img_height
                    xmax = (x_center + w / 2) * img_width
                    ymax = (y_center + h / 2) * img_height
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
        # If there are no boxes, return the empty tensors
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Resize(300, 300),  
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

val_transform = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))