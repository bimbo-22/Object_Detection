import os
import yaml
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class SSDDataset(Dataset):
    def __init__(self, data_yaml, split='train', transform=None):
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        self.image_dir = data_config[split]
        self.label_dir = self.image_dir.replace('images', 'labels')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform
        self.classes = data_config['names']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')

        image = cv2.imread(image_path) 
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        boxes, labels = self.load_labels(label_path, width, height)

        if self.transform:

            boxes_list = boxes.tolist() if isinstance(boxes, torch.Tensor) else boxes
            labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else labels
            transformed = self.transform(image=image, bboxes=boxes_list, class_labels=labels_list)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)
            # Ensure boxes is a tensor with shape [N,4]
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
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
                    class_id = int(parts[0]) + 1  # Increment by 1, as SSD reserves 0 for background.
                    x_center, y_center, w, h = map(float, parts[1:])
                    xmin = (x_center - w / 2) * img_width
                    ymin = (y_center - h / 2) * img_height
                    xmax = (x_center + w / 2) * img_width
                    ymax = (y_center + h / 2) * img_height
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels

# Define Albumentations transforms for training and validation
train_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))

val_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))
