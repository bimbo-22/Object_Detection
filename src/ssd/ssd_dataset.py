import os
import yaml
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class SSDDataset(Dataset):
    def __init__(self, data_yaml, split='train', transform=None):
        # Load configuration from data_yaml file (assumed to be in YOLO format)
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        # For a given split (e.g. 'train' or 'val'), assume the key stores the images path.
        self.image_dir = data_config[split]
        # Assume that the labels live in a directory with same name as images but replacing 'images' with 'labels'
        self.label_dir = self.image_dir.replace('images', 'labels')
        # List image files that have a jpg or png extension.
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform
        # Also save the class names if provided in the YAML (for example 'names' key)
        self.classes = data_config['names']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Construct image and label file paths.
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        # Replace file extension for label file. (Works for both .jpg and .png)
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')

        # Read image with OpenCV and convert from BGR to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Load bounding boxes and labels from file (YOLO format) and convert to absolute pascal_voc boxes
        boxes, labels = self.load_labels(label_path, width, height)
        
        # If a transform is provided, convert boxes and labels appropriately.
        if self.transform:
            # Convert boxes to list (albumentations expects a list)
            boxes_list = boxes.tolist() if isinstance(boxes, torch.Tensor) else boxes
            # Similarly, if labels is a tensor, convert it into a list of ints
            labels_list = labels.tolist() if isinstance(labels, torch.Tensor) else labels
            transformed = self.transform(image=image, bboxes=boxes_list, class_labels=labels_list)
            image = transformed['image']
            boxes = transformed['bboxes']
            # Convert the transformed labels back to a torch tensor
            labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Return dictionary target with boxes and labels (SSD expects these in a dict)
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if not isinstance(boxes, torch.Tensor) else boxes,
            'labels': labels
        }
        return image, target

    def load_labels(self, label_path, img_width, img_height):
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # Convert YOLO format: add 1 because SSD expects labels to start from 1 (with 0 as background)
                    class_id = int(parts[0]) + 1  
                    x_center, y_center, w, h = map(float, parts[1:])
                    # Convert normalized coordinates to absolute coordinates in [xmin, ymin, xmax, ymax] format
                    xmin = (x_center - w / 2) * img_width
                    ymin = (y_center - h / 2) * img_height
                    xmax = (x_center + w / 2) * img_width
                    ymax = (y_center + h / 2) * img_height
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
        # If no boxes found, return empty tensors/lists
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels

# Define Albumentations transforms for training and validation
train_transform = A.Compose([
    A.Resize(320, 320),  # Resize the image to 320x320 for SSD input size
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))

val_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(p=1.0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))
