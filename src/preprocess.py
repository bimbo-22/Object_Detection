import os
import yaml
import cv2
import albumentations as A
import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util
import argparse

# Load parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['preprocess']


np.random.seed(42)


transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.0), contrast_limit=0.1, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=0.1, p=0.3),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.15),
    A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.15),
    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.15),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True))

def load_labels(label_path):
    """Load YOLO format labels from a text file."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as file:
        labels = [list(map(float, line.strip().split())) for line in file]
    return labels

def save_labels(label_path, labels, class_labels):
    """Save labels in YOLO format to a text file."""
    with open(label_path, 'w') as file:
        for label, class_id in zip(labels, class_labels):
            file.write(f"{int(class_id)} {' '.join(map(str, label))}\n")

def create_tf_example(image, bboxes, class_labels, filename):
    """Create a TFExample for SSD from image and YOLO format labels."""
    height, width = image.shape[:2]
    encoded_image = cv2.imencode('.jpg', image)[1].tobytes()
    filename = filename.encode('utf8')
    image_format = b'jpeg'

    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []
    for bbox, class_id in zip(bboxes, class_labels):
        cx, cy, w, h = bbox
        class_id = int(class_id)
        classes.append(class_id)
        # Convert YOLO center format to SSD corner format
        xmin = cx - w / 2
        xmax = cx + w / 2
        ymin = cy - h / 2
        ymax = cy + h / 2
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(str(class_id).encode('utf8'))  # Placeholder for class text

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def preprocess_yolo(input_path, output_image, input_label, output_label):
    """Preprocess data for YOLO: apply augmentations and save in YOLO format."""
    os.makedirs(output_image, exist_ok=True)
    os.makedirs(output_label, exist_ok=True)
    
    print("Starting to preprocess YOLO images and labels.")
    for filename in os.listdir(input_path):
        if not any(ext in filename.lower() for ext in ('.png', '.jpg', '.jpeg')):
            print(f"Skipping {filename}: Unsupported format")
            continue
        
        input_image_path = os.path.join(input_path, filename)
        input_label_path = os.path.join(input_label, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Failed to load image: {input_image_path}")
            continue
        
        labels = load_labels(input_label_path)
        if labels is None:
            print(f"No labels found for {filename}")
            continue
        
        bboxes = [label[1:] for label in labels]  # [cx, cy, w, h]
        class_labels = [label[0] for label in labels]  # class_id
        
        # Save original
        original_output_image_path = os.path.join(output_image, filename)
        cv2.imwrite(original_output_image_path, image)
        original_output_label_path = os.path.join(output_label, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        save_labels(original_output_label_path, bboxes, class_labels)
        print(f"Saved original: {original_output_image_path} and {original_output_label_path}")
        
        
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_class_labels = augmented['class_labels']
        
        base_filename = filename.split('.')[0]
        output_image_path = os.path.join(output_image, f"{base_filename}_aug.png")
        output_label_path = os.path.join(output_label, f"{base_filename}_aug.txt")
        
        cv2.imwrite(output_image_path, aug_image)
        save_labels(output_label_path, aug_bboxes, aug_class_labels)
        print(f"Saved augmented: {output_image_path} and {output_label_path}")
    
    print("YOLO preprocessing completed.")

def preprocess_ssd(input_path, input_label, output_tfrecord, apply_augmentation=False):
    """Preprocess data for SSD: convert to TFRecords with optional augmentations."""
    os.makedirs(os.path.dirname(output_tfrecord), exist_ok=True)
    writer = tf.io.TFRecordWriter(output_tfrecord)
    
    print(f"Starting to preprocess SSD data, saving to {output_tfrecord}")
    for filename in os.listdir(input_path):
        if not any(ext in filename.lower() for ext in ('.png', '.jpg', '.jpeg')):
            print(f"Skipping {filename}: Unsupported format")
            continue
        
        input_image_path = os.path.join(input_path, filename)
        input_label_path = os.path.join(input_label, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Failed to load image: {input_image_path}")
            continue
        
        labels = load_labels(input_label_path)
        if labels is None:
            print(f"No labels found for {filename}")
            continue
        
        bboxes = [label[1:] for label in labels]  # [cx, cy, w, h]
        class_labels = [label[0] for label in labels]  # class_id
        
        # Create TFExample 
        tf_example = create_tf_example(image, bboxes, class_labels, filename)
        writer.write(tf_example.SerializeToString())
        print(f"Added original {filename} to TFRecord")
        
        
        if apply_augmentation:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']
            if aug_bboxes:  
                aug_filename = f"{filename.split('.')[0]}_aug.jpg"
                tf_example_aug = create_tf_example(aug_image, aug_bboxes, aug_class_labels, aug_filename)
                writer.write(tf_example_aug.SerializeToString())
                print(f"Added augmented {aug_filename} to TFRecord")
    
    writer.close()
    print("SSD preprocessing completed.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preprocess data for YOLO or SSD')
    parser.add_argument('--model', choices=['yolo', 'ssd'], required=True, help='Model type (yolo or ssd)')
    parser.add_argument('--dataset', choices=['train', 'valid'], required=True, help='Dataset type (train or valid)')
    args = parser.parse_args()
    
    print(f"Script executing for model: {args.model}, dataset: {args.dataset}")
    
    if args.model == 'yolo':
        
        input_images = params['input_images']
        input_labels = params['input_labels']
        output_images = params['output_images']
        output_labels = params['output_labels']
        preprocess_yolo(input_images, output_images, input_labels, output_labels)
    elif args.model == 'ssd':
        
        input_images = params['ssd'][args.dataset]['input_images']
        input_labels = params['ssd'][args.dataset]['input_labels']
        output_tfrecord = params['ssd'][args.dataset]['output_tfrecord']
        apply_augmentation = (args.dataset == 'train')
        preprocess_ssd(input_images, input_labels, output_tfrecord, apply_augmentation)