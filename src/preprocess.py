import os
import yaml
import cv2
import albumentations as A
import numpy as np

# Load parameters from params.yaml
params = yaml.safe_load(open('params.yaml'))['preprocess']

np.random.seed(42)

# Define augmentation pipeline
transform = A.Compose([
    # Rotation for variety
    A.Rotate(limit=15, p=0.5),  # Rotate by up to 15 degrees, 50% chance
    
    # Horizontal flip
    A.HorizontalFlip(p=0.5),  # 50% chance of flipping horizontally
    
    # Low/Dim Light: Reduce brightness (less aggressive)
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.0), contrast_limit=0.1, p=0.3),  # 30% chance
    
    # Too Much Light: Increase brightness (less aggressive)
    A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=0.1, p=0.3),  # 30% chance
    
    # Adverse Weather Conditions (reduced probability)
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.15),  # 15% chance
    A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.15),  # 15% chance
    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.15),  # 15% chance
    
    # Optional: Noise for realism
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),  # Light noise, 20% chance
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True))  # Clip to fix errors

def load_labels(label_path):
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as file:
        labels = [list(map(float, line.strip().split())) for line in file]
    return labels

def save_labels(label_path, labels, class_labels):
    with open(label_path, 'w') as file:
        for label, class_id in zip(labels, class_labels):
            file.write(f"{int(class_id)} {' '.join(map(str, label))}\n")

def preprocess(input_path, output_image, input_label, output_label):
    os.makedirs(output_image, exist_ok=True)
    os.makedirs(output_label, exist_ok=True)
    
    print("Starting to preprocess images and labels.")
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
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    print("Script executing...")
    preprocess(params['input_images'], params['output_images'], params['input_labels'], params['output_labels'])