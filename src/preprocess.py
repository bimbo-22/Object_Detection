import pandas as pd 
import sys
import os
import yaml
import cv2
import glob
import random
import numpy as np
import stat


params = yaml.safe_load(open('params.yaml'))['preprocess']

# Baseline Augmentation (Initial Training):
# Rotation: Between -15° and +15°
# Saturation: Between -10% and +10%
# Brightness: Between -10% and +10%
# Exposure: Between -10% and +10%
# Blur: Up to 2px
# Noise: Up to 3% of pixels

# load label needed
def load_labels(label_path):
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as file :
        labels = file.readlines()
    return [list(map(float, line.strip().split())) for line in labels]

# save the aug yolo labels
def save_labels(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(' '.join(map(str, label)) +  '\n')


# rotating bounding box in respect to augumentation
def rotate_bbox(cx,cy,w,h,angle,img_w,img_h):
    angle = np.deg2rad(angle)
    new_cx = cx * img_w
    new_cy = cy * img_h
    
    new_x = (new_cx - img_w / 2) * np.cos(angle) - (new_cy - img_h / 2) * np.sin(angle) + img_w / 2
    new_y = (new_cx - img_w / 2) * np.cos(angle) + (new_cy - img_h / 2) * np.sin(angle) + img_w / 2
    
    return new_x / img_w, new_y / img_h, w, h # keeping the width and height no need to change
    

def preprocess(input_path, output_image_path, input_label, output_label):
    
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    if not os.path.exists(output_label):
        os.makedirs(output_label)
        
    print("Starting to preprocess images and labels.")
    for filename in os.listdir(input_path):
        input_image_path = os.path.join(input_path, filename)
        input_label_path = os.path.join(input_label, filename.replace('.jpg', '.txt')).replace('.png', '.txt')
        if not any(ext in filename.lower() for ext in ('.png', '.jpg', '.jpeg')):
            continue
        
        image = cv2.imread(input_image_path)
        print(type(image))
        if image is None:
            print(f"Could not read image {input_image_path}")
            continue
        else:
            print(f"Image loaded")
        
        label = load_labels(input_label_path)
        if label is None:
            print(f"No Labels for {filename} found")
            continue
            
        # Data Augumentation
        height, width = image.shape[:2]
        augumented_images = []
        augumented_labels = []
        
        # Rotation
        rotation_angle = random.randint(-15, 15)
        center = (width//2, height[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        rotated_label = [rotate_bbox(cx, cy, w, h, rotation_angle, width. height) for class_id, cx,cy,w,h in label]
        augumented_images.append(rotated_image)
        augumented_labels.append(rotated_label)
        
        # Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + random.uniform(-10, 10)), 0, 255)
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augumented_images.append(saturated)
        
        # Brightness
        brightness_factor = random.uniform(-0.10, 0.10)
        brightened = np.clip(image * (1 + brightness_factor), 0, 255).astype(np.uint8)
        augumented_images.append(brightened)
        # no change to bounding box
        
        # Blur
        blur_factor = random.choice([3,5])
        blurred = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
        augumented_images.append(blurred)
        
        # Noise
        noise_factor = np.random.normal(0,0.03,image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise_factor)
        augumented_images.append(noisy)
        print(f"Augumented images created successfully.")
        
        for i, (aug_image, aug_label) in enumerate(zip(augumented_images, augumented_labels)):
            output_image_path = os.path.join(output_image_path, f"{filename.split('.')[0]}_{i}.png")
            output_label_path = os.path.join(output_label_path, f"{filename.split('.')[0]}_{i}.txt")
            
            
            cv2.imwrite(output_image_path, aug_image)
            save_labels(output_label_path, [[class_id] + list(map(float,bbox)) for class_id, *bbox in aug_label])
            print(f"Image saved {output_image_path}")
            print(f"Labels saved {output_label_path}")
        
    print("Preprocessing completed. ****")
            

if __name__ == "__main__":
    print("Script is executing.")
    preprocess(params['input_images'],params['input_labels'], params['output_images'] params['output_labels'])
        