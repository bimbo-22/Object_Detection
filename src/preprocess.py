import pandas as pd 
import sys
import os
import yaml
import cv2
import glob
import random
import numpy as np


params = yaml.safe_load(open('params.yaml'))['preprocess']

# Baseline Augmentation (Initial Training):
# Rotation: Between -15° and +15°
# Saturation: Between -10% and +10%
# Brightness: Between -10% and +10%
# Exposure: Between -10% and +10%
# Blur: Up to 2px
# Noise: Up to 3% of pixels

def preprocess(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Starting to preprocess images.")
    for filename in os.listdir(input_path):
        input_image_path = os.path.join(input_path, filename)
        
        if not any(ext in filename.lower() for ext in ('.png', '.jpg', '.jpeg')):
            continue
        
        image = cv2.imread(input_image_path)
        print(type(image))
        if image is None:
            print(f"Could not read image {input_image_path}")
        else:
            print(f"Image loaded {input_image_path}")
            
        # Data Augumentation
        augumented_images = []
        
        # Rotation
        rotation_angle = random.randint(-15, 15)
        height, width = image.shape[:2]
        center = (image.shape[1]//2, image.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        augumented_images.append(rotated_image)
        
        # Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + random.uniform(-10, 10)), 0, 255)
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augumented_images.append(saturated)
        
        # Brightness
        brightness_factor = random.uniform(-0.10, 0.10)
        brightened = np.clip(image * (1 + brightness_factor), 0, 255).astype(np.uint8)
        augumented_images.append(brightened)
        
        # Blur
        blur_factor = random.choice([3,5])
        blurred = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
        augumented_images.append(blurred)
        
        # Noise
        noise_factor = np.random.normal(0,0.03,image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise_factor)
        augumented_images.append(noisy)
        print(f"Augumented images created successfully.")
        for i, augumented_image in enumerate(augumented_images):
            output_image_path = os.path.join(output_path, f"{filename.split('.')[0]}_{i}.png")
            cv2.imwrite(output_image_path, augumented_image)
            print(f"Image saved {output_image_path}")
    print("Preprocessing completed. ****")
            

if __name__ == "__main__":
    print("Script is executing.")
    preprocess(params['input'], params['output'])
        