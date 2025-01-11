import pandas as pd 
import sys
import os
import yaml
import cv2
import glob
import random
import numpy as np

# create a function then apply data augmentation on the images and
# save them in same directory but call it augumented_images use params.yaml
params = yaml.safe_load(open('params.yaml'))['preprocess']

# preadded data augmentation on training data
# Rotation: Between -25° and +25°
# Saturation: Between -15% and +15%
# Brightness: Between -15% and +15%
# Exposure: Between -15% and +15%
# Blur: Up to 3px
# Noise: Up to 5% of pixels


def preprocess(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for filename in os.listdir(input_path):
        input_image_path = os.path.join(input_path, filename)
        
        if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
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
        rotation_angle = random.randint(-25, 25)
        height, width = image.shape[:2]
        center = (image.shape[1]//2, image.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        augumented_images.append(rotated_image)
        
        # Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + random.uniform(-15, 15)), 0, 255)
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augumented_images.append(saturated)
        
        # Brightness
        brightness_factor = random.uniform(-0.15, 0.15)
        brightened = np.clip(image * (1 + brightness_factor), 0, 255).astype(np.uint8)
        augumented_images.append(brightened)
        
        # Blur
        blur_factor = random.choice([3,5])
        blurred = cv2.GaussianBlur(image, (blur_factor, blur_factor), 0)
        augumented_images.append(blurred)
        
        # Noise
        noise_factor = np.random.normal(0,25,image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise_factor)
        augumented_images.append(noisy)
        
        for i, augumented_image in enumerate(augumented_images):
            output_image_path = os.path.join(output_path, f"{filename.split('.')[0]}_{i}.png")
            cv2.imwrite(output_image_path, augumented_image)
            print(f"Image saved {output_image_path}")
            

if __name__ == "__main__":
    print("Script is executing.")
    preprocess(params['input'], params['output'])
        