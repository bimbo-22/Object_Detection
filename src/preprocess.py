import pandas as pd 
import sys
import os
import yaml
import cv2
import glob

# create a function then apply data augmentation on the images and
# save them in same directory but call it augumented_images use params.yaml
params = yaml.safe_load(open('params.yaml'))['preprocess']



def preprocess(input_path, output_path):
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
            

if __name__ == "__main__":
    print("Script is executing.")
    preprocess(params['input'], params['output'])
        