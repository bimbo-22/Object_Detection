import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import tempfile
import yt_dlp
from ultralytics import YOLO
import torchvision.models.detection as detection
import torch
import time
import yaml

# Load parameters from params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Cache model loading for performance
@st.cache_resource
def load_model(model_type, model_version):
    """Load the selected model (YOLO or SSD, base or optimized) from params.yaml."""
    if model_type == "YOLO":
        model_path = params['YOLO']['base_model'] if model_version == "Base" else params['YOLO']['optimized_model']
        return YOLO(model_path)
    elif model_type == "SSD":
        # Create the model with the correct configuration
        num_classes = params['SSD']['train']['num_classes']  
        model = detection.ssdlite320_mobilenet_v3_large(
            num_classes=num_classes
        )
        # Determine the weights path based on the version
        weights_path = params['SSD']['base_model'] if model_version == "Base" else params['SSD']['optimized_model']
        # Load the weights
        device = torch.device('cpu')
        model.load_state_dict(torch.load(weights_path, map_location=device), strict = False)
        # Set to evaluation mode
        model.eval()
        # Ensure model is on CPU
        model.to(device)
        checkpoint = torch.load(params['SSD']['optimized_model'], map_location='cpu')
        checkpoint = torch.load(weights_path, map_location='cpu')
        for key, value in checkpoint.items():
            print(key, value.shape)

        return model

# Resize image while maintaining aspect ratio
def resize_image(image, max_size=640):
    """Downscale image for faster CPU processing."""
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))

# Standardize detection results
def get_detections(model, image, model_type, confidence, selected_classes):
    """Run inference and return detections, filtered by selected classes."""
    if model_type == "YOLO":
        results = model.predict(image, conf=confidence)
        detections = []
        for result in results[0].boxes:
            class_id = int(result.cls)
            if class_id in selected_classes:
                detections.append({
                    'class_id': class_id,
                    'confidence': float(result.conf),
                    'box': [int(x) for x in result.xyxy[0].tolist()]
                })
        return detections
    elif model_type == "SSD":
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)[0]
        detections = []
        for i in range(len(outputs['scores'])):
            class_id = int(outputs['labels'][i]) - 1  # SSD labels start from 1 (0 is background)
            if outputs['scores'][i] > confidence and class_id in selected_classes:
                box = outputs['boxes'][i].tolist()
                detections.append({
                    'class_id': class_id,
                    'confidence': float(outputs['scores'][i]),
                    'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                })
        return detections

# Draw bounding boxes on the image
def draw_boxes(image, detections, class_names, confidence):
    """Draw bounding boxes and labels on the image."""
    for det in detections:
        if det['confidence'] >= confidence:
            box = det['box']
            label = class_names[det['class_id']] if det['class_id'] < len(class_names) else "Unknown"
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {det['confidence']:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# Count detected objects
def count_detections(detections, class_names):
    """Count occurrences of each detected class."""
    counts = {}
    for det in detections:
        label = class_names[det['class_id']] if det['class_id'] < len(class_names) else "Unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts

# Process video files with progress bar and frame skipping
def process_video(video_path, model, model_type, confidence, class_names, selected_classes):
    """Process a video with frame skipping and return the annotated video path, detections, and FPS."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Downscale video frames
    max_size = 640  # Reduce resolution for faster processing
    scale = min(max_size / width, max_size / height)
    new_width, new_height = int(width * scale), int(height * scale)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
    all_detections = []
    progress_bar = st.progress(0)
    frame_times = []
    frame_skip = 5  # Process every 5th frame to reduce CPU load
    processed_frames = 0
    
    with st.spinner("Processing video..."):
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_skip != 0:  # Skip frames
                continue
            start_time = time.time()
            # Downscale frame
            frame = cv2.resize(frame, (new_width, new_height))
            detections = get_detections(model, frame, model_type, confidence, selected_classes)
            all_detections.extend(detections)
            annotated_frame = draw_boxes(frame.copy(), detections, class_names, confidence)
            out.write(annotated_frame)
            frame_times.append(time.time() - start_time)
            processed_frames += 1
            progress_bar.progress(min((i + 1) / frame_count, 1.0))
            # Write the same annotated frame for skipped frames to maintain video length
            for _ in range(frame_skip - 1):
                if i + 1 < frame_count:
                    out.write(annotated_frame)
                    i += 1
    cap.release()
    out.release()
    avg_fps = processed_frames / sum(frame_times) if frame_times else 0
    return temp_file.name, all_detections, avg_fps

# Main app function
def main():
    st.title("Object Detection Model APP")
    st.write("Test YOLO and SSD models on images, videos, or YouTube links.")
    
    
    # Sidebar controls
    st.sidebar.title("Settings")
    model_type = st.sidebar.selectbox("Select Model", ["YOLO", "SSD"])
    model_version = st.sidebar.selectbox("Select Model Version", ["Base", "Optimized"])
    input_type = st.sidebar.selectbox("Select Input Type", ["Image", "Video", "YouTube"])
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    selected_class_names = st.sidebar.multiselect("Select Classes to Detect", params['class_names'], default=params['class_names'])
    run_button = st.sidebar.button("Run Inference")
    

    # Map selected class names to indices
    class_names = params['class_names']
    selected_classes = [class_names.index(cls) for cls in selected_class_names]

    # Load the selected model
    model = load_model(model_type, model_version)

    # Model information in sidebar
    st.sidebar.write("**About Models:**")
    st.sidebar.write("- [YOLOv8](https://github.com/ultralytics/ultralytics): Fast and accurate object detection.")
    st.sidebar.write("- [SSD](https://pytorch.org/vision/stable/models.html): Single Shot Detector, lightweight.")
    
    st.sidebar.write("### Read the paper ###")
    ## link to my thesis
    st.sidebar.write("- [Object Detection for Security Camera System]()")
    # Handle different input types
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload Image (max 200MB)", type=['jpg', 'jpeg', 'png'])
        if uploaded_file and run_button:
            if uploaded_file.size > 200 * 1024 * 1024:
                st.error("Image exceeds 200MB limit.")
            else:
                start_time = time.time()
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                # Downscale image
                image_cv = resize_image(image_cv, max_size=640)
                with st.spinner("Running inference..."):
                    detections = get_detections(model, image_cv, model_type, confidence, selected_classes)
                    annotated_image = draw_boxes(image_cv.copy(), detections, class_names, confidence)
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, caption="Detection Results", use_container_width=True)
                counts = count_detections(detections, class_names)
                st.write("**Detected Classes:**", counts)
                avg_conf = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
                st.write(f"**Average Confidence:** {avg_conf:.2f}")
                st.write(f"**Inference Time:** {time.time() - start_time:.2f} seconds")
                st.balloons()
                # Download annotated image
                annotated_image_path = "annotated_image.png"
                cv2.imwrite(annotated_image_path, annotated_image)
                with open(annotated_image_path, "rb") as file:
                    st.download_button("Download Annotated Image", file, "annotated_image.png")

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload Video (max 200MB)", type=['mp4', 'avi', 'mov'])
        if uploaded_file and run_button:
            if uploaded_file.size > 200 * 1024 * 1024:
                st.error("Video exceeds 200MB limit.")
            else:
                start_time = time.time()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                processed_video_path, detections, avg_fps = process_video(video_path, model, model_type, confidence, class_names, selected_classes)
                st.video(processed_video_path)
                counts = count_detections(detections, class_names)
                st.write("**Detected Classes:**", counts)
                avg_conf = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
                st.write(f"**Average Confidence:** {avg_conf:.2f}")
                st.write(f"**Processing Time:** {time.time() - start_time:.2f} seconds")
                st.write(f"**Frames Per Second:** {avg_fps:.2f}")
                st.balloons()
                # Download annotated video
                with open(processed_video_path, "rb") as file:
                    st.download_button("Download Annotated Video", file, "annotated_video.mp4")
                os.unlink(video_path)
                os.unlink(processed_video_path)

    elif input_type == "YouTube":
        youtube_url = st.text_input("Enter YouTube URL")
        if youtube_url and run_button:
            with st.spinner("Downloading YouTube video..."):
                ydl_opts = {'outtmpl': 'downloaded_video.%(ext)s', 'format': 'best'}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                video_path = "downloaded_video.mp4"
                if os.path.getsize(video_path) > 200 * 1024 * 1024:
                    st.error("Video exceeds 200MB limit.")
                    os.unlink(video_path)
                else:
                    start_time = time.time()
                    processed_video_path, detections, avg_fps = process_video(video_path, model, model_type, confidence, class_names, selected_classes)
                    st.video(processed_video_path)
                    counts = count_detections(detections, class_names)
                    st.write("**Detected Classes:**", counts)
                    avg_conf = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
                    st.write(f"**Average Confidence:** {avg_conf:.2f}")
                    st.write(f"**Processing Time:** {time.time() - start_time:.2f} seconds")
                    st.write(f"**Frames Per Second:** {avg_fps:.2f}")
                    st.balloons()
                    # Download annotated video
                    with open(processed_video_path, "rb") as file:
                        st.download_button("Download Annotated Video", file, "annotated_video.mp4")
                    os.unlink(video_path)
                    os.unlink(processed_video_path)

if __name__ == "__main__":
    main()