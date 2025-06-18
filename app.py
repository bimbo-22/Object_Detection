import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import tempfile
import yt_dlp
import torchvision.models.detection as detection
import torch
import time
import yaml
from ultralytics import YOLO
from functools import partial
import torchvision.ops as ops
from collections import defaultdict

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
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        num_classes = params['SSD']['train']['num_classes']
        # Use the width multiplier provided in the YAML (set to 1.0 for checkpoint compatibility)
        width_mult = params['SSD']['train'].get('width_mult', 1.0)
        # Instantiate the SSD model with the desired width multiplier and number of classes.
        model = detection.ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            width_mult=width_mult,
            num_classes=num_classes
        )
        # Replace the classification head to match the number of classes.
        in_channels = [module[0][0].in_channels for module in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.BatchNorm2d, eps=1e-3, momentum=0.03)
        )
        # Determine the weights path based on the version.
        weights_path = params['SSD']['base_model'] if model_version == "Base" else params['SSD']['optimized_model']
        device = torch.device('cpu')
        # Load the fine-tuned weights.
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        model.to(device)
        # (Optional) Print checkpoint keys for debugging.
        checkpoint = torch.load(weights_path, map_location='cpu')
        for key, value in checkpoint.items():
            print(key, value.shape)
        return model

# Resize image while maintaining aspect ratio
def resize_image(image, max_size=640):
    h, w = image.shape[:2]
    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: Tensor of shape [N, 4]
    scores: Tensor of shape [N]
    iou_threshold: Intersection-over-Union threshold to suppress boxes
    Returns the indices of boxes to keep.
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    return ops.nms(boxes, scores, iou_threshold)

# Standardize detection results for YOLO and SSD.
def get_detections(model, image, model_type, confidence, selected_classes):
    if model_type == "YOLO":
        results = model.predict(image, conf=confidence)
        raw_detections = []
        for box_obj in results[0].boxes:
            try:
                class_id = int(box_obj.cls)
                if class_id not in selected_classes:
                    continue
                conf = float(box_obj.conf)
                # Ensure box_obj.xyxy returns a flat list of 4 numbers.
                coords = box_obj.xyxy.detach().cpu().numpy().flatten().tolist()
                if len(coords) != 4:
                    # Skip the detection if it doesn't have exactly 4 coordinates.
                    continue
                raw_detections.append({
                    'class_id': class_id,
                    'confidence': conf,
                    'box': coords
                })
            except Exception as e:
                # Optionally log or ignore errors in processing a detection.
                continue

        # Apply NMS per class
        final_detections = []
        groups = defaultdict(list)
        for det in raw_detections:
            groups[det['class_id']].append(det)
        for cls, det_list in groups.items():
            boxes = torch.tensor([d['box'] for d in det_list], dtype=torch.float32)
            scores = torch.tensor([d['confidence'] for d in det_list], dtype=torch.float32)
            keep = apply_nms(boxes, scores, iou_threshold=0.5)
            for idx in keep:
                final_detections.append(det_list[idx])
        return final_detections
    elif model_type == "SSD":
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)[0]
        detections = []
        boxes_list, scores_list, classes_list = [], [], []
        for i in range(len(outputs['scores'])):
            class_id = int(outputs['labels'][i]) - 1  # Adjust label index.
            score = outputs['scores'][i]
            if score > confidence and class_id in selected_classes:
                box = outputs['boxes'][i].tolist()
                boxes_list.append(box)
                scores_list.append(score)
                classes_list.append(class_id)
        if boxes_list:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            scores_tensor = torch.tensor(scores_list, dtype=torch.float32)
            keep_indices = apply_nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
            for idx in keep_indices:
                detections.append({
                    'class_id': classes_list[idx],
                    'confidence': float(scores_list[idx]),
                    'box': [int(x) for x in boxes_list[idx]]
                })
        return detections

# Draw bounding boxes on the image.
def draw_boxes(image, detections, class_names, confidence):
    """Draw bounding boxes and labels on the image."""
    color_map = {
        'bus': (255, 0, 0),          # Blue
        'car': (0, 255, 0),          # Green
        'motorcycle': (0, 0, 255),   # Red
        'person': (0, 255, 255),     # Yellow
        'truck': (255, 0, 255)       # Magenta
    }
    for det in detections:
        if det['confidence'] >= confidence:
            box = det['box']
            # Ensure the detection box is a list of 4 numbers.
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            # Get the class name and look up the color. If not found, default to white.
            class_id = det['class_id']
            class_label = class_names[class_id] if class_id < len(class_names) else "Unknown"
            color = color_map.get(class_label, (255, 255, 255))
            # Draw bounding box and label with the specific color.
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(image, f"{class_label}: {det['confidence']:.2f}", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def count_detections(detections, class_names):
    counts = {}
    for det in detections:
        label = class_names[det['class_id']] if det['class_id'] < len(class_names) else "Unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts

def process_video(video_path, model, model_type, confidence, class_names, selected_classes):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    max_size = 640
    scale = min(max_size / width, max_size / height)
    new_width, new_height = int(width * scale), int(height * scale)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (new_width, new_height))
    all_detections = []
    progress_bar = st.progress(0)
    frame_times = []
    frame_skip = 5
    processed_frames = 0
    
    with st.spinner("Processing video..."):
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_skip != 0:
                continue
            start_time = time.time()
            frame = cv2.resize(frame, (new_width, new_height))
            detections = get_detections(model, frame, model_type, confidence, selected_classes)
            all_detections.extend(detections)
            annotated_frame = draw_boxes(frame.copy(), detections, class_names, confidence)
            out.write(annotated_frame)
            frame_times.append(time.time() - start_time)
            processed_frames += 1
            progress_bar.progress(min((i + 1) / frame_count, 1.0))
            for _ in range(frame_skip - 1):
                if i + 1 < frame_count:
                    out.write(annotated_frame)
                    i += 1
    cap.release()
    out.release()
    avg_fps = processed_frames / sum(frame_times) if frame_times else 0
    return temp_file.name, all_detections, avg_fps

def main():
    st.title("SecureDetect: Object Detection Model APP")
    st.write("Test YOLO and SSD models on images, videos, or YouTube links.")
    
    st.sidebar.title("Settings")
    model_type = st.sidebar.selectbox("Select Model", ["YOLO", "SSD"])
    model_version = st.sidebar.selectbox("Select Model Version", ["Base", "Optimized"])
    input_type = st.sidebar.selectbox("Select Input Type", ["Image", "Video", "YouTube"])
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.39, 0.01)
    selected_class_names = st.sidebar.multiselect("Select Classes to Detect", params['class_names'], default=params['class_names'])
    run_button = st.sidebar.button("Run Inference")
    
    class_names = params['class_names']
    selected_classes = [class_names.index(cls) for cls in selected_class_names]
    
    model = load_model(model_type, model_version)
    
    st.sidebar.write("**About Models:**")
    st.sidebar.write("- [YOLOv8](https://github.com/ultralytics/ultralytics): Fast and accurate object detection.")
    st.sidebar.write("- [SSD](https://pytorch.org/vision/stable/models.html): Single Shot Detector, lightweight.")
    st.sidebar.write("### Read the paper ###")
    st.sidebar.write("- [Object Detection for Security Camera System](https://github.com/bimbo-22/Object_Detection/blob/main/thesis_paper/cs_2025_abimbola_mohammed_ogunsakin.pdf): This project is based on the thesis paper by Abimbola Mohammed Ogunsakin.")
    
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
                with open(processed_video_path, "rb") as file:
                    st.download_button("Download Annotated Video", file, "annotated_video.mp4")
                os.unlink(video_path)
                os.unlink(processed_video_path)
    
    elif input_type == "YouTube":
        youtube_url = st.text_input("Enter YouTube URL")
        if youtube_url and run_button:
            with st.spinner("Downloading YouTube video..."):
                ydl_opts = {'outtmpl': 'downloaded_video.%(ext)s', 'format': 'best'}
                import yt_dlp
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
                    with open(processed_video_path, "rb") as file:
                        st.download_button("Download Annotated Video", file, "annotated_video.mp4")
                    os.unlink(video_path)
                    os.unlink(processed_video_path)

if __name__ == "__main__":
    main()