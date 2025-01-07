from ultralytics import YOLO
from mlflow.models import infer_signature
import mlflow

# use python sdk
model = YOLO("yolov8n.pt")
print("model info: ",model.info())
results = model.train(data="params.yaml", epochs=1, imgsz=640)
print("results: ",results)

if __name__ == "__main__":
    print("Script is executing.")
    
