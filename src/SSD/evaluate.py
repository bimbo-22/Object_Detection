import argparse
import tensorflow as tf
from models.research.object_detection import model_lib_v2
from models.research.object_detection.utils import config_util
import mlflow
import os
import yaml
from dotenv import load_dotenv

load_dotenv()
params = yaml.safe_load(open('params.yaml'))['train']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

def evaluate_model(model_dir, config_path):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(run_name="Evaluating SSD on Unified dataset"):
        pipeline_config = config_util.get_configs_from_pipeline_file(config_path)
        
        mlflow.log_param("model_dir", model_dir)
        mlflow.log_param("config_path", config_path)
        
        model_lib_v2.eval_continuously(
            pipeline_config_path = config_path,
            model_dir = model_dir,
            checkpoint_dir = model_dir,
            wait_interval = 180,
            eval_dir = os.path.join(model_dir, "eval"),
            all_metrics = True,
            timeout = None
        )
        
        
        with open("results/SSD/evaluations.txt", "w") as f:
            f.write("Evaluation completed successfully")
            mlflow.log_artifact("results/SSD/evaluations.txt", artifact_path="evaluations")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SSD model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    print(f"Evaluating model in {args.model_dir} with config file {args.config_path}")
    evaluate_model(args.model_dir, args.config_path)