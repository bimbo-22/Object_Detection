import argparse 
import os 
import tensorflow as tf 
from object_detection import model_lib_v2
from object_detection.builders import model_builder
from object_detection.utils import config_util  
import mlflow
from dotenv import load_dotenv
import yaml

load_dotenv()
params = yaml.safe_load(open('params.yaml'))['train']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

def update_pipeline_config(pipeline_config, mode, optimized_params = None):
    train_config = pipeline_config['train_config']
    train_config["fine_tune_checkpoint_type"] = "detection"
    
    pipeline_config['model']['ssd']['num_classes'] = 5
    
    if mode == "initial":
        print("Training with default hyperparameters")
    elif mode == "fine-tune":
        train_config["optimizer"]["momentum_optimizer"]["learning_rate"]["cosine_restart_learning_rate"]["initial_learning_rate"] = 0.01
    elif mode == "optimized" and optimized_params:
                        train_config["optimizer"]["momentum_optimizer"]["learning_rate"]["cosine_decay_learning_rate"]["learning_rate_base"] = optimized_params.get("learning_rate", 0.01)
                        train_config["batch_size"] = optimized_params.get("batch_size", 32)
    return pipeline_config



def train_model(mode, config_path, model_dir, num_steps):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    run_name = f"SSD Training ({mode}) on custom dataset"
    with mlflow.start_run(run_name=run_name):
        pipeline_config = config_util.get_configs_from_pipeline_file(config_path)
        train_config = pipeline_config['train_config']

        if mode == "optimized":
            with open(config_path, "r") as f:
                optimized_params = yaml.safe_load(f)
            pipeline_config = update_pipeline_config(pipeline_config, mode, optimized_params)
        else:
            pipeline_config = update_pipeline_config(pipeline_config, mode)
        
        
        os.makedirs(model_dir, exist_ok=True)
        update_config_path = os.path.join(model_dir, "pipeline.config")
        config_util.save_pipeline_config(pipeline_config, update_config_path)
        print(f"Pipeline config saved to {update_config_path}")
        mlflow.log_artifact(config_path, artifact_path="config")
        
        mlflow.log_param({
            "mode": mode,
            "num_steps": num_steps,
            "model_dir": model_dir
        })
        
        model_lib_v2.train_loop(
            pipeline_config_path = update_config_path,
            model_dir = model_dir,
            train_steps = num_steps,
            use_tpu = False,
            checkpoint_every_n = 1000,
            record_summaries = True
            )
        
        mlflow.log_artifact(model_dir, artifact_path="model")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSD model in different modes")
    parser.add_argument("--mode", choices=["initial", "fine-tune", "optimized"], required=True)
    parser.add_argument("--config_path", default = "src/SSD/pipeline.config", type=str, help="path to config file")
    parser.add_argument("--model_dir", default = "models/ssd/initial_model", type=str, help="path to save model")
    parser.add_argument("--num_steps", default = 50000, type=int, help="number of training steps")
    args = parser.parse_args()
    
    print(f"Starting {args.mode} training on custom dataset")
    train_model(args.mode, args.config_path, args.model_dir, args.num_steps)