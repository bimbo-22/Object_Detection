import argparse
import os
import tensorflow as tf
import sys
import yaml
from dotenv import load_dotenv
import mlflow

from models.research.object_detection import model_lib_v2
from models.research.object_detection.utils import config_util
from models.research.object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Load environment variables and parameters
load_dotenv()
params = yaml.safe_load(open('params.yaml'))

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

def update_pipeline_config(pipeline_config, mode, num_steps, pretrained_model_dir=None, optimized_params=None):
    train_config = pipeline_config.train_config
    
   
    train_config.fine_tune_checkpoint_type = "detection"
    
    
    pipeline_config.model.ssd.num_classes = 5
    
    # Access the learning rate config
    lr_config = train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate
    
    # Always set total_steps to match num_steps for proper decay scheduling
    lr_config.total_steps = num_steps
    print(f"Setting total_steps to {num_steps}")
    
    if mode == "fine-tune":
        if pretrained_model_dir:
            latest_checkpoint = tf.train.latest_checkpoint(pretrained_model_dir)
            if latest_checkpoint:
                train_config.fine_tune_checkpoint = latest_checkpoint
                print(f"Setting fine_tune_checkpoint to {latest_checkpoint}")
            else:
                print(f"Warning: No checkpoint found in {pretrained_model_dir}")
        else:
            print("Warning: pretrained_model_dir not provided for fine-tuning")
        
        # Adjust learning rates for fine-tuning
        lr_config.learning_rate_base = 0.01
        lr_config.warmup_learning_rate = 0.001  # Ensure < learning_rate_base
        lr_config = pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate
        lr_config.total_steps = num_steps
        print("Setting learning_rate_base to 0.01 and warmup_learning_rate to 0.00333 for fine-tuning")
    
    elif mode == "initial":
        print("Training with default hyperparameters")
    
    elif mode == "optimized" and optimized_params:
        lr_config.learning_rate_base = optimized_params.get("learning_rate", 0.01)
        train_config.batch_size = optimized_params.get("batch_size", 16)
    
    return pipeline_config

def train_model(mode, config_path, model_dir, num_steps, pretrained_model_dir=None):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    run_name = f"SSD Training ({mode}) on custom dataset"
    with mlflow.start_run(run_name=run_name):
        # Load the pipeline configuration
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(config_path, "r") as f:
            text_format.Merge(f.read(), pipeline_config)

        # Update the configuration based on mode
        if mode == "optimized":
            optimized_params = params['SSD'].get('train', {}).get('optimized', {})
            pipeline_config = update_pipeline_config(pipeline_config, mode, pretrained_model_dir, optimized_params)
        else:
            pipeline_config = update_pipeline_config(pipeline_config, mode, num_steps, pretrained_model_dir)

        # Prepare model directory and save updated config
        os.makedirs(model_dir, exist_ok=True)
        updated_config_path = os.path.join(model_dir, "pipeline.config")
        config_util.save_pipeline_config(pipeline_config, model_dir)
        print(f"Pipeline config saved to {updated_config_path}")
        mlflow.log_artifact(config_path, artifact_path="config")

        # Log training parameters
        mlflow.log_params({
            "mode": mode,
            "num_steps": num_steps,
            "model_dir": model_dir,
            "batch_size": pipeline_config.train_config.batch_size
        })

        # Start training
        model_lib_v2.train_loop(
            pipeline_config_path=updated_config_path,
            model_dir=model_dir,
            train_steps=num_steps,
            use_tpu=False,
            checkpoint_every_n=1000,
            record_summaries=True
        )

        # Log the trained model
        mlflow.log_artifact(model_dir, artifact_path="model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SSD model in different modes")
    parser.add_argument("--mode", choices=["initial", "fine-tune", "optimized"], required=True,
                        help="Training mode: initial, fine-tune, or optimized")
    parser.add_argument("--config_path", default="src/SSD/pipeline.config", type=str,
                        help="Path to pipeline config file")
    parser.add_argument("--model_dir", default="model/SSD/fine_tuned_model", type=str,
                        help="Directory to save the trained model")
    parser.add_argument("--pretrained_model_dir", type=str,
                        help="Directory containing pre-trained checkpoints for fine-tuning")
    parser.add_argument("--num_steps", default=50000, type=int,
                        help="Number of training steps")
    args = parser.parse_args()

    print(f"Starting {args.mode} training on custom dataset")
    train_model(args.mode, args.config_path, args.model_dir, args.num_steps, args.pretrained_model_dir)