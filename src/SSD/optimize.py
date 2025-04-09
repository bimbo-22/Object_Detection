import optuna
import tensorflow as tf
from models.research.object_detection import model_lib_v2
from models.research.object_detection.utils import config_util
import mlflow
import os
from dotenv import load_dotenv
import yaml

load_dotenv()
params = yaml.safe_load(open('params.yaml'))['train']

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.001, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    num_steps = trial.suggest_int("num_steps", 1000, 50000, step = 10000)
    
    config_path = "configs/SSD/ssd_pipeline.config"
    pipeline_config = config_util.get_configs_from_pipeline_file(config_path)
    train_config = pipeline_config['train_config']
    train_config["batch_size"] = batch_size
    train_config["optimizer"]["momentum_optimizer"]["learning_rate"]["cosine_restart_learning_rate"]["initial_learning_rate"] = learning_rate
    
    model_dir = f"models/SSD/trial_{trial.number}"
    os.makedirs(model_dir, exist_ok=True)
    config_util.save_pipeline_config(pipeline_config, model_dir)
    
    model_lib_v2.train_loop(
        pipeline_config_path = os.path.join(model_dir, "pipeline.config"),
        model_dir = model_dir,
        train_steps = num_steps,
        use_tpu = False,
        checkpoint_every_n = 1000,
        record_summaries = True
    )
    
if __name__ == "__main__":
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    with open("best_params.yaml", "w") as f:
        yaml.dump(best_params, f)
    print("Best parameters: ", best_params)