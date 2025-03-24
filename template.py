import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "Human_Tracking"

list_of_files = [
    ".github/workflows/.gitkeep",
    
    # Data folders
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    "data/interim/.gitkeep",
    "data/README.md",
    
    # Notebooks
    "notebooks/EDA/.gitkeep",
    "notebooks/training/.gitkeep",
    "notebooks/inference/.gitkeep",
    
    # Source code
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/features/__init__.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/inference/__init__.py",
    f"src/{project_name}/visualization/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    
    # Configs
    "configs/model_config.yaml",
    "configs/data_config.yaml",
    "configs/hyperparams.yaml",
    "configs/logging.yaml",
    
    # Experiments
    "experiments/run1/.gitkeep",
    "experiments/run2/.gitkeep",
    "experiments/README.md",
    
    # Tests
    "tests/test_data.py",
    "tests/test_model.py",
    "tests/test_api.py",
    "tests/__init__.py",
    
    # Logs
    "logs/training.log",
    "logs/error.log",
    "logs/inference.log",
    
    # Scripts
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/predict.py",
    "scripts/data_pipeline.py",
    
    # Checkpoints
    "checkpoints/model_v1.pth",
    "checkpoints/model_v2.pth",
    "checkpoints/saved_model/.gitkeep",
    
    # Deployment
    "deployment/app.py",
    "deployment/Dockerfile",
    "deployment/requirements.txt",
    "deployment/README.md",
    "deployment/utils.py",
    
    # Docs
    "docs/architecture.png",
    "docs/pipeline.md",
    "docs/README.md",
    
    # MLflow & Monitoring
    "mlruns/.gitkeep",
    "monitoring/.gitkeep",
    
    # Research/Experiments
    "research/trials.ipynb",
    
    # Project Level Files
    "requirements.txt",
    "environment.yml",
    ".gitignore",
    "README.md",
    "LICENSE",
    "Makefile",
    "app.py",
    "main.py",
    "Dockerfile",
    "setup.py",
    "config/config.yaml",
    "params.yaml"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"✅ Directory created: {filedir} for the file {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, 'w') as f:
            f.write(f"# {filename}\n")
        logging.info(f"✅ File created: {filepath}")
    else:
        logging.info(f"⚠️ File already exists: {filepath}")