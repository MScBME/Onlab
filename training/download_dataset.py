"""Download the annotated dataset from Roboflow for local training.

Usage:
    python training/download_dataset.py

Configuration is read from .env file in the project root.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")
VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))

if not all([API_KEY, WORKSPACE, PROJECT]):
    print("Error: Missing Roboflow config. Fill in .env file at the project root.")
    print("Required: ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT")
    sys.exit(1)

DATASET_DIR = os.path.join(TRAINING_DIR, "dataset")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

print(f"Downloading dataset to: {DATASET_DIR}")
dataset = version.download("yolov11", location=DATASET_DIR)
print("Done!")
