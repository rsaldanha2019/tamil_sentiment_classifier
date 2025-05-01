from setuptools import setup, find_packages
import os
import subprocess
import sys
from pathlib import Path
import shutil

PACKAGE_NAME = "tamil_sentiment_classifier"
MODEL_FOLDER = "saved_models"
GDRIVE_URL = "https://drive.google.com/drive/u/1/folders/14x1UdKTLEaCh8--WTt_TaEkOjqf3tF0A"

def ensure_clean_model_dir():
    model_dir = Path(__file__).parent / PACKAGE_NAME / MODEL_FOLDER
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def download_models(model_dir):
    print(f"Downloading models into: {model_dir}")
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    gdown.download_folder(
        url=GDRIVE_URL,
        output=str(model_dir),
        quiet=False,
        use_cookies=False
    )

def list_model_files():
    model_dir = Path(PACKAGE_NAME) / MODEL_FOLDER
    return [f"{MODEL_FOLDER}/{f.name}" for f in model_dir.glob("*.pt")]

# --- Run before setup ---
model_dir = ensure_clean_model_dir()
download_models(model_dir)

# --- Read requirements ---
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# --- Setup ---
setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    author="Richard Saldanha",
    author_email="rsaldanha554@gmail.com",
    description="A Python package for Tamil topic & sentiment classification using tamil-sentiment-classifier 3.2 3B",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rsaldanha2019/tamil_sentiment_classifier",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        PACKAGE_NAME: list_model_files(),
    },
    entry_points={
        "console_scripts": [
            "tamil-sentiment-classifier-cli=tamil_sentiment_classifier.cli:main",
            "tamil-sentiment-classifier-gui=tamil_sentiment_classifier.gui:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
