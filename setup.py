from setuptools import setup, find_packages
import os
import subprocess
import sys
from pathlib import Path

# Ensure saved_models exists or download it
def ensure_models():
    model_dir = Path(__file__).parent / "tamil_sentiment_classifier" / "saved_models"
    if not model_dir.exists() or not any(model_dir.glob("*.pt")):
        print("Downloading saved models into:", model_dir)
        try:
            import gdown
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        model_dir.mkdir(parents=True, exist_ok=True)

        gdown.download_folder(
            url="https://drive.google.com/drive/u/1/folders/14x1UdKTLEaCh8--WTt_TaEkOjqf3tF0A",
            output=str(model_dir),
            quiet=False,
            use_cookies=False
        )
    else:
        print("Models already exist. Skipping download.")

# Call before setup
ensure_models()

# Read dependencies
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# Collect all .pt files under saved_models for package_data
def collect_model_files():
    model_dir = Path("tamil_sentiment_classifier/saved_models")
    return [f"saved_models/{p.name}" for p in model_dir.glob("*.pt")]

setup(
    name="tamil_sentiment_classifier",
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
        "tamil_sentiment_classifier": collect_model_files(),
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
