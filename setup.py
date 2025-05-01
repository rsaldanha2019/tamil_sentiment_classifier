from setuptools import setup, find_packages
import os
import pathlib
import subprocess
import sys

# Read dependencies from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# Ensure saved_models exists or download from Google Drive
def download_saved_models():
    print("Checking for 'saved_models' folder...")
    model_dir = pathlib.Path(__file__).parent / "tamil_sentiment_classifier" / "saved_models"

    if not model_dir.exists() or not any(model_dir.iterdir()):
        print("Downloading saved models from Google Drive...")

        try:
            import gdown
        except ImportError:
            print("Installing 'gdown' package...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        model_dir.mkdir(parents=True, exist_ok=True)

        # Google Drive folder download
        gdown.download_folder(
            url="https://drive.google.com/drive/u/1/folders/14x1UdKTLEaCh8--WTt_TaEkOjqf3tF0A",
            output=str(model_dir),
            quiet=False,
            use_cookies=False,
        )
    else:
        print("'saved_models' folder already exists. Skipping download.")

# Run model download before setup
download_saved_models()

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
