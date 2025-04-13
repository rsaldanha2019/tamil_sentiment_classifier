# Tamil Sentiment Classifier

A Python package for Tamil topic and sentiment classification using models like LLaMA, MuRIL, XLM-R, and IndicBERT.  
Supports both GUI and CLI, and handles code-mixed Tamil/English text input.

## Installation

### 1. Create and Activate a Virtual Environment

#### Using Conda

```bash
conda create -n tamil_sentiment python=3.10 -y  
conda activate tamil_sentiment
```

#### Using venv

```bash
python -m venv tamil_sentiment  
# macOS / Linux  
source tamil_sentiment/bin/activate  
# Windows  
tamil_sentiment\Scripts\activate
```

### 2. Install the Package

```bash
pip install git+https://github.com/rsaldanha2019/tamil_sentiment_classifier.git
```

## Usage

### Run the GUI

```bash
tamil-sentiment-classifier-gui
```

### Run the CLI

```bash
tamil-sentiment-classifier-cli --input_text="எனக்கு தமிழ் பிடிக்கும்" --model muril --explain
```

## Example CLI Output

```bash
=== Prediction ===  
Topic: Entertainment  
Sentiment: Positive  

=== Explanation ====
Word: Value
```

## Development Setup

```bash
git clone https://github.com/rsaldanha2019/tamil_sentiment_classifier.git  
cd tamil_sentiment_classifier  
pip install -e .
```

## Uninstallation

```bash
pip uninstall tamil_sentiment_classifier
```

## Models Supported

- **llama** – Fast, efficient, but no explainability  
- **muril** – Multilingual, supports explainability  
- **xlmr** – Strong multilingual model, supports explainability  
- **indicbert** – Indic-based multilingual model, supports explainability  

## Model Files

You can download the pre-trained models from [this link](https://drive.google.com/drive/u/1/folders/14x1UdKTLEaCh8--WTt_TaEkOjqf3tF0A).

## Features

- Sentiment and topic classification  
- Supports Tamil and Tanglish input  
- Transliteration (Tanglish to Tamil)  
- LIME-based explanations (where supported)  
- GUI and CLI interfaces

## Author

Email: rsaldanha554@gmail.com
