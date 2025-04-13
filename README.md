# Tamil Sentiment Classifier

A Python package for Tamil topic and sentiment classification using models like LLaMA, MuRIL, XLM-R, and IndicBERT.  
Supports both GUI and CLI, and handles code-mixed Tamil/English input.

**Note:** This package works only on **Ubuntu** and requires **Python 3.11**

## Installation

### Using Conda (Recommended)

```bash
conda create -n tamil_sentiment python=3.11 -y  
conda activate tamil_sentiment
```

Then install the package:

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

### LLaMA Output
```json
{
  "model": "llama",
  "response": {
    "Topic_Sentiments": [
      {
        "Topic": "Entertainment",
        "Sentiment": "Positive",
        "Words": "Tamil, film, Suriya, thriller, twist"
      }
    ],
    "Overall_Topic": "Entertainment",
    "Overall_Sentiment": "Positive"
  }
}
```

### BERT-family Output (muril, xlmr, indicbert)
```json
{
  "model": "muril",
  "response": {
    "sentiment": "Positive"
  }
}
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

- **llama** – Fast and efficient, no explainability  
- **muril** – Multilingual, supports explainability  
- **xlmr** – Strong multilingual model, supports explainability  
- **indicbert** – Indic-focused model, supports explainability  

### Download Pretrained Model Files

[Google Drive Link](https://drive.google.com/drive/u/1/folders/14x1UdKTLEaCh8--WTt_TaEkOjqf3tF0A)

## Features

- Sentiment and topic classification  
- Tamil and Tanglish input support  
- Tanglish-to-Tamil transliteration  
- LIME-based explanation (where supported)  
- CLI and GUI interfaces

## Author

Email: rsaldanha554@gmail.com
