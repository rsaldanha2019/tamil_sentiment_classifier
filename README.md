# Tamil Sentiment Classifier

A Python package for Tamil topic and sentiment classification using models like LLaMA, MuRIL, and XLM-R.  
Supports both GUI and CLI, and handles code-mixed Tamil/English text input.

## Installation

### 1. Create and Activate a Virtual Environment

#### Using Conda

conda create -n tamil_sentiment python=3.11 -y  
conda activate tamil_sentiment

#### Using venv

python -m venv tamil_sentiment  
# macOS / Linux  
source tamil_sentiment/bin/activate  
# Windows  
tamil_sentiment\Scripts\activate

### 2. Install the Package

pip install git+https://github.com/rsaldanha2019/tamil_sentiment_classifier.git

## Usage

### Run the GUI

tamil-sentiment-classifier-gui

### Run the CLI

tamil-sentiment-classifier-cli "எனக்கு தமிழ் பிடிக்கும்"

## Example CLI Output

=== Prediction ===  
Topic: Movies, Language  
Sentiment: Positive  
Reason: The text expresses a liking towards Tamil movies or language.

## Development Setup

git clone https://github.com/rsaldanha2019/tamil_sentiment_classifier.git  
cd tamil_sentiment_classifier  
pip install -e .

## Uninstallation

pip uninstall tamil_sentiment_classifier

## Models Supported

- llama – Fast, efficient, but no explainability  
- muril – Multilingual, supports explainability  
- xlmr – Strong multilingual model, supports explainability

## Features

- Sentiment and topic classification  
- Supports Tamil and Tanglish input  
- Transliteration (Tanglish to Tamil)  
- LIME-based explanations (where supported)  
- GUI and CLI interfaces

# Tamil Sentiment Classifier

A Python package for Tamil topic and sentiment classification using models like LLaMA, MuRIL, and XLM-R.  
Supports both GUI and CLI, and handles code-mixed Tamil/English text input.

## Installation

### 1. Create and Activate a Virtual Environment

#### Using Conda

conda create -n tamil_sentiment python=3.11 -y  
conda activate tamil_sentiment

#### Using venv

python -m venv tamil_sentiment  
# macOS / Linux  
source tamil_sentiment/bin/activate  
# Windows  
tamil_sentiment\Scripts\activate

### 2. Install the Package

pip install git+https://github.com/rsaldanha2019/tamil_sentiment_classifier.git

## Usage

### Run the GUI

tamil-sentiment-classifier-gui

### Run the CLI

tamil-sentiment-classifier-cli --input_text="எனக்கு தமிழ் பிடிக்கும்" --model muril --explain

## Example CLI Output

=== Prediction ===  
Topic: Entertainment  
Sentiment: Positive  

=== Explanation ====
Word: Value

## Development Setup

git clone https://github.com/rsaldanha2019/tamil_sentiment_classifier.git  
cd tamil_sentiment_classifier  
pip install -e .

## Uninstallation

pip uninstall tamil_sentiment_classifier

## Models Supported

- llama – Fast, efficient, but no explainability  
- muril – Multilingual, supports explainability  

## Features

- Sentiment and topic classification  
- Supports Tamil and Tanglish input  
- Transliteration (Tanglish to Tamil)  
- LIME-based explanations (where supported)  
- GUI and CLI interfaces

## Author

Email: rsaldanha554@gmail.com  