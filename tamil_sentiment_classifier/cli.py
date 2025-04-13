import argparse
import sys

from tamil_sentiment_classifier.llama_classifier import LlamaClassifier
from tamil_sentiment_classifier.bert_family_classifier import BertFamilyClassifier

# Unified model map using BertFamilyClassifier
MODEL_MAP = {
    "llama": LlamaClassifier,
    "muril": lambda: BertFamilyClassifier("muril"),
    "xlmr": lambda: BertFamilyClassifier("xlmr"),
    "indicbert": lambda: BertFamilyClassifier("indicbert"),
}

class UnifiedClassifierCLI:
    def __init__(self, model_type):
        if model_type not in MODEL_MAP:
            raise ValueError(f"Unsupported model: {model_type}. Choose from: {list(MODEL_MAP.keys())}")
        self.model_type = model_type
        self.classifier = MODEL_MAP[model_type]()

    def classify_text(self, text, explain=False):
        """Classify the given Tamil or Code-Mixed text."""
        prediction = self.classifier.classify(text)
        print("\n=== Prediction ===")
        print(prediction)

        if explain:
            print("\n=== Explanation ===")
            explanation = self.classifier.explain(text)
            for word, weight in explanation.as_list():
                color = "green" if weight > 0 else "red"
                print(f"{word}: {weight:.4f} ({color})")

def main():
    parser = argparse.ArgumentParser(description="Unified Tamil Sentiment Classifier CLI")
    parser.add_argument("--model", type=str, default="llama", help="Model to use: llama / muril / xlmr / indicbert")
    parser.add_argument("--input_text", type=str, required=True, help="Tamil or Code-Mixed input text")
    parser.add_argument("--explain", action="store_true", help="Enable LIME/attention explanation")

    args = parser.parse_args()

    # Explanation is only supported for transformer models
    if args.model == "llama" and args.explain:
        print("Explanation is not supported for the 'llama' model. Disabling --explain.")
        args.explain = False

    try:
        classifier_cli = UnifiedClassifierCLI(model_type=args.model)
        classifier_cli.classify_text(args.input_text, explain=args.explain)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
