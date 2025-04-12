import argparse
from tamil_sentiment_classifier.llama_classifier import LlamaClassifier
from tamil_sentiment_classifier.muril_classifier import MurilClassifier  # assuming you have this
from tamil_sentiment_classifier.xlmr_classifier import XLMRClassifier
import sys

MODEL_MAP = {
    "llama": LlamaClassifier,
    "muril": MurilClassifier,
    "xlmr": XLMRClassifier
}

class UnifiedClassifierCLI:
    def __init__(self, model_type):
        if model_type not in MODEL_MAP:
            raise ValueError(f"Unsupported model: {model_type}. Choose from: {list(MODEL_MAP.keys())}")
        self.classifier = MODEL_MAP[model_type]()

    def classify_text(self, text, explain=False):
        """Classify the given Tamil or Code-Mixed text."""
        prediction = self.classifier.classify(text)
        print("\n=== Prediction ===")
        print(prediction)

        if explain:
            print("\n=== Explanation ===")
            self.classifier.explain(text)

def main():
    parser = argparse.ArgumentParser(description="Unified Tamil Topic & Sentiment Classifier")
    parser.add_argument("--model", type=str, default="llama", help="Model to use: llama / muril / xlmr")
    parser.add_argument("--input_text", type=str, required=True, help="Tamil or Code-Mixed input text")
    parser.add_argument("--explain", action="store_true", help="Enable LIME/attention explanation")

    args = parser.parse_args()

    # Disable explain for LLaMA model
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
