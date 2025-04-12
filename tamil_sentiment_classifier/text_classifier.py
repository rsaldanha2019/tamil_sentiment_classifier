class TextClassifier:
    def __init__(self, model_type="phi3"):
        if model_type == "phi3":
            from tamil_sentiment_classifier.llama_classifier import Phi3Classifier
            self.model = Phi3Classifier()
        elif model_type == "muril":
            from tamil_sentiment_classifier.muril_classifier import MurilClassifier
            self.model = MurilClassifier()
        elif model_type == "xlmr":
            from tamil_sentiment_classifier.xlmr_classifier import XLMRClassifier
            self.model = XLMRClassifier()
        else:
            raise ValueError("Unsupported model_type. Choose from: llama, muril")

    def classify(self, text):
        return self.model.classify(text)

    def explain(self, text):
        return self.model.explain(text)
