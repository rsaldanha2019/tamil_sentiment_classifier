import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
import numpy as np

class Model(nn.Module):
    def __init__(self, text_model, num_labels):
        super(Model, self).__init__()
        self.text_model = text_model
        self.fc = nn.Linear(text_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        text_embed = self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state[:, 0, :]
        logits = self.fc(text_embed)
        return logits

class MurilClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

        base_model = AutoModel.from_pretrained("google/muril-base-cased")
        self.model = Model(text_model=base_model, num_labels=4)

        state_dict = torch.load("tamil_sentiment_classifier/saved_models/muril-xai.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.label_map = {
            0: "Positive",
            1: "Unknown",
            2: "Negative",
            3: "Mixed_feelings"
        }

    def preprocess(self, text_list):
        return self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

    def classify(self, text):
        inputs = self.preprocess([text])
        with torch.inference_mode():
            logits = self.model(**inputs)
        prediction = torch.argmax(logits, dim=1).item()
        sentiment = self.label_map.get(prediction, "Unknown")

        return {
            "model": "muril",
            "input_text": text,
            "sentiment": sentiment
        }

    def get_probs(self, text):
        """Method to get the probabilities for each sentiment class."""
        inputs = self.preprocess([text])
        with torch.inference_mode():
            logits = self.model(**inputs)
            probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        
        # Return a dictionary with sentiment labels and their corresponding probabilities
        probs_dict = {self.label_map[i]: prob for i, prob in enumerate(probs[0])}
        return probs_dict

    def explain(self, text, num_samples=300, num_features=5):
        class_names = list(self.label_map.values())
        explainer = LimeTextExplainer(class_names=class_names, split_expression=r'\W+')

        def predict_proba(texts, batch_size=32):
            all_probs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.preprocess(batch)
                with torch.inference_mode():
                    logits = self.model(**inputs)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.extend(probs)
            return np.array(all_probs)

        explanation = explainer.explain_instance(
            text,
            predict_proba,
            num_samples=num_samples,
            num_features=num_features
        )

        print("\n=== LIME Explanation ===")
        for feature, weight in explanation.as_list():
            print(f"{feature}: {weight:.4f}")

        return explanation
