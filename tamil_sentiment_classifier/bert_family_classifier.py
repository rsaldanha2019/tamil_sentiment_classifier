import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
import numpy as np
import os
import re


class Model(nn.Module):
    def __init__(self, text_model, num_labels):
        super(Model, self).__init__()
        self.text_model = text_model
        self.fc = nn.Linear(text_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        text_embed = self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state[:, 0, :]
        logits = self.fc(text_embed)
        return logits


class BertFamilyClassifier:
    def __init__(self, model_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type.lower()

        # Supported model configs
        self.hf_models = {
            "muril": "google/muril-base-cased",
            "xlmr": "xlm-roberta-base",
            "indicbert": "ai4bharat/indic-bert"
        }

        if self.model_type not in self.hf_models:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose from: {list(self.hf_models.keys())}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_models[self.model_type])
        base_model = AutoModel.from_pretrained(self.hf_models[self.model_type])
        self.model = Model(text_model=base_model, num_labels=4)

        # Dynamically find matching checkpoint in saved_models
        ckpt_file = self._find_checkpoint_file(self.model_type)
        model_path = os.path.join("tamil_sentiment_classifier", "saved_models", ckpt_file)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.label_map = {
            0: "Mixed_feelings",
            1: "Negative",
            2: "Positive",
            3: "Unknown"
        }

    def _find_checkpoint_file(self, model_type):
        ckpt_dir = os.path.join("tamil_sentiment_classifier", "saved_models")
        for file in os.listdir(ckpt_dir):
            if model_type in file.lower() and file.endswith(".pt"):
                return file
        raise FileNotFoundError(f"No checkpoint found for model type '{model_type}' in {ckpt_dir}")

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
            "model": self.model_type,
            "input_text": text,
            "sentiment": sentiment
        }

    def get_probs(self, text):
        inputs = self.preprocess([text])
        with torch.inference_mode():
            logits = self.model(**inputs)
            probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        return {self.label_map[i]: prob for i, prob in enumerate(probs[0])}

    def predict_proba(self, texts, batch_size=32):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.preprocess(batch)
            with torch.inference_mode():
                logits = self.model(**inputs)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    def explain(self, text, num_samples=1000, num_features=10):
        class_names = list(self.label_map.values())
        explainer = LimeTextExplainer(class_names=class_names, split_expression=r'\W+')

        # Get predicted class index and label
        probs = self.predict_proba([text])[0]
        pred_class_idx = int(np.argmax(probs))
        pred_class_label = self.label_map[pred_class_idx]

        # Generate explanation for all classes (use range(len(class_names)) instead of None)
        explanation = explainer.explain_instance(
            text,
            self.predict_proba,
            num_samples=num_samples,
            num_features=num_features,
            labels=range(len(class_names))  # Focus on all classes
        )

        # Get (word, weight) contributions for each class
        all_contributions = {}
        for class_idx in range(len(class_names)):
            word_weights = explanation.as_list(label=class_idx)
            all_contributions[class_names[class_idx]] = word_weights

        return {
            "label": pred_class_label,
            "class_idx": pred_class_idx,
            "contributions": all_contributions
        }


