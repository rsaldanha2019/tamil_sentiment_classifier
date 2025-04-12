import torch
import torch.nn as nn
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from lime.lime_text import LimeTextExplainer
import numpy as np


class Model(nn.Module):
    def __init__(self, text_model, num_labels):
        super(Model, self).__init__()
        self.text_model = text_model
        self.fc = nn.Linear(text_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(text_embed)
        return logits


class XLMRClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load state dict from checkpoint
        state_dict_path = "tamil_sentiment_classifier/saved_models/xlmr-xai.pt"
        state_dict = torch.load(state_dict_path, map_location=self.device)

        # Extract vocab size and hidden size from state dict
        embedding_weight = state_dict["text_model.embeddings.word_embeddings.weight"]
        vocab_size, hidden_size = embedding_weight.shape

        # Update config with correct vocab_size and hidden_size
        config = AlbertConfig(
            vocab_size=vocab_size,    # Ensure this matches the vocab size from the checkpoint
            hidden_size=768,          # Update hidden_size to 768 (or use the size from the checkpoint)
            intermediate_size=hidden_size * 4,  # Typically 4 * hidden_size for ALBERT
            num_attention_heads=12,   # Default attention heads for ALBERT
            num_hidden_layers=12      # Default number of hidden layers
        )

        # Load tokenizer (must match training vocab)
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

        # Initialize the model with the updated config
        base_model = AlbertModel(config)
        self.model = Model(text_model=base_model, num_labels=4)

        # Load the weights from the checkpoint
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
            "model": "albert",
            "input_text": text,
            "sentiment": sentiment
        }

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
