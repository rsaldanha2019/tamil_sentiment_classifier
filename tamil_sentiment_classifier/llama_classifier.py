import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from lime.lime_text import LimeTextExplainer
import numpy as np


class LlamaClassifier:
    def __init__(self):
        login(token="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")  # Replace with your token if needed

        BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map={"": self.device}
            ).to(self.device)

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def classify(self, text):
        prompt = (
            f"You are an extractor. Given a Tamil or mixed Tamil-English sentence, extract only the **relevant** topics "
            f"from the following: Politics, Entertainment, Sports, Technology, General. "
            f"For each relevant topic, return the Topic, Sentiment (Positive, Negative, Neutral), and Words "
            f"(comma-separated words related to the topic). "
            f"Finally, include Overall_Topic and Overall_Sentiment based on the entire text.\n\n"
            f"Text: {text}\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=300,
                        do_sample=False,
                        temperature=0.3,
                        repetition_penalty=1.1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        parsed = self.parse_response(output_text)

        return {
            "model": "llama",
            "response": parsed,
            # "output_text": output_text
        }

    def parse_response(self, text):
        topics = []
        current = {}
        overall = {}

        lines = text.strip().splitlines()
        topic_pattern = re.compile(r"^\*\s*\*\*(.+?)\*\*:\s*(Positive|Negative|Neutral)", re.IGNORECASE)
        words_pattern = re.compile(r"^\s*\*\s*Words:\s*(.+)", re.IGNORECASE)
        overall_topic_pattern = re.compile(r"^Overall_Topic:\s*(.+)", re.IGNORECASE)
        overall_sentiment_pattern = re.compile(r"^Overall_Sentiment:\s*(.+)", re.IGNORECASE)

        for line in lines:
            line = line.strip()

            if topic_match := topic_pattern.match(line):
                if current:
                    topics.append(current)
                current = {
                    "Topic": topic_match.group(1).strip(),
                    "Sentiment": topic_match.group(2).strip(),
                    "Words": ""
                }

            elif words_match := words_pattern.match(line):
                if current:
                    current["Words"] = words_match.group(1).strip()

            elif overall_topic_match := overall_topic_pattern.match(line):
                overall["Overall_Topic"] = overall_topic_match.group(1).strip()

            elif overall_sentiment_match := overall_sentiment_pattern.match(line):
                overall["Overall_Sentiment"] = overall_sentiment_match.group(1).strip()

        # Catch the last topic if present
        if current and current.get("Topic"):
            topics.append(current)

        return {
            "Topic_Sentiments": topics,
            **overall
        }

