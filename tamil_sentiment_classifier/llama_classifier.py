import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


class LlamaClassifier:
    def __init__(self):
        # Login to Hugging Face (you can set your own token here)
        login(token="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")

        BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Topics and sentiments - dynamic
        self.topics = ["Politics", "Entertainment", "Sports", "Technology", "General"]
        self.sentiments = ["Positive", "Negative", "Neutral"]

        # Load model with or without quantization
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

    def build_prompt(self, text):
        topic_list = ", ".join(self.topics)
        return (
            f"EXTRACT relevant topics ONLY from the following list: {topic_list}.\n"
            f"For each relevant topic, respond using the format below:\n"
            f"Topic: <Topic>\n"
            f"Sentiment: <Positive | Negative | Neutral>\n"
            f"Words: <1 to 5 comma-separated keywords from the text>\n\n"
            f"At the end, return:\n"
            f"Overall_Topic: <Topic>\n"
            f"Overall_Sentiment: <Sentiment>\n\n"
            f"DO NOT explain or repeat instructions.\n"
            f"DO NOT return empty or irrelevant topics.\n\n"
            f"Text: {text}"
        )

    def classify(self, text):
        prompt = self.build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=500,
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
        topic_pattern = re.compile(r"^\s*Topic:\s*(.+)", re.IGNORECASE)
        sentiment_pattern = re.compile(r"^\s*Sentiment:\s*(.+)", re.IGNORECASE)
        words_pattern = re.compile(r"^\s*Words:\s*(.+)", re.IGNORECASE)
        overall_topic_pattern = re.compile(r"^Overall_Topic:\s*(.+)", re.IGNORECASE)
        overall_sentiment_pattern = re.compile(r"^Overall_Sentiment:\s*(.+)", re.IGNORECASE)

        def is_valid(value):
            return bool(value and value.strip() and not value.strip().startswith("<"))

        for line in lines:
            line = line.strip()

            if topic_match := topic_pattern.match(line):
                if (
                    current
                    and is_valid(current.get("Topic"))
                    and is_valid(current.get("Sentiment"))
                    and is_valid(current.get("Words"))
                ):
                    topics.append(current)

                current = {
                    "Topic": topic_match.group(1).strip(),
                    "Sentiment": "",
                    "Words": ""
                }

            elif sentiment_match := sentiment_pattern.match(line):
                if current:
                    current["Sentiment"] = sentiment_match.group(1).strip()

            elif words_match := words_pattern.match(line):
                if current:
                    current["Words"] = words_match.group(1).strip()

            elif overall_topic_match := overall_topic_pattern.match(line):
                overall["Overall_Topic"] = overall_topic_match.group(1).strip()

            elif overall_sentiment_match := overall_sentiment_pattern.match(line):
                overall["Overall_Sentiment"] = overall_sentiment_match.group(1).strip()

        # Append the last topic if valid
        if (
            current
            and is_valid(current.get("Topic"))
            and is_valid(current.get("Sentiment"))
            and is_valid(current.get("Words"))
        ):
            topics.append(current)

        return {
            "Topic_Sentiments": topics,
            **overall
        }

