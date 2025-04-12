import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

class LlamaClassifier:
    def __init__(self):
        login(token="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")
        MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

        # 4-bit quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        ).to(self.device)

    def classify(self, text):
        # Optimized concise prompt
        prompt = (
            f"System: You are a structured response generator. You will not explain, describe, or add commentary.\n"
            f"User:\n"
            f"Given the following Tamil or code-mixed Tamil-English text, do the following:\n"
            f"1. Identify all relevant topics from this list: Politics, Entertainment, Sports, Technology, General.\n"
            f"2. Classify each topic’s sentiment as Positive, Negative, Neutral, or Unknown.\n"
            f"3. Extract key words/phrases contributing to that topic and sentiment.\n"
            f"4. Determine the Overall_Topic and Overall_Sentiment based on majority.\n\n"
            f"⚠️ Very Important: You MUST output in this exact format and nothing else:\n"
            f"{{\n"
            f"  Topic_Sentiments: {{\n"
            f"    1: {{Topic: \"<Topic>\", Sentiment: \"<Sentiment>\", Words: \"<comma-separated words>\"}},\n"
            f"    2: {{Topic: \"<Topic>\", Sentiment: \"<Sentiment>\", Words: \"<comma-separated words>\"}},\n"
            f"    ...\n"
            f"    Overall_Topic: \"<Overall_Topic>\",\n"
            f"    Overall_Sentiment: \"<Overall_Sentiment>\"\n"
            f"  }}\n"
            f"}}\n\n"
            f"Text: {text}\n\n"
            f"Output only the above format. Do not break format. Do not explain anything."
        )

        # Tokenization
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Model inference with optimized settings
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and clean the output
        result = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Extract only the relevant classification result
        if "Output:" in result:
            result = result.split("Output:")[-1].strip()

        # Define required fields for structured output
        required_fields = ["Topic:", "Sentiment:", "Words:", "Overall_Topic:", "Overall_Sentiment:"]
        formatted_result = []

        # Extract necessary fields or provide defaults if missing
        for field in required_fields:
            match = re.search(rf"{field} (.+)", result)
            formatted_result.append(f"{field} {match.group(1).strip()}" if match else f"{field} Unknown")

        # Format output cleanly
        formatted_output = "\n".join(formatted_result)

        # Truncate long input text for better display
        input_display = text if len(text) <= 100 else text[:100] + "..."

        return f"Input Text: {input_display}\n\n{formatted_output}"


