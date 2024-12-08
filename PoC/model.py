import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


def config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct", device_map="cuda:3", token=None
):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config(),
        device_map=device_map,
        max_length=4096,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return base_model, tokenizer


def settings():
    return {
        "max_length": 350,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": True,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }


class Llama3Generator:
    def __init__(self, model, tokenizer, device="cuda:3"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_simple(self, prompt, gen_settings, completion_only=True):
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)

        output = self.model.generate(
            input_ids,
            max_length=gen_settings.get("max_length", 100),
            temperature=gen_settings.get("temperature", 0.7),
            top_p=gen_settings.get("top_p", 0.9),
            top_k=gen_settings.get("top_k", 50),
            repetition_penalty=gen_settings.get("repetition_penalty", 1.2),
            do_sample=gen_settings.get("do_sample", True),
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt) :] if completion_only else response
