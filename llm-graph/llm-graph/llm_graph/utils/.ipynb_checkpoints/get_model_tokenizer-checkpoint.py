from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_model_tokenizer():
    model_name = "google/gemma-2b"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
