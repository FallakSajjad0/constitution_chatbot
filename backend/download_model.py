from transformers import AutoTokenizer, AutoModelForCausalLM

# Example: Download a causal language model
model_name = "facebook/opt-125m"  # or any other causal LM
save_directory = "backend/causal_model"  # Save to a different directory

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
