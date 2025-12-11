# llm_hf.py - Use HuggingFace Inference API for LLM
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class HFInferenceLLM:
    def __init__(self, model="microsoft/phi-2"):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in .env")
        
        self.api_url = f"https://api-inference.huggingface.co/models/{model}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def invoke(self, prompt: str) -> str:
        """Generate response using HF Inference API"""
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if 'generated_text' in result[0]:
                        return result[0]['generated_text']
                    else:
                        return str(result[0])
                return str(result)
            else:
                raise Exception(f"HF API Error {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")

# For compatibility
class Ollama(HFInferenceLLM):
    def __init__(self, model="llama3.1", **kwargs):
        # Map to a HF model
        hf_model = "microsoft/phi-2"  # Small, fast model
        super().__init__(model=hf_model)