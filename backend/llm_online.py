# llm_online.py - UPDATED WITH WORKING MODEL
import os
import time
from dotenv import load_dotenv

load_dotenv()

class OnlineLLM:
    def __init__(self, provider="groq"):
        self.provider = provider
        
        if provider == "groq":
            self._init_groq()
        elif provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_groq(self):
        """Initialize Groq LLM with WORKING model"""
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        
        self.client = Groq(api_key=api_key)
        # ONLY THIS MODEL CURRENTLY WORKS:
        self.model = "llama-3.1-8b-instant"
        
        # Test the model
        try:
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print(f"✅ Groq model '{self.model}' is working")
        except Exception as e:
            raise ValueError(f"Groq model '{self.model}' failed: {str(e)[:100]}")
        
    def _init_openai(self):
        """Initialize OpenAI LLM"""
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
    
    def invoke(self, prompt: str, max_retries: int = 2) -> str:
        """Generate response from LLM"""
        for attempt in range(max_retries):
            try:
                if self.provider == "groq":
                    return self._groq_invoke(prompt)
                elif self.provider == "openai":
                    return self._openai_invoke(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 1 * (attempt + 1)
                print(f"⚠️ Retry {attempt+1} in {wait_time}s...")
                time.sleep(wait_time)
    
    def _groq_invoke(self, prompt: str) -> str:
        """Invoke Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that answers questions based on provided context."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                raise Exception("Rate limit exceeded. Please wait a moment.")
            elif "model_decommissioned" in error_msg:
                raise Exception(f"Model {self.model} is decommissioned. Please update the model name.")
            else:
                raise Exception(f"Groq API error: {error_msg[:200]}")
    
    def _openai_invoke(self, prompt: str) -> str:
        """Invoke OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

# For backward compatibility
class Ollama(OnlineLLM):
    """Compatibility class to replace Ollama"""
    def __init__(self, model="llama3.1", **kwargs):
        # Use Groq by default
        super().__init__(provider="groq")