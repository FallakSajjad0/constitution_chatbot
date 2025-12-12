# backend/llm_online.py - BEST SETUP: Groq (DeepSeek-R1) + Fallback to xAI Grok
import os
import time
from dotenv import load_dotenv

load_dotenv()

class OnlineLLM:
    def __init__(self, provider="groq"):  # Default to Groq + DeepSeek-R1
        self.provider = provider
        
        if provider == "groq":
            self._init_groq()
        elif provider == "xai":
            self._init_xai()
        elif provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_groq(self):
        """Initialize Groq with DeepSeek-R1 (BEST model for reasoning)"""
        try:
            from groq import Groq
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            
            self.client = Groq(api_key=api_key)
            self.model = "deepseek-r1-distill-llama-70b"  # BEST available model
            
            # Test connection
            test = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            print(f"Groq + DeepSeek-R1 model loaded successfully")
            
        except Exception as e:
            print(f"Groq failed: {str(e)[:100]}")
            raise
    
    def _init_xai(self):
        """Fallback: xAI Grok (if Groq is down)"""
        try:
            from xai_sdk import Client
            from xai_sdk.chat import user, system
            
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                raise ValueError("XAI_API_KEY missing")
            
            self.client = Client(api_key=api_key)
            self.model = "grok-4"
            self.system_msg = system
            self.user_msg = user
            
            # Test
            chat = self.client.chat.create(model=self.model)
            chat.append(self.system_msg("You are helpful."))
            chat.append(self.user_msg("test"))
            chat.sample()
            print(f"xAI Grok model loaded (fallback active)")
            
        except Exception as e:
            raise ValueError(f"xAI failed: {str(e)[:100]}")
    
    def _init_openai(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
        print("OpenAI GPT-3.5 loaded")

    def invoke(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "groq":
                    return self._groq_invoke(prompt)
                elif self.provider == "xai":
                    return self._xai_invoke(prompt)
                elif self.provider == "openai":
                    return self._openai_invoke(prompt)
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(f"Attempt {attempt + 1} failed, retrying in {2 ** attempt}s...")
                time.sleep(2 ** attempt)

    def _groq_invoke(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise constitutional law assistant for Pakistan. Answer clearly and accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()

    def _xai_invoke(self, prompt: str) -> str:
        chat = self.client.chat.create(model=self.model)
        chat.append(self.system_msg("You are a precise constitutional law assistant for Pakistan."))
        chat.append(self.user_msg(prompt))
        return chat.sample().content.strip()

    def _openai_invoke(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a constitutional law expert for Pakistan."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()


# Recommended usage
llm = OnlineLLM(provider="groq")  # This gives you DeepSeek-R1 (best reasoning)

# Legacy compatibility
class Ollama(OnlineLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(provider="groq")  # Always use Groq + DeepSeek