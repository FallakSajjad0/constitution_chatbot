"""
LLM Online - Groq API with updated working models
Fixed: deepseek-r1-distill-llama-70b has been decommissioned
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

class OnlineLLM:
    def __init__(self, provider="groq"):  # Default to Groq
        self.provider = provider
        
        if provider == "groq":
            self._init_groq()
        elif provider == "openai":
            self._init_openai()
        elif provider == "gemini":
            self._init_gemini()
        elif provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_groq(self):
        """Initialize Groq with available models"""
        try:
            from groq import Groq
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            
            self.client = Groq(api_key=api_key)
            
            # Available Groq models (as of Dec 2024)
            # Check: https://console.groq.com/docs/models
            available_models = [
                "llama-3.3-70b-versatile",  # Fast, capable
                "llama-3.1-8b-instant",      # Very fast
                "llama-3.2-3b-preview",      # Lightweight
                "llama-3.2-1b-preview",      # Ultra fast
                "gemma2-9b-it",              # Good for reasoning
                "mixtral-8x7b-32768",        # Large context
            ]
            
            # Try models in order of preference
            for model in available_models:
                try:
                    # Test connection with minimal request
                    test = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    self.model = model
                    print(f"✅ Groq model loaded: {model}")
                    return
                except Exception as e:
                    print(f"⚠️ Model {model} failed: {str(e)[:80]}")
                    continue
            
            # If all fail, fallback to the first one
            self.model = available_models[0]
            print(f"⚠️ Using fallback model: {self.model}")
            
        except Exception as e:
            print(f"❌ Groq failed: {str(e)[:100]}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI"""
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY missing")
            
            self.client = OpenAI(api_key=api_key)
            
            # Try available models
            available_models = [
                "gpt-4o-mini",      # Fast, affordable
                "gpt-3.5-turbo",    # Legacy but reliable
                "gpt-4-turbo",      # More capable
            ]
            
            for model in available_models:
                try:
                    # Test connection
                    test = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    self.model = model
                    print(f"✅ OpenAI model loaded: {model}")
                    return
                except:
                    continue
            
            self.model = "gpt-3.5-turbo"
            print(f"⚠️ Using fallback OpenAI model: {self.model}")
            
        except Exception as e:
            raise ValueError(f"OpenAI failed: {str(e)[:100]}")
    
    def _init_gemini(self):
        """Initialize Google Gemini"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY missing")
            
            genai.configure(api_key=api_key)
            
            # Try available models
            available_models = [
                "gemini-1.5-flash",    # Fast
                "gemini-1.5-pro",      # More capable
                "gemini-1.0-pro",      # Legacy
            ]
            
            for model_name in available_models:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test
                    response = self.model.generate_content("test")
                    self.model_name = model_name
                    print(f"✅ Gemini model loaded: {model_name}")
                    return
                except:
                    continue
            
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.model_name = "gemini-1.5-flash"
            print(f"⚠️ Using fallback Gemini model: {self.model_name}")
            
        except Exception as e:
            raise ValueError(f"Gemini failed: {str(e)[:100]}")
    
    def _init_anthropic(self):
        """Initialize Anthropic Claude"""
        try:
            from anthropic import Anthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY missing")
            
            self.client = Anthropic(api_key=api_key)
            
            # Available models
            available_models = [
                "claude-3-haiku-20240307",    # Fast
                "claude-3-5-sonnet-20241022", # Balanced
                "claude-3-opus-20240229",     # Most capable
            ]
            
            self.model = available_models[0]  # Default to Haiku
            print(f"✅ Anthropic model loaded: {self.model}")
            
        except Exception as e:
            raise ValueError(f"Anthropic failed: {str(e)[:100]}")
    
    def invoke(self, prompt: str, max_retries: int = 3) -> str:
        """Invoke the LLM with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "groq":
                    return self._groq_invoke(prompt)
                elif self.provider == "openai":
                    return self._openai_invoke(prompt)
                elif self.provider == "gemini":
                    return self._gemini_invoke(prompt)
                elif self.provider == "anthropic":
                    return self._anthropic_invoke(prompt)
            except Exception as e:
                if attempt == max_retries:
                    # Last attempt, raise the error
                    raise RuntimeError(f"All {max_retries + 1} attempts failed: {str(e)}")
                
                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"⚠️ Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    def _groq_invoke(self, prompt: str) -> str:
        """Invoke Groq API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a constitutional law expert for Pakistan. Provide accurate, clear answers based on constitutional context. Cite relevant articles when possible."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,      # Low temperature for factual answers
            max_tokens=1024,      # Reasonable response length
            top_p=0.9,           # Nucleus sampling
        )
        return response.choices[0].message.content.strip()
    
    def _openai_invoke(self, prompt: str) -> str:
        """Invoke OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a constitutional law expert for Pakistan. Answer precisely and cite relevant articles."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    
    def _gemini_invoke(self, prompt: str) -> str:
        """Invoke Gemini API"""
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
                "top_p": 0.9,
            }
        )
        return response.text.strip()
    
    def _anthropic_invoke(self, prompt: str) -> str:
        """Invoke Anthropic API"""
        response = self.client.messages.create(
            model=self.model,
            system="You are a constitutional law expert for Pakistan.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        return response.content[0].text.strip()


def get_llm_provider(preference="groq"):
    """
    Get LLM instance with fallback logic
    
    Args:
        preference: Preferred provider ("groq", "openai", "gemini", "anthropic")
    
    Returns:
        OnlineLLM instance or None if all fail
    """
    providers = [preference]
    
    # Add fallback providers based on preference
    if preference == "groq":
        providers.extend(["openai", "gemini"])
    elif preference == "openai":
        providers.extend(["groq", "gemini"])
    elif preference == "gemini":
        providers.extend(["openai", "groq"])
    
    # Try each provider
    for provider in providers:
        try:
            llm = OnlineLLM(provider=provider)
            print(f"✅ Successfully initialized {provider.upper()} LLM")
            return llm
        except Exception as e:
            print(f"⚠️ {provider.upper()} failed: {str(e)[:80]}")
            continue
    
    print("❌ All LLM providers failed")
    return None


# Legacy compatibility
class Ollama:
    def __init__(self, model="llama3.1", **kwargs):
        # Map to available online models
        self.llm = get_llm_provider("groq")  # Try Groq first
        if not self.llm:
            raise RuntimeError("No working LLM provider found")
    
    def invoke(self, prompt: str) -> str:
        return self.llm.invoke(prompt)