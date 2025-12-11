# embeddings_hf_fixed.py
import os
import requests
import time
from typing import List
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class WorkingHFEmbeddings:
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found")
        
        # This endpoint works with Classic tokens
        self.api_url = "https://router.huggingface.co"
        self.model = "sentence-transformers/all-mpnet-base-v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if API key has proper permissions"""
        logger.info("🔍 Testing API connection...")
        
        test_payload = {
            "inputs": ["test"],
            "model": self.model,
            "task": "feature-extraction"
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/hf-inference/models",
                json=test_payload,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ API connection successful")
            elif response.status_code == 403:
                logger.error("❌ API key lacks permissions")
                logger.info("💡 Create a NEW Classic token at: https://huggingface.co/settings/tokens")
                raise PermissionError("API key needs proper permissions")
            else:
                logger.warning(f"⚠️ API response: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with retry logic"""
        if isinstance(texts, str):
            texts = [texts]
        
        for attempt in range(3):
            try:
                # Use inference endpoint
                response = requests.post(
                    f"{self.api_url}/hf-inference/models",
                    json={
                        "inputs": texts,
                        "model": self.model,
                        "task": "feature-extraction"
                    },
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract embeddings from response
                    if isinstance(result, dict) and 'embeddings' in result:
                        return result['embeddings']
                    return result
                    
                elif response.status_code == 503:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"⏳ Model loading, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                    
                else:
                    logger.error(f"Error {response.status_code}: {response.text[:200]}")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == 2:
                    raise
        
        raise RuntimeError("All attempts failed")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]