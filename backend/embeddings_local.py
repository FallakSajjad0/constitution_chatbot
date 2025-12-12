# embeddings_local.py
import os
import logging
from typing import List
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddings:
    """Local embedding model for text similarity"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): 
        """
        Initialize local embedding model
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        logger.info(f"🔄 Loading local embedding model: {model_name}")
        
        try:
            # Load the model
            self.model = SentenceTransformer(model_name)
            
            # Test the model
            test_embedding = self.model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            
            logger.info(f"✅ Model '{model_name}' loaded successfully")
            logger.info(f"📏 Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model '{model_name}': {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        logger.info(f"📤 Embedding {len(texts)} documents locally...")
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True if len(texts) > 10 else False)
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"❌ Failed to embed documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        logger.info(f"🔍 Embedding query: {text[:50]}...")
        
        try:
            embedding = self.model.encode([text], show_progress_bar=False)[0]
            logger.info(f"✅ Query embedded")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"❌ Failed to embed query: {str(e)}")
            raise

# For backward compatibility
def get_local_embeddings():
    """Get local embeddings instance"""
    return LocalEmbeddings()