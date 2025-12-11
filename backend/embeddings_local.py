# embeddings_local.py
from sentence_transformers import SentenceTransformer
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddings:
    """Local embeddings using sentence-transformers (NO API NEEDED)"""
    
    def __init__(self, model_name="all-mpnet-base-v2"):
        try:
            logger.info(f"🔄 Loading local embedding model: {model_name}")
            
            # Load the model (will download automatically first time)
            self.model = SentenceTransformer(f'sentence-transformers/{model_name}')
            
            # Test the model
            test_embedding = self.model.encode(["test"])
            embedding_dim = test_embedding.shape[1]
            
            logger.info(f"✅ Model '{model_name}' loaded successfully")
            logger.info(f"📏 Embedding dimension: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            logger.info("💡 Install: pip install sentence-transformers")
            raise
    
    def embed_documents(self, texts):
        """Embed multiple documents"""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        logger.info(f"📤 Embedding {len(texts)} documents locally...")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=False,  # Return numpy array
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False,
                batch_size=32
            )
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings.tolist()  # Convert to list for ChromaDB
            
        except Exception as e:
            logger.error(f"❌ Embedding failed: {str(e)}")
            raise
    
    def embed_query(self, text):
        """Embed a single query"""
        logger.info(f"🔍 Embedding query: {text[:50]}...")
        result = self.embed_documents([text])[0]
        logger.info("✅ Query embedded")
        return result

# For backward compatibility with your existing code
HuggingFaceEmbeddings = LocalEmbeddings