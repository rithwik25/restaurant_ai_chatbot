"""
FAISS indexing and retrieval functions for the restaurant agent.
"""
import os
from functools import lru_cache
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from zeal.backend.logger import logger
from zeal.backend.config import EMBEDDING_MODEL, RESTAURANTS_JSON_PATH, FAISS_INDEX_DIR
from zeal.backend.database.restaurant_loader import load_restaurants, prepare_restaurant_docs

from dotenv import load_dotenv
load_dotenv(".env")

def save_faiss_index(vector_store, directory_path: str) -> None:
    """
    Save a FAISS vector store to disk.
    
    Args:
        vector_store: The FAISS vector store to save
        directory_path: The directory path where the index will be saved
    """
    try:
        logger.info(f"Saving FAISS index to {directory_path}")
        vector_store.save_local(directory_path)
        logger.info(f"Successfully saved FAISS index to {directory_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}", exc_info=True)

def load_faiss_index(directory_path: str, embedding_model=None) -> FAISS:
    """
    Load a FAISS vector store from disk.
    
    Args:
        directory_path: The directory path where the index is stored
        embedding_model: The embedding model to use (optional if saved with the index)
    
    Returns:
        A FAISS vector store
    """
    try:
        logger.info(f"Loading FAISS index from {directory_path}")
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            logger.debug("Created new embedding model instance")
        
        vector_store = FAISS.load_local(directory_path, embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"Successfully loaded FAISS index from {directory_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}", exc_info=True)
        return None

def create_and_save_index(restaurants_json_path: str, index_dir: str) -> FAISS:
    """
    Creates a new FAISS index from restaurant data and saves it to disk.
    
    Args:
        restaurants_json_path: Path to the JSON file containing restaurant data
        index_dir: Directory to save the FAISS index
    
    Returns:
        A FAISS vector store
    """
    logger.info(f"Creating new FAISS index from {restaurants_json_path}")
    restaurants = load_restaurants(restaurants_json_path)
    logger.debug(f"Loaded {len(restaurants)} restaurants")
    
    docs = prepare_restaurant_docs(restaurants)
    logger.debug(f"Prepared {len(docs)} documents")
    
    logger.info("Creating embedding model and vector store")
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(docs, embeddings)
        logger.info("Successfully created FAISS vector store")
        
        # Save the index
        save_faiss_index(vector_store, index_dir)
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise

@lru_cache(maxsize=1)
def setup_retriever_with_persistence(restaurants_json_path: str = RESTAURANTS_JSON_PATH, 
                                    index_dir: str = FAISS_INDEX_DIR) -> VectorStoreRetriever:
    """
    Sets up a retriever using a persisted FAISS index if available, otherwise creates and saves a new index.
    
    Args:
        restaurants_json_path: Path to the JSON file containing restaurant data
        index_dir: Directory to save/load the FAISS index
    
    Returns:
        A VectorStoreRetriever
    """
    # Check if index exists
    if os.path.exists(index_dir) and os.path.isdir(index_dir):
        logger.info(f"Found existing FAISS index at {index_dir}")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = load_faiss_index(index_dir, embeddings)
        
        # If loading failed, create a new index
        if vector_store is None:
            logger.warning("Failed to load existing index. Creating a new one...")
            vector_store = create_and_save_index(restaurants_json_path, index_dir)
    else:
        logger.info(f"No existing index found at {index_dir}. Creating a new one...")
        vector_store = create_and_save_index(restaurants_json_path, index_dir)
    
    # Create and return the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Return top 5 matches
    )

    logger.info("Created and configured vector store retriever")
    return retriever
