"""
LLM initialization and management for the restaurant agent.
"""
import queue
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from zeal.backend.config import OPENAI_API_KEY, LLM_MODEL
from zeal.backend.logger import logger

# Global LLM cache
LLM_CACHE = {}  # llm_cache is a dictionary that stores AI model instances

# Define custom callback handler for streaming
class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses from the AI model."""
    def __init__(self, queue):
        """Initialize the callback handler with a queue for storing tokens."""
        self.queue = queue  # queue is used to stream responses in real-time
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called whenever the AI generates a new token."""
        self.queue.put(token)

# Initializes and retrieves the AI language model
def get_llm(temperature=0.2, streaming=False, queue=None):
    """
    Get an LLM instance, reusing cached instances when possible.
    
    Args:
        temperature: Temperature parameter for the LLM
        streaming: Whether to enable token-by-token streaming
        queue: Queue for streaming tokens (required if streaming=True)
        
    Returns:
        A configured LLM instance
    """
    cache_key = f"llm_{temperature}_{streaming}"
    if cache_key in LLM_CACHE:
        return LLM_CACHE[cache_key]
    
    callbacks = []
    if streaming and queue:
        callbacks.append(StreamingCallbackHandler(queue))
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        api_key=OPENAI_API_KEY,
        streaming=streaming,
        callbacks=callbacks if callbacks else None
    )
    
    LLM_CACHE[cache_key] = llm
    logger.info(f"Created new LLM instance with temperature={temperature}, streaming={streaming}")
    return llm
