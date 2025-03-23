"""
Query caching for the restaurant agent.
"""
import threading
from zeal.backend.logger import logger
from zeal.backend.config import MAX_CACHE_ENTRIES

# Query cache
QUERY_CACHE = {}  # dictionary to store cached search results 
QUERY_CACHE_LOCK = threading.Lock()  # A lock to prevent multiple users from modifying the cache at the same time

def get_cached_response(query_key):
    """
    Get a cached response for a query key.
    
    Args:
        query_key: The key for the cached response
        
    Returns:
        The cached response, or None if not found
    """
    with QUERY_CACHE_LOCK:
        if query_key in QUERY_CACHE:
            logger.debug(f"Cache hit for key: {query_key[:50]}...")
            return QUERY_CACHE.get(query_key)
        logger.debug(f"Cache miss for key: {query_key[:50]}...")
        return None

def set_cached_response(query_key, response):
    """
    Cache a response for a query key.
    
    Args:
        query_key: The key for the cached response
        response: The response to cache
    """
    with QUERY_CACHE_LOCK:
        QUERY_CACHE[query_key] = response
        logger.debug(f"Cached response for key: {query_key[:50]}...")

        # Limit cache size
        if len(QUERY_CACHE) > MAX_CACHE_ENTRIES:
            oldest_key = next(iter(QUERY_CACHE))  # Removes oldest entry
            logger.info(f"Cache limit reached. Removing oldest entry: {oldest_key[:50]}...")
            del QUERY_CACHE[oldest_key]
