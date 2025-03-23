"""
Configuration and environment variables for the restaurant agent.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(r"C:\Users\Rithwik Khera\OneDrive - iitr.ac.in\Desktop\assignment\zeal\.env")

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# File paths
RESTAURANTS_JSON_PATH = r"C:\Users\Rithwik Khera\OneDrive - iitr.ac.in\Desktop\assignment\zeal\100_restaurant_data.json"
FAISS_INDEX_DIR = r"C:\Users\Rithwik Khera\OneDrive - iitr.ac.in\Desktop\assignment\zeal\restaurant_idx"

# Cache settings
MAX_CACHE_ENTRIES = 100

# Logger settings
LOG_FILE = "restaurant_agent.log"
LOG_LEVEL = "INFO"