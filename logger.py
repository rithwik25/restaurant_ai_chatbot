import logging
import sys
import codecs

# Ensure the console can handle UTF-8
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# Create a StreamHandler with explicit UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("restaurant_agent.log", encoding='utf-8'),
        console_handler
    ]
)

logger = logging.getLogger(__name__)