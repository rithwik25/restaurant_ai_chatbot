"""
Restaurant data loading and preprocessing for the restaurant agent.
"""
import json
from functools import lru_cache
from typing import Dict, List, Any
from langchain.schema import Document
from zeal.backend.logger import logger
from zeal.backend.config import RESTAURANTS_JSON_PATH

@lru_cache(maxsize=1)  
def load_restaurants(json_file_path: str = RESTAURANTS_JSON_PATH) -> List[Dict[str, Any]]:
    """
    Load restaurant data from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing restaurant data
        
    Returns:
        List of restaurant dictionaries
    """
    try:
        logger.info(f"Loading restaurant data from {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as file:
            restaurants = json.load(file)
        logger.info(f"Successfully loaded {len(restaurants)} restaurants from the database")
        return restaurants
    except Exception as e:
        logger.error(f"Error loading restaurant data: {e}", exc_info=True)
        return []


def prepare_restaurant_docs(restaurants: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert restaurant data into document format for vector storage.
    
    Args:
        restaurants: List of restaurant dictionaries
        
    Returns:
        List of Document objects
    """
    logger.info(f"Preparing document representations for {len(restaurants)} restaurants")
    docs = []

    for i, restaurant in enumerate(restaurants):
        if i % 50 == 0 and i > 0:
            logger.debug(f"Processed {i} restaurants so far")
        
        text_content = f"Restaurant Name: {restaurant.get('name', '')}\n"
        
        # Location information
        city = restaurant.get('city', '')
        state = restaurant.get('state', '')
        neighborhood = restaurant.get('neighborhood', '')
        address = restaurant.get('street_address', '')
        zipcode = restaurant.get('zipcode', '')
        country = restaurant.get('country', '')
        cross_street = restaurant.get('cross_street', '')
        
        location_parts = []
        if address:
            location_parts.append(address)
        if neighborhood:
            location_parts.append(f"Neighborhood: {neighborhood}")
        if cross_street:
            location_parts.append(f"Cross Street: {cross_street}")
        if city:
            location_parts.append(city)
        if state:
            location_parts.append(state)
        if country:
            location_parts.append(country)
        if zipcode:
            location_parts.append(zipcode)
        
        location_str = ", ".join(location_parts)
        text_content += f"Location: {location_str}\n"
        
        # Rating and reviews
        rating = restaurant.get('rating')
        review_count = restaurant.get('review_count')
        if rating is not None:
            text_content += f"Rating: {rating}"
        if review_count is not None:
            text_content += f" (from {review_count} reviews)"
            text_content += "\n"
        
        # Price information
        price = restaurant.get('price')
        payment_options = restaurant.get('payment_options', [])
        if price is not None:
            text_content += f"Price Level: {price}\n"
        if payment_options:
            text_content += f"Payment Options: {', '.join(payment_options)}\n"
        
        # Cuisines
        cuisines = restaurant.get('cuisines', [])
        if cuisines:
            text_content += f"Cuisines: {', '.join(cuisines)}\n"
        
        # Tags (for additional food types, ambiance, etc.)
        tags = restaurant.get('tags', [])
        if tags:
            text_content += f"Tags: {', '.join(tags)}\n"
        
        # Popular dishes
        popular_dishes = restaurant.get('popular_dishes', [])
        if popular_dishes:
            text_content += f"Popular Dishes: {', '.join(popular_dishes)}\n"
        
        # Description or endorsement
        description = restaurant.get('description')
        endorsement = restaurant.get('endorsement_copy')
        if description:
            text_content += f"Description: {description}\n"
        elif endorsement:
            text_content += f"Description: {endorsement}\n"
        
        # Featured in publications
        featured_in = restaurant.get('featured_in')
        if featured_in:
            text_content += f"Featured in: {featured_in}\n"
        
        # Contact details
        phone_number = restaurant.get('phone_number', '')
        restaurant_url = restaurant.get('restaurant_url', '')
        if phone_number and restaurant_url:
            text_content += f"Phone number is {phone_number} and restaurant url is {restaurant_url}."
        elif phone_number:
            text_content += f"Phone number is {phone_number}."
        elif restaurant_url:
            text_content += f"The restaurant url is {restaurant_url}."
        
        # Additional amenities
        if restaurant.get('reservations_required') is True:
            text_content += "Reservations required.\n"
        
        if restaurant.get('dining_style'):
            text_content += f"Dining style: {restaurant.get('dining_style')}\n"
        
        if restaurant.get('parking_details'):
            text_content += f"Parking: {restaurant.get('parking_details')}\n"
        
        if restaurant.get('public_transport'):
            text_content += f"Public transport: {restaurant.get('public_transport')}\n"
        
        # Create document for vectorstore with rich metadata
        metadata = {
            "id": restaurant.get("id") if restaurant.get("id") else None,
            "name": restaurant.get("name") if restaurant.get("name") else None,
            "location": location_str,
            "price": restaurant.get("price") if restaurant.get("price") else None,
            "restaurant_url": restaurant.get("restaurant_url") if restaurant.get("restaurant_url") else None,
            "images_url": restaurant.get("images_url") if restaurant.get("images_url") else None,
            "coordinates": restaurant.get("location_geom", {}).get("coordinates") if restaurant.get("location_geom") else None,
            "original_data": restaurant
        }
        docs.append(Document(page_content=text_content, metadata=metadata))
    
    logger.info(f"Finished preparing {len(docs)} restaurant documents")
    return docs