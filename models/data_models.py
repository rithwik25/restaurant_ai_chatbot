"""
TypedDict definitions and data models for the restaurant agent.
"""
from typing import Dict, List, Any, TypedDict, Union, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class UserPreferences(TypedDict):
    """User preferences for restaurant recommendations."""
    cuisine_type: Optional[List[str]]
    food_type: Optional[List[str]]
    location: str
    special_features: Optional[List[str]]  # special requirements (e.g., outdoor dining/area, payment_options, etc.)

class ChatState(TypedDict):
    """Chat state for the restaurant agent."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]  # list of messages exchanged in the chat 
    intent: Optional[str]  # intent of the user query
    user_preferences: UserPreferences 
    specific_restaurant: Optional[List[str]]  # multiple restaurants can be mentioned in the user query while asking for restaurant information
    restaurant_matches: Optional[List[Dict[str, Any]]]  # list of restaurant options that match the user's query
    conversation_history: Optional[List[Dict[str, Any]]]  # record of past conversations (for continuity)
    session_id: Optional[str]  # unique identifier for the chat session
