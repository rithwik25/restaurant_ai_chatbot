"""
Query routing based on intent for the restaurant agent.
"""
from zeal.backend.models.data_models import ChatState
from zeal.backend.logger import logger

def route_query(state: ChatState) -> str:
    """
    Routes the query to the appropriate handler based on intent
    
    Args:
        state: The current chat state containing intent and other context
        
    Returns:
        String indicating which node to route to
    """
    intent = state.get("intent", "casual_conversation")
    session_id = state.get("session_id", "unknown_session")
    
    logger.info(f"Routing query for session {session_id} with intent: {intent}")
    
    if intent == "restaurant_recommendation":
        logger.debug("Routing to restaurant_recommendation handler")
        return "restaurant_recommendation"
    elif intent == "specific_restaurant_info":
        logger.debug("Routing to restaurant_info handler")
        return "restaurant_info"
    else:
        logger.debug("Routing to casual_conversation handler")
        return "casual_conversation"
