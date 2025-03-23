"""
Intent classification and information extraction for the restaurant agent.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from zeal.backend.models.data_models import ChatState
from zeal.backend.llm.llm_interface import get_llm
from zeal.backend.memory.conversation import CONVERSATION_MEMORY
from zeal.backend.memory.cache import get_cached_response, set_cached_response
from zeal.backend.logger import logger

def analyze_user_query(state: ChatState) -> ChatState:
    """
    Combined function to analyze the latest user query:
    1. Classifies the intent into restaurant_recommendation, specific_restaurant_info, or casual_conversation
    2. Extracts relevant information like cuisine type, location, price, etc.
    
    Args:
        state: The current chat state containing messages and other context
        
    Returns:
        Updated state with intent classification and extracted information
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    session_id = state.get("session_id", "unknown_session")
    
    logger.info(f"Analyzing user query for session {session_id}")
    logger.debug(f"User query: {last_message[:100]}...")
    
    # Checks cache first for both intent and info extraction
    cache_key = f"analysis_{last_message}"
    cached_analysis = get_cached_response(cache_key)
    if cached_analysis:
        logger.info("Using cached analysis result")
        state.update(cached_analysis)
        return state
    
    # Get conversation history for context
    conversation_history = []
    if "session_id" in state and state["session_id"]:
        logger.debug(f"Retrieving conversation history for session {session_id}")
        history = CONVERSATION_MEMORY.get_history(state["session_id"], limit=5)
        conversation_history = [
            {"user": item["user_message"], "bot": item["bot_response"]} 
            for item in history
        ]
    
    history_context = "\n".join([
        f"User: {item['user']}\nBot: {item['bot']}" 
        for item in conversation_history
    ])
    
    # Create a prompt for combined intent and information extraction
    logger.debug("Creating prompt for intent classification and info extraction")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""
        Analyze the user's message for a restaurant chatbot by:

        1. CLASSIFYING THE INTENT into exactly one of these categories:
           - restaurant_recommendation: User is looking for restaurant suggestions
           - specific_restaurant_info: User is asking about a specific restaurant or set of restaurants
           - casual_conversation: General greetings, farewells, or off-topic conversation

        2. EXTRACTING INFORMATION relevant to their request (include only if mentioned or implied):
           - cuisine_type: Type of cuisine (e.g., Chinese, Italian, Japanese)
           - food_type: Specific food or dish (e.g., pasta, sushi, pizza)
           - location: City, neighborhood, area, address, cross street, country etc.
           - special_features: Any special requirements (e.g., outdoor dining/areaseating, payment options etc.)
           - restaurant_name: List of "Names" of specific restaurants if mentioned, only implies when the INTENT of the query is specific_restaurant_info 

        Be interpretive - if user says "nice Italian place in NYC", extract the cuisine_type(Italian), location(NYC) and a rating(nice).
        
        Recent conversation history (consider this for context):
        {history_context}
        
        Respond with a JSON object containing both "intent" and "extracted_info" fields.
        
        Example response format:
        {{
            "intent": "restaurant_recommendation" OR "specific_restaurant_info" OR "casual_conversation",
            "extracted_info": {{
                "cuisine_type": ["list of cuisines mentioned or empty list"],
                "food_type": ["list of food types mentioned or empty list"],
                "location": "location mentioned or empty string if none",
                "special_features": ["list of special features mentioned or empty list"],
                "restaurant_names": ["list of restaurant names mentioned or empty list"],
            }}
        }}
        
        Only include fields that are explicitly mentioned or clearly implied in the user's message and return a strictly JSON response with no additional text as shown above in the response format.
        """),
        HumanMessage(content=last_message)
    ])
    
    # Using the LLM to analyze the query
    logger.info("Sending query to LLM for analysis")
    llm = get_llm(temperature=0)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({})
        logger.debug(f"Received analysis result: {result}")
        
        state["intent"] = result.get("intent", "casual_conversation")
        logger.info(f"Classified intent: {state['intent']}")
        
        extracted_info = result.get("extracted_info", {})
        logger.debug(f"Extracted information: {extracted_info}")
        
        # Initialize or update user_preferences
        if "user_preferences" not in state:
            state["user_preferences"] = {
                "cuisine_type": [],
                "food_type": [],
                "location": "",
                "special_features": []
            }
        
        # Update user preferences with extracted information
        if "cuisine_type" in extracted_info and extracted_info["cuisine_type"]:
            state["user_preferences"]["cuisine_type"] = extracted_info["cuisine_type"]
        
        if "food_type" in extracted_info and extracted_info["food_type"]:
            state["user_preferences"]["food_type"] = extracted_info["food_type"]
        
        if "location" in extracted_info and extracted_info["location"]:
            state["user_preferences"]["location"] = extracted_info["location"]
        
        if "special_features" in extracted_info and extracted_info["special_features"]:
            state["user_preferences"]["special_features"] = extracted_info["special_features"]
        
        # Handle restaurant names for specific_restaurant_info intent
        if state["intent"] == "specific_restaurant_info" and "restaurant_names" in extracted_info:
            state["specific_restaurant"] = extracted_info.get("restaurant_names", [])
            logger.debug(f"Set specific restaurant: {state['specific_restaurant']}")
        
        # Cache the analysis for future use
        set_cached_response(cache_key, {
            "intent": state["intent"],
            "user_preferences": state["user_preferences"],
            "specific_restaurant": state.get("specific_restaurant", None)
        })
        logger.info("Analysis complete and cached")
        
    except Exception as e:
        # Logs error and continues with default values
        logger.error(f"Error parsing LLM response: {e}", exc_info=True)
        state["intent"] = "casual_conversation"
        logger.info("Defaulting to casual_conversation intent due to error")
    
    return state