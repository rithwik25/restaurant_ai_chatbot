from zeal.backend.logger import logger
from zeal.backend.models.data_models import ChatState
from zeal.backend.memory.cache import get_cached_response, set_cached_response
from zeal.backend.memory.conversation import CONVERSATION_MEMORY
from zeal.backend.database.vector_store import setup_retriever_with_persistence
from zeal.backend.llm.llm_interface import get_llm

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
# filter on the basis of the id of the restaurant

from zeal.backend.logger import logger
from zeal.backend.models.data_models import ChatState
from zeal.backend.memory.cache import get_cached_response, set_cached_response
from zeal.backend.memory.conversation import CONVERSATION_MEMORY
from zeal.backend.database.vector_store import setup_retriever_with_persistence
from zeal.backend.llm.llm_interface import get_llm

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

def handle_restaurant_recommendation(state: ChatState) -> ChatState:
    """
    Handles restaurant recommendation queries by searching the vector database
    and returning matching restaurants, with filtering based on extended criteria
    
    Args:
        state: The current chat state
        
    Returns:
        Updated state with restaurant recommendations
    """
    
    session_id = state.get("session_id", "unknown_session")
    logger.info(f"Processing restaurant recommendation for session {session_id}")
    
    search_criteria = state.get("user_preferences", {})
    logger.debug(f"Search criteria: {search_criteria}")
    
    # Build a rich query from the search criteria
    query_parts = []
    last_message = state["messages"][-1].content if state["messages"] else ""
    query_parts.append(last_message)
    
    # Add specific criteria
    if search_criteria.get("cuisine_type"):
        cuisines = search_criteria["cuisine_type"]
        if isinstance(cuisines, list):
            query_parts.append(f"Cuisine types: {', '.join(cuisines)}")
        else:
            query_parts.append(f"Cuisine type: {cuisines}")
    
    if search_criteria.get("food_type"):
        food_types = search_criteria["food_type"]
        if isinstance(food_types, list):
            query_parts.append(f"Food types: {', '.join(food_types)}")
        else:
            query_parts.append(f"Food type: {food_types}")
    
    if search_criteria.get("location"):
        query_parts.append(f"Location: {search_criteria['location']}")
    
    if search_criteria.get("special_features"):
        special_features = search_criteria["special_features"]
        if isinstance(special_features, list):
            query_parts.append(f"Special features: {', '.join(special_features)}")
        else:
            query_parts.append(f"Special feature: {special_features}")
    
    search_query = " ".join(query_parts) # Builds the complete query
    logger.info(f"Built search query: {search_query[:100]}...")
    
    retriever = setup_retriever_with_persistence(r"C:\Users\Rithwik Khera\OneDrive - iitr.ac.in\Desktop\assignment\zeal\100_restaurant_data.json", r"C:\Users\Rithwik Khera\OneDrive - iitr.ac.in\Desktop\assignment\zeal\restaurant_idx")
    logger.debug("Retriever setup complete")

    # Search for matching restaurants
    # Check cache first
    cache_key = f"recommendation_{search_query}"
    cached_matches = get_cached_response(cache_key)
    
    if cached_matches:
        logger.info("Using cached restaurant matches")
        all_matches = cached_matches
    else:
        # Perform the search
        logger.info("Performing vector search for restaurants")
        try:
            # Retrieve 5 results from vector search
            results = retriever.invoke(search_query, top_k=5)
            logger.debug(f"Retrieved {len(results)} results from vector search")

            # Tracking unique restaurant IDs to avoid duplicates using set data structure
            seen_restaurant_ids = set()
            unique_matches = []

            for doc in results:
                metadata = doc.metadata
                restaurant_id = metadata.get("id", "")
                
                # Only add this restaurant if we haven't seen it before
                if restaurant_id and restaurant_id not in seen_restaurant_ids:
                    seen_restaurant_ids.add(restaurant_id)
                    unique_matches.append({
                        "name": metadata.get("name", "Unknown Restaurant"),
                        "id": restaurant_id,
                        "content": doc.page_content,
                        "price": metadata.get("price"),
                        "restaurant_url": metadata.get("restaurant_url"),
                        "images_url": metadata.get("images_url"),
                        "coordinates": metadata.get("coordinates"),
                        "original_data": metadata.get("original_data", {})
                    })
                    
                    # Stop after finding 3 unique restaurants
                    if len(unique_matches) >= 3:
                        break
            
            # Cache and use the unique matches
            all_matches = unique_matches
            set_cached_response(cache_key, all_matches)
        except Exception as e:
            logger.error(f"Error during restaurant search: {e}", exc_info=True)
            all_matches = []
    
    # Updates the state with the matches
    state["restaurant_matches"] = all_matches

    # Get chat history
    chat_history = []
    if "session_id" in state and state["session_id"]:
        history = CONVERSATION_MEMORY.get_history(state["session_id"], limit=3)
        for item in history:
            chat_history.append(HumanMessage(content=item["user_message"]))
            chat_history.append(AIMessage(content=item["bot_response"]))

    user_context = f"""
        User query: {search_query}
        
        User's search criteria:
        {search_criteria}
        
        Available restaurant matches:
        {all_matches}  # Only using unique restaurant matches (maximum 3)
    """
    
    # Generate a response using an LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
                You are a restaurant recommendation assistant. Your task is to recommend restaurants based on the user's preferences and the retrieved restaurant data.
                
                Format your response precisely as follows:
                1. Begin with a brief, friendly introduction (1-2 sentences only)
                2. Present each restaurant recommendation as a numbered point
                3. For each restaurant point, use this exact structure:
                
                🍽️ [RESTAURANT NAME]
                • Cuisine: [cuisine type]
                • Price: [price range]
                • Notable features: [key features that match user preferences]
                • Why it matches: [brief explanation of how it meets the user's criteria]
                
                4. End with a single, brief follow-up question about whether these recommendations are helpful.
                
                IMPORTANT: Do not recommend the same restaurant more than once, even if it appears multiple times in the data. Check restaurant "id" carefully and ensure each recommendation is for a unique restaurant. If you've already suggested a restaurant with a particular "id", do not suggest it again even if it has different details.

                Keep your response concise and well-structured with clear formatting for easy readability.
            """
        ),
        *chat_history,
        HumanMessage(content=user_context)
    ])
    
    try:
        llm = get_llm(temperature=0.2)
        chain = prompt | llm
        
        logger.info("Sending recommendation request to LLM")
        response = chain.invoke({})
        logger.debug(f"Received LLM response of length {len(response.content)}")
        
        # Add the response to the messages
        state["messages"].append(AIMessage(content=response.content))
        logger.info("Added restaurant recommendation response to state")
        
    except Exception as e:
        logger.error(f"Error generating restaurant recommendation: {e}", exc_info=True)
        error_msg = "I'm sorry, I'm having trouble finding restaurant recommendations right now. Could you please try again or provide more details about what you're looking for?"
        state["messages"].append(AIMessage(content=error_msg))
    
    return state 

def handle_restaurant_info(state: ChatState) -> ChatState:
    """
    Handles queries about specific restaurants by searching for that restaurant
    and providing detailed information
    
    Args:
        state: The current chat state
        
    Returns:
        Updated state with specific restaurant information
    """
    
    session_id = state.get("session_id", "unknown_session")
    logger.info(f"Processing specific restaurant info for session {session_id}")

    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Build a query focused on the restaurant name
    query_parts = [last_message]
    
    if state.get("specific_restaurant"):
        restaurant_names = state["specific_restaurant"]
        if isinstance(restaurant_names, list):
            query_parts.append(f"Restaurant name: {', '.join(restaurant_names)}")
            logger.debug(f"Looking for specific restaurants: {', '.join(restaurant_names)}")
        else:
            query_parts.append(f"Restaurant name: {restaurant_names}")
            logger.debug(f"Looking for specific restaurant: {restaurant_names}")
    
    # Build the complete query
    search_query = " ".join(query_parts)
    logger.info(f"Built restaurant info query: {search_query[:100]}...")
    
    # Set up retriever if not already done
    retriever = setup_retriever_with_persistence(r"C:\Users\Rithwik Khera\OneDrive - iitr.ac.in\Desktop\assignment\zeal\100_restaurant_data.json", r"C:\Users\Rithwik Khera\OneDrive - iitr.ac.in\Desktop\assignment\zeal\restaurant_idx")  # Assuming the file path
    
    # Search for the restaurant
    # Check cache first
    cache_key = f"info_{search_query}"
    cached_matches = get_cached_response(cache_key)
    
    if cached_matches:
        logger.info("Using cached restaurant info matches")
        matches = cached_matches
    else:
        # Perform the search
        logger.info("Performing vector search for specific restaurant info")
        # Retrieve more results initially to ensure we can find at least 3 unique restaurants
        results = retriever.invoke(search_query, top_k=5)
        logger.debug(f"Retrieved {len(results)} results from vector search")

        # Tracking unique restaurant IDs to avoid duplicates using set data structure
        seen_restaurant_ids = set()
        unique_matches = []
        
        # Process the results
        for doc in results:
            metadata = doc.metadata
            restaurant_id = metadata.get("id", "")
            
            # Only add this restaurant if we haven't seen it before
            if restaurant_id and restaurant_id not in seen_restaurant_ids:
                seen_restaurant_ids.add(restaurant_id)
                unique_matches.append({
                    "name": metadata.get("name", "Unknown Restaurant"),
                    "id": restaurant_id,
                    "content": doc.page_content,
                    "price": metadata.get("price"),
                    "restaurant_url": metadata.get("restaurant_url"),
                    "images_url": metadata.get("images_url"),
                    "coordinates": metadata.get("coordinates"),
                    "original_data": metadata.get("original_data", {})
                })
                
                # Stop after finding 3 unique restaurants
                if len(unique_matches) >= 3:
                    break

        # Cache and use the unique matches
        matches = unique_matches
        set_cached_response(cache_key, matches)
            
    # Update the state with the matches
    state["restaurant_matches"] = matches
    
    # Get chat history
    chat_history = []
    if "session_id" in state and state["session_id"]:
        history = CONVERSATION_MEMORY.get_history(state["session_id"], limit=3)
        for item in history:
            chat_history.append(HumanMessage(content=item["user_message"]))
            chat_history.append(AIMessage(content=item["bot_response"]))
    
    user_context = f"""
        User query: {search_query}
        
        Search criteria: {state.get("specific_restaurant", [])}
        
        Restaurant matches: {matches}  # Using all unique matches (maximum 3)
    """

    # Generate a response using an LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
                You are a restaurant information assistant. Based on the user's query about a specific restaurant,
                provide detailed information in a structured, point-by-point format.
                
                Format your response precisely as follows:
                
                If you can identify ONE specific restaurant the user is asking about:
                
                🍽️ [RESTAURANT NAME]
                • Cuisine: [cuisine type]
                • Price: [price range]
                • Location: [location details]
                • Highlights: [key features, specialties, or popular dishes]
                • Hours: [if available]
                • Contact: [if available]
                • [Any other specific information the user requested]
                
                If MULTIPLE restaurants match and you're unsure which one:
                1. Start with a brief note mentioning you found multiple matches
                2. For each restaurant, provide a brief summary using the format above
                3. Ask which specific restaurant they'd like more details about
                
                Keep your response concise with clear, consistent formatting and structure.
            """
        ),
        *chat_history,
        HumanMessage(content=user_context)
    ])
    
    
    llm = get_llm(temperature=0.2)
    chain = prompt | llm
    response = chain.invoke({})
    
    # Adding the response to the messages
    state["messages"].append(AIMessage(content=response.content))
    return state

def handle_casual_conversation(state: ChatState) -> ChatState:
    """
    Handles casual conversation with the user
    
    Args:
        state: The current chat state
        
    Returns:
        Updated state with a casual response
    """
    last_message = state["messages"][-1].content if state["messages"] else ""

    # Get chat history
    chat_history = []
    if "session_id" in state and state["session_id"]:
        history = CONVERSATION_MEMORY.get_history(state["session_id"], limit=3)
        for item in history:
            chat_history.append(HumanMessage(content=item["user_message"]))
            chat_history.append(AIMessage(content=item["bot_response"]))
    
    # Generate a casual response using an LLM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
                You are a friendly restaurant assistant chatbot. Respond naturally to casual conversation,
                greetings, thanks, or general questions. Be friendly, helpful, and conversational.
                
                If the conversation shifts to restaurants, pivot to offering structured help:
                
                "I can help you find restaurants based on:
                • Cuisine type
                • Location
                • Price range
                • Special features (outdoor seating, vegan options, etc.)
                
                Just let me know what you're looking for!"
                
                Keep casual responses brief and engaging. If the user is asking a non-restaurant question,
                still be helpful but gently remind them that you specialize in restaurant recommendations
                and information.
            """
        ),
        *chat_history,
        HumanMessage(content=last_message)
    ])
    
    # Use the LLM to generate a response
    llm = get_llm(temperature=0.2)
    chain = prompt | llm
    
    response = chain.invoke({})
    
    # Add the response to the messages
    state["messages"].append(AIMessage(content=response.content))
    return state