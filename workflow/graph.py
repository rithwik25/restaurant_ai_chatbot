from zeal.backend.handlers.intent_handlers import handle_restaurant_recommendation, handle_restaurant_info, handle_casual_conversation
from zeal.backend.handlers.query_analyzer import analyze_user_query
from zeal.backend.handlers.router import route_query
from zeal.backend.models.data_models import ChatState
from zeal.backend.logger import logger
from zeal.backend.memory.conversation import ConversationMemory, CONVERSATION_MEMORY
from zeal.backend.llm.llm_interface import get_llm
from zeal.backend.config import OPENAI_API_KEY

import time
import queue
import threading
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# Main workflow graph definition
def create_restaurant_assistant_graph() -> StateGraph:
    """
    Creates the main workflow graph for the restaurant chatbot
    
    Returns:
        A StateGraph object representing the workflow
    """
    # Define the workflow
    workflow = StateGraph(ChatState)
    
    # Add nodes to the graph
    workflow.add_node("analyze_query", analyze_user_query)
    workflow.add_node("restaurant_recommendation", handle_restaurant_recommendation)
    workflow.add_node("restaurant_info", handle_restaurant_info)
    workflow.add_node("casual_conversation", handle_casual_conversation)
    
    # Add edges
    workflow.add_conditional_edges(
        "analyze_query",
        route_query,
        {
            "restaurant_recommendation": "restaurant_recommendation",
            "restaurant_info": "restaurant_info",
            "casual_conversation": "casual_conversation"
        }
    )
    
    # Set completion paths
    workflow.add_edge("restaurant_recommendation", END)
    workflow.add_edge("restaurant_info", END)
    workflow.add_edge("casual_conversation", END)
    
    # Set the entry point
    workflow.set_entry_point("analyze_query")
    
    return workflow.compile()

# Create an application function to handle incoming messages
def handle_message(message, session_id=None, stream=False):
    """
    Handle an incoming message from a user
    
    Args:
        message (str): The user's message
        session_id (str, optional): A unique session identifier
        stream (bool, optional): Whether to stream the response
        
    Returns:
        If stream=False: str with the complete response
        If stream=True: Generator that yields tokens one by one
    """
    # Default session ID if none provided
    if not session_id:
        session_id = str(int(time.time()))
    
    # Initialize streaming queue if needed
    response_queue = queue.Queue() if stream else None
    
    # Create or get the compiled graph
    graph = create_restaurant_assistant_graph()
    
    # Initialize the state
    state = ChatState(
        messages=[HumanMessage(content=message)],
        intent=None,
        user_preferences={"cuisine_type": [], "food_type": [], "location": "", "special_features": []},
        specific_restaurant=None,
        restaurant_matches=None,
        conversation_history=None,
        session_id=session_id
    )

    # Get streaming LLM if streaming is requested
    if stream:
        # Override the get_llm function result in the global namespace
        global get_llm
        original_get_llm = get_llm
        
        def streaming_get_llm(*args, **kwargs):
            return original_get_llm(streaming=True, queue=response_queue, *args, **kwargs)
        
        get_llm = streaming_get_llm # Temporarily replaces get_llm with our streaming version
    
    # Starts processing in a separate thread if streaming
    if stream:
        def response_generator(): # generator to yield streaming tokens
            # Start a thread to process the request
            def process_request():
                nonlocal graph, state
                try:
                    result = graph.invoke(state)
                    response_queue.put(None) # Signal completion
                    
                    # Get the final response for memory storage
                    response = result["messages"][-1].content if result["messages"] else "I'm not sure how to respond to that."
                    
                    CONVERSATION_MEMORY.add_interaction( # Stores the interaction in memory
                        session_id=session_id,
                        user_message=message,
                        bot_response=response,
                        metadata={
                            "intent": result.get("intent"),
                            "preferences": result.get("user_preferences")
                        }
                    )
                except Exception as e:
                    logger.error(f"Error in streaming process: {e}")
                    response_queue.put(None)  # Signal completion even on error
                
                # Restore the original get_llm function
                global get_llm
                get_llm = original_get_llm
                
            # Start processing thread
            threading.Thread(target=process_request).start()
            
            # Yield tokens as they arrive
            while True:
                token = response_queue.get()
                if token is None:  # End of response
                    break
                yield token
        
        # Return the generator
        return response_generator()

    else:
        # Non-streaming mode - execute synchronously
        result = graph.invoke(state)
        
        # Get the final response
        response = result["messages"][-1].content if result["messages"] else "I'm not sure how to respond to that."
        
        # Store the interaction in memory
        CONVERSATION_MEMORY.add_interaction(
            session_id=session_id,
            user_message=message,
            bot_response=response,
            metadata={
                "intent": result.get("intent"),
                "preferences": result.get("user_preferences")
            }
        )
        
        return response
