"""
Conversation history management for the restaurant agent.
"""
from datetime import datetime
from zeal.backend.logger import logger

class ConversationMemory:
    """Storage and management for chat history between a user and a chatbot."""
    def __init__(self, max_sessions=50, max_history_per_session=15):
        """
        Initialize the conversation memory.
        
        Args:
            max_sessions: Maximum number of sessions to store
            max_history_per_session: Maximum number of interactions per session
        """
        self.sessions = {}  # stores conversations per session
        self.max_sessions = max_sessions
        self.max_history_per_session = max_history_per_session  # stores up to 15 messages per session
        logger.info(f"Initialized ConversationMemory with max_sessions={max_sessions}, max_history={max_history_per_session}")
        
    def add_interaction(self, session_id, user_message, bot_response, metadata=None):
        """
        Add a new user-bot interaction to memory.
        
        Args:
            session_id: Unique identifier for the chat session
            user_message: Message from the user
            bot_response: Response from the bot
            metadata: Optional additional data
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            logger.info(f"Created new session: {session_id}")
            
        # Limit sessions
        if len(self.sessions) > self.max_sessions:
            oldest_session = min(self.sessions.keys(), key=lambda k: self.sessions[k][0]['timestamp'] if self.sessions[k] else datetime.now().timestamp())
            logger.info(f"Session limit reached. Removing oldest session: {oldest_session}")
            del self.sessions[oldest_session]
        
        # Add the new interaction
        interaction = {
            'timestamp': datetime.now().timestamp(),
            'user_message': user_message,
            'bot_response': bot_response,
            'metadata': metadata or {}
        }
        
        self.sessions[session_id].append(interaction)
        logger.debug(f"Added interaction to session {session_id}. Message length: User={len(user_message)}, Bot={len(bot_response)}")
        
        # Trim history if needed
        if len(self.sessions[session_id]) > self.max_history_per_session:
            logger.debug(f"Trimming history for session {session_id}")
            self.sessions[session_id] = self.sessions[session_id][-self.max_history_per_session:]
    
    def get_history(self, session_id, limit=None):
        """
        Get past messages for a given session.
        
        Args:
            session_id: Unique identifier for the chat session
            limit: Maximum number of interactions to return
            
        Returns:
            List of interaction records
        """
        if session_id not in self.sessions:
            logger.debug(f"No history found for session {session_id}")
            return []
        
        history = self.sessions[session_id]
        if limit:
            logger.debug(f"Returning {min(limit, len(history))} history items for session {session_id}")
            return history[-limit:]
        
        logger.debug(f"Returning all {len(history)} history items for session {session_id}")
        return history

# Create global conversation memory
CONVERSATION_MEMORY = ConversationMemory()
logger.info("Global ConversationMemory initialized")