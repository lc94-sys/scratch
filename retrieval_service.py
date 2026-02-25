"""
Retrieval Service

Uses EmbeddingModel and LLMModel from models.py
Uses OpenSearchClient from opensearch_client.py
"""

from typing import List, Dict, Optional
import uuid
from datetime import datetime

from services.models import EmbeddingModel, LLMModel
from services.opensearch_client import OpenSearchClient


class RetrievalService:
    """
    Retrieval service using OpenSearch for search and LLM for answer generation.
    
    Flow: Query → OpenSearch (candidates) → LLM (selects best document) → Answer
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_model: LLMModel,
        opensearch_client: OpenSearchClient
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.opensearch_client = opensearch_client
        
        # Session storage
        self.sessions: Dict[str, Dict] = {}
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(
        self,
        query: str,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Vector search with filtering using OpenSearch"""
        
        # Get query embedding using EmbeddingModel
        query_embedding = self.embedding_model.get_embedding(query)
        
        # Search using OpenSearchClient
        return self.opensearch_client.vector_search(
            query_embedding=query_embedding,
            entitlement=entitlement,
            org_id=org_id,
            tags=tags,
            top_k=top_k
        )
    
    # =========================================================================
    # ANSWER GENERATION
    # =========================================================================
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """Generate answer using LLM with document selection"""
        
        # Use LLMModel for answer generation
        result = self.llm_model.generate_answer_with_document_selection(
            query=query,
            context_chunks=context_chunks,
            conversation_history=conversation_history
        )
        
        # Get source document
        selected_doc_num = result['selected_document_number']
        source_document = None
        sources = []
        
        if 1 <= selected_doc_num <= len(context_chunks):
            chunk = context_chunks[selected_doc_num - 1]
            source_document = {
                'title': chunk['title'],
                'doc_id': chunk['doc_id'],
                'score': chunk['score']
            }
            sources = [source_document]
        
        return {
            'answer': result['answer'],
            'sources': sources,
            'source_document': source_document,
            'candidates_provided': len(context_chunks)
        }
    
    # =========================================================================
    # QUERY PIPELINE
    # =========================================================================
    
    def query(
        self,
        query: str,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None,
        top_k: int = 10,
        conversation_history: List[Dict] = None
    ) -> Dict:
        """Complete query pipeline: Search → LLM → Answer"""
        
        # Get candidates from OpenSearch
        chunks = self.search(
            query=query,
            entitlement=entitlement,
            org_id=org_id,
            tags=tags,
            top_k=top_k
        )
        
        if not chunks:
            return {
                'answer': 'No relevant information found.',
                'sources': [],
                'source_document': None,
                'candidates_provided': 0
            }
        
        # Generate answer with LLM
        return self.generate_answer(query, chunks, conversation_history)
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def create_session(
        self,
        user_id: str,
        entitlement: str,
        org_id: str = None
    ) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'entitlement': entitlement,
            'org_id': org_id,
            'created_at': datetime.now().isoformat(),
            'conversation_history': [],
            'last_activity': datetime.now().isoformat()
        }
        print(f"✓ Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = None
    ) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        history = session['conversation_history']
        return history[-limit:] if limit else history
    
    def query_with_session(
        self,
        session_id: str,
        query: str,
        tags: List[str] = None,
        top_k: int = 10,
        history_limit: int = 3
    ) -> Dict:
        """Query with session context and conversation history"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session['last_activity'] = datetime.now().isoformat()
        history = self.get_conversation_history(session_id, limit=history_limit)
        
        # Query with history
        result = self.query(
            query=query,
            entitlement=session['entitlement'],
            org_id=session['org_id'],
            tags=tags,
            top_k=top_k,
            conversation_history=history
        )
        
        # Store in history
        session['conversation_history'].append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': result['answer'],
            'sources': result['sources']
        })
        
        result['session_id'] = session_id
        return result
    
    def clear_session(self, session_id: str) -> bool:
        """Clear/end a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def export_session(self, session_id: str) -> Optional[Dict]:
        """Export session data"""
        return self.get_session(session_id)
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def health_check(self) -> Dict:
        """Check OpenSearch connection"""
        return self.opensearch_client.health_check()
