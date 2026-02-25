"""
Retrieval Service with OpenSearch and LLM

Handles search and answer generation.
Based on notebook_03_indexing.py - UPDATED to use OpenSearch instead of FAISS + BM25
"""

import json
import re
from typing import List, Dict, Optional
import numpy as np
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import uuid
from datetime import datetime


class RetrievalService:
    """
    Retrieval service using OpenSearch for vector search and LLM for answer generation.
    
    REPLACES: HybridRetriever from notebook_03
    
    Flow: Query → OpenSearch (candidates) → LLM (selects best document) → Answer
    """
    
    def __init__(
        self,
        opensearch_host: str,
        opensearch_region: str,
        index_name: str,
        embedding_endpoint: str,
        llm_endpoint: str,
        sagemaker_client: boto3.client,
        llm_max_tokens: int = 512,
        llm_temperature: float = 0.1
    ):
        self.opensearch_host = opensearch_host
        self.opensearch_region = opensearch_region
        self.index_name = index_name
        self.embedding_endpoint = embedding_endpoint
        self.llm_endpoint = llm_endpoint
        self.sagemaker_client = sagemaker_client
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        
        # Initialize OpenSearch client (REPLACES loading FAISS + BM25 indexes)
        self.opensearch_client = self._create_opensearch_client()
        print(f"✓ OpenSearch client initialized")
        
        # Session storage (same as notebook_03)
        self.sessions: Dict[str, Dict] = {}
    
    def _create_opensearch_client(self) -> OpenSearch:
        """Create OpenSearch client with AWS IAM authentication"""
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.opensearch_region, 'es')
        
        return OpenSearch(
            hosts=[{'host': self.opensearch_host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Qwen SageMaker endpoint (same as notebook_03)"""
        params = {
            "inputs": [text], 
            "encoding_format": "float"
        }
        body = json.dumps(params)
        
        response = self.sagemaker_client.invoke_endpoint(
            EndpointName=self.embedding_endpoint,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        return np.array(output_data[0], dtype='float32')
    
    # =========================================================================
    # SEARCH - REPLACES hybrid_search() from notebook_03
    # =========================================================================
    
    def search(
        self,
        query: str,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Vector search with filtering using OpenSearch k-NN.
        
        REPLACES: hybrid_search() which used FAISS + BM25 with manual score combination
        NOW: OpenSearch handles vector search + filtering in one query
        """
        # Get query embedding (same as notebook_03)
        query_embedding = self.get_embedding(query)
        
        # Build filters (REPLACES manual entitlement/org/tag checking in Python loop)
        filters = self._build_filters(entitlement, org_id, tags)
        
        # OpenSearch k-NN query with filters
        # REPLACES: faiss_index.search() + bm25_index.get_scores() + manual score fusion
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding.tolist(),
                                    "k": top_k
                                }
                            }
                        }
                    ],
                    "filter": filters
                }
            }
        }
        
        response = self.opensearch_client.search(
            index=self.index_name,
            body=search_body
        )
        
        # Parse results (REPLACES manual chunk lookup from self.chunks)
        return self._parse_results(response)
    
    def _build_filters(
        self,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None
    ) -> List[Dict]:
        """
        Build filter clauses for OpenSearch query.
        
        REPLACES: The Python loop with manual entitlement/org/tag checking:
            - if 'universal' in chunk_entitlements or entitlement in chunk_entitlements
            - if org_id and chunk['orgId'] != org_id
            - if tags and not any(t in chunk['metadata']['tags'] for t in tags)
        """
        filters = []
        
        # Entitlement filter
        filters.append({
            "bool": {
                "should": [
                    {"term": {"entitlement": entitlement}},
                    {"term": {"entitlement": "universal"}}
                ],
                "minimum_should_match": 1
            }
        })
        
        # Org filter
        if org_id:
            filters.append({"term": {"org_id": org_id}})
        
        # Tags filter
        if tags:
            filters.append({"terms": {"tags": tags}})
        
        return filters
    
    def _parse_results(self, response: Dict) -> List[Dict]:
        """Parse OpenSearch response into same format as notebook_03 chunks"""
        results = []
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            results.append({
                'title': source.get('title', ''),
                'content': source.get('content', ''),
                'doc_id': source.get('doc_id', ''),
                'chunk_id': source.get('chunk_id', ''),
                'chunk_index': source.get('chunk_index', 0),
                'entitlement': source.get('entitlement', []),
                'orgId': source.get('org_id', ''),
                'metadata': {'tags': source.get('tags', [])},
                'summary': source.get('summary', ''),
                'score': hit['_score']
            })
        
        return results
    
    # =========================================================================
    # LLM ANSWER GENERATION - UPDATED to select ONE document
    # =========================================================================
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Generate answer using LLM.
        
        UPDATED: LLM now selects the ONE best document instead of using all.
        This fixes the issue where embedding scores were not accurate.
        """
        # Build numbered document context
        context = "\n\n".join([
            f"[Document {i+1}]: {chunk['title']}\n{chunk['content']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Build conversation history (same as notebook_03)
        history_context = ""
        if conversation_history:
            history_context = "Previous conversation:\n"
            for turn in conversation_history:
                history_context += f"User: {turn['query']}\n"
                history_context += f"Assistant: {turn['answer']}\n\n"
        
        # UPDATED prompt - asks LLM to select ONE document
        prompt = f"""<begin_of_text>You are a document selector and information extractor.

{history_context}

DOCUMENTS:
{context}

---

Question: {query}

INSTRUCTIONS:
1. Read all documents above
2. Select the ONE document that best answers the question
3. Provide your answer using ONLY information from that document
4. End your response with exactly: SELECTED_DOCUMENT: [number]

If no document answers the question, respond with:
"The requested information is not available in the provided documents."
SELECTED_DOCUMENT: 0

Answer:"""

        # Call LLM (same as notebook_03)
        params = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.llm_max_tokens,
                "temperature": self.llm_temperature
            }
        }
        body = json.dumps(params)
        
        response = self.sagemaker_client.invoke_endpoint(
            EndpointName=self.llm_endpoint,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        # Extract answer
        raw_answer = output_data[0]['generated_text'] if isinstance(output_data, list) else output_data['generated_text']
        
        # Parse response to get selected document
        answer_text, selected_doc_num = self._parse_llm_response(raw_answer)
        
        # Get source document (LLM selected, not embedding score)
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
            'answer': answer_text,
            'sources': sources,
            'source_document': source_document,
            'candidates_provided': len(context_chunks)
        }
    
    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract answer and selected document number"""
        answer_text = response.strip()
        selected_doc_num = 0
        
        patterns = [
            r'SELECTED_DOCUMENT\s*:\s*\[?(\d+)\]?',
            r'SELECTED DOCUMENT\s*:\s*\[?(\d+)\]?',
            r'Document\s*:\s*\[?(\d+)\]?$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                selected_doc_num = int(match.group(1))
                answer_text = re.sub(pattern, '', response, flags=re.IGNORECASE).strip()
                break
        
        return answer_text, selected_doc_num
    
    # =========================================================================
    # QUERY PIPELINE (same as notebook_03)
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
        """
        Complete query pipeline: Search → LLM → Answer
        Same flow as notebook_03, but using OpenSearch
        """
        # Get candidates from OpenSearch (REPLACES hybrid_search with FAISS+BM25)
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
    # SESSION MANAGEMENT (same as notebook_03)
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
        """Query with session context and conversation history (same as notebook_03)"""
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
        try:
            info = self.opensearch_client.info()
            count = self.opensearch_client.count(index=self.index_name)
            return {
                'status': 'healthy',
                'cluster_name': info['cluster_name'],
                'document_count': count['count']
            }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
