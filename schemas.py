"""
Pydantic Models for RAG API

Request and response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


# =============================================================================
# DOCUMENT PROCESSING
# =============================================================================

class DocumentMetadataInput(BaseModel):
    """Metadata for a document to process"""
    file: str = Field(..., description="Filename in raw_docs folder")
    entitlement: List[str] = Field(..., description="List of entitlements (e.g., ['agent_support', 'universal'])")
    metadata: Dict = Field(default={}, description="Additional metadata like tags")
    orgId: str = Field(..., description="Organization ID")
    title: str = Field(..., description="Document title")
    summary: str = Field(default="", description="Document summary")


class ProcessDocumentsRequest(BaseModel):
    """Request to process documents"""
    documents: List[DocumentMetadataInput]


class ProcessDocumentsResponse(BaseModel):
    """Response from document processing"""
    success: bool
    documents_processed: int
    message: str


# =============================================================================
# CHUNKING & INDEXING
# =============================================================================

class ChunkAndIndexRequest(BaseModel):
    """Request to chunk and index documents"""
    documents: Optional[List[DocumentMetadataInput]] = Field(
        None, 
        description="Documents to process. If empty, loads already processed docs."
    )
    create_new_index: bool = Field(
        default=False, 
        description="Delete and recreate index if True"
    )


class ChunkAndIndexResponse(BaseModel):
    """Response from chunk and index operation"""
    success: bool
    message: str
    documents_processed: int
    chunks_created: int
    chunks_indexed: int


class IndexCreateResponse(BaseModel):
    """Response from index creation"""
    success: bool
    message: str


class IndexStatsResponse(BaseModel):
    """Index statistics"""
    index_name: str
    document_count: int
    status: str


# =============================================================================
# SEARCH
# =============================================================================

class SearchRequest(BaseModel):
    """Search request"""
    query: str = Field(..., description="Search query")
    entitlement: str = Field(..., description="User's entitlement level")
    org_id: Optional[str] = Field(None, description="Organization ID filter")
    tags: Optional[List[str]] = Field(None, description="Tag filters")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")


class SearchResult(BaseModel):
    """Single search result"""
    title: str
    content: str
    doc_id: str
    chunk_id: Optional[str] = ""
    chunk_index: int
    entitlement: List[str]
    orgId: str = ""
    tags: List[str] = []
    summary: Optional[str] = ""
    score: float


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[SearchResult]
    count: int


# =============================================================================
# QUERY (SEARCH + LLM)
# =============================================================================

class ConversationTurn(BaseModel):
    """Single conversation turn"""
    query: str
    answer: str


class QueryRequest(BaseModel):
    """Full query request (search + LLM)"""
    query: str = Field(..., description="User query")
    entitlement: str = Field(..., description="User's entitlement level")
    org_id: Optional[str] = Field(None, description="Organization ID filter")
    tags: Optional[List[str]] = Field(None, description="Tag filters")
    candidates: int = Field(default=10, ge=1, le=50, description="Number of candidates for LLM")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        None, 
        description="Previous conversation turns"
    )


class SourceDocument(BaseModel):
    """Source document selected by LLM"""
    title: str
    doc_id: str
    score: float


class QueryResponse(BaseModel):
    """Full query response"""
    answer: str
    source_document: Optional[SourceDocument] = None
    sources: List[SourceDocument] = []
    candidates_provided: int


# =============================================================================
# SESSIONS
# =============================================================================

class SessionCreateRequest(BaseModel):
    """Create session request"""
    user_id: str
    entitlement: str
    org_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    """Session creation response"""
    session_id: str
    user_id: str
    entitlement: str
    org_id: Optional[str] = None


class SessionQueryRequest(BaseModel):
    """Query with session"""
    query: str
    tags: Optional[List[str]] = None
    candidates: int = Field(default=10, ge=1, le=50)
    history_limit: int = Field(default=3, ge=1, le=10)


class SessionHistoryResponse(BaseModel):
    """Session history response"""
    session_id: str
    history: List[Dict]


# =============================================================================
# HEALTH
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    cluster_name: Optional[str] = None
    document_count: Optional[int] = None
    version: Optional[str] = None
    error: Optional[str] = None
