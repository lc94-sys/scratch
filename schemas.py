"""
Pydantic Models for RAG API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# Document Processing
class DocumentMetadataInput(BaseModel):
    file: str
    entitlement: List[str]
    metadata: Dict = {}
    orgId: str
    title: str
    summary: str = ""


class ProcessDocumentsRequest(BaseModel):
    documents: List[DocumentMetadataInput]


class ProcessDocumentsResponse(BaseModel):
    success: bool
    documents_processed: int
    message: str


# Indexing
class ChunkAndIndexRequest(BaseModel):
    documents: Optional[List[DocumentMetadataInput]] = None
    create_new_index: bool = False


class ChunkAndIndexResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    chunks_created: int
    chunks_indexed: int


class IndexCreateResponse(BaseModel):
    success: bool
    message: str


class IndexStatsResponse(BaseModel):
    index_name: str
    document_count: int
    status: str


# Search
class SearchRequest(BaseModel):
    query: str
    entitlement: str
    org_id: Optional[str] = None
    tags: Optional[List[str]] = None
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    title: str
    content: str
    doc_id: str
    chunk_id: str = ""
    chunk_index: int
    entitlement: List[str]
    orgId: str = ""
    tags: List[str] = []
    summary: str = ""
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    count: int


# Query
class ConversationTurn(BaseModel):
    query: str
    answer: str


class QueryRequest(BaseModel):
    query: str
    entitlement: str
    org_id: Optional[str] = None
    tags: Optional[List[str]] = None
    candidates: int = Field(default=10, ge=1, le=50)
    conversation_history: Optional[List[ConversationTurn]] = None


class SourceDocument(BaseModel):
    title: str
    doc_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    source_document: Optional[SourceDocument] = None
    sources: List[SourceDocument] = []
    candidates_provided: int


# Sessions
class SessionCreateRequest(BaseModel):
    user_id: str
    entitlement: str
    org_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    user_id: str
    entitlement: str
    org_id: Optional[str] = None


class SessionQueryRequest(BaseModel):
    query: str
    tags: Optional[List[str]] = None
    candidates: int = Field(default=10, ge=1, le=50)
    history_limit: int = Field(default=3, ge=1, le=10)


# Health
class HealthResponse(BaseModel):
    status: str
    cluster_name: Optional[str] = None
    document_count: Optional[int] = None
    version: Optional[str] = None
    error: Optional[str] = None
