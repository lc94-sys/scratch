"""
FastAPI Router for RAG API

Uses refactored services with centralized models.py and opensearch_client.py
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from models.schemas import (
    DocumentMetadataInput,
    ProcessDocumentsRequest,
    ProcessDocumentsResponse,
    IndexCreateResponse,
    IndexStatsResponse,
    ChunkAndIndexRequest,
    ChunkAndIndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionQueryRequest,
    HealthResponse
)

from services.models import ModelFactory
from services.opensearch_client import OpenSearchClient
from services.document_processor import DocumentProcessor
from services.chunking_service import ChunkingPipeline
from services.retrieval_service import RetrievalService
from config import settings


router = APIRouter(prefix="/api/v1", tags=["RAG"])


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_model_factory: ModelFactory = None
_opensearch_client: OpenSearchClient = None
_retrieval_service: RetrievalService = None


def get_model_factory() -> ModelFactory:
    """Get or create ModelFactory"""
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory(region=settings.aws_region)
    return _model_factory


def get_opensearch_client() -> OpenSearchClient:
    """Get or create OpenSearchClient"""
    global _opensearch_client
    if _opensearch_client is None:
        _opensearch_client = OpenSearchClient(
            host=settings.opensearch_host,
            region=settings.opensearch_region,
            index_name=settings.opensearch_index,
            embedding_dimension=settings.embedding_dimension
        )
    return _opensearch_client


def get_document_processor() -> DocumentProcessor:
    """Get document processor"""
    return DocumentProcessor(
        raw_docs_path=settings.raw_docs_path,
        processed_docs_path=settings.processed_docs_path
    )


def get_chunking_pipeline() -> ChunkingPipeline:
    """Get chunking pipeline"""
    factory = get_model_factory()
    return ChunkingPipeline(
        embedding_model=factory.create_embedding_model(settings.embedding_endpoint),
        opensearch_client=get_opensearch_client(),
        breakpoint_threshold=settings.chunking_breakpoint_threshold,
        min_chunk_size=settings.chunking_min_size,
        max_chunk_size=settings.chunking_max_size
    )


def get_retrieval_service() -> RetrievalService:
    """Get or create retrieval service (singleton for session persistence)"""
    global _retrieval_service
    if _retrieval_service is None:
        factory = get_model_factory()
        _retrieval_service = RetrievalService(
            embedding_model=factory.create_embedding_model(settings.embedding_endpoint),
            llm_model=factory.create_llm_model(
                endpoint_name=settings.llm_endpoint,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature
            ),
            opensearch_client=get_opensearch_client()
        )
    return _retrieval_service


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health"""
    try:
        result = get_opensearch_client().health_check()
        return HealthResponse(**result)
    except Exception as e:
        return HealthResponse(status='unhealthy', error=str(e))


# =============================================================================
# INDEX MANAGEMENT
# =============================================================================

@router.post("/index/create", response_model=IndexCreateResponse)
async def create_index(delete_if_exists: bool = False):
    """Create OpenSearch index"""
    try:
        result = get_opensearch_client().create_index(delete_if_exists=delete_if_exists)
        return IndexCreateResponse(success=True, message=result.get('message', 'Index created'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index", response_model=IndexCreateResponse)
async def delete_index():
    """Delete OpenSearch index"""
    try:
        result = get_opensearch_client().delete_index()
        return IndexCreateResponse(success=True, message=result.get('message', 'Index deleted'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """Get index statistics"""
    try:
        client = get_opensearch_client()
        health = client.health_check()
        return IndexStatsResponse(
            index_name=client.index_name,
            document_count=health.get('document_count', 0),
            status=health.get('status', 'unknown')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DOCUMENT PROCESSING & INDEXING
# =============================================================================

@router.post("/documents/process", response_model=ProcessDocumentsResponse)
async def process_documents(request: ProcessDocumentsRequest):
    """Process DOCX documents"""
    try:
        processor = get_document_processor()
        documents_metadata = [doc.model_dump() for doc in request.documents]
        processed = processor.process_all(documents_metadata)
        return ProcessDocumentsResponse(
            success=True,
            documents_processed=len(processed),
            message=f"Processed {len(processed)} documents"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/index", response_model=ChunkAndIndexResponse)
async def chunk_and_index_documents(request: ChunkAndIndexRequest):
    """Process, chunk, and index documents to OpenSearch"""
    try:
        processor = get_document_processor()
        pipeline = get_chunking_pipeline()
        
        if request.documents:
            documents_metadata = [doc.model_dump() for doc in request.documents]
            processed_docs = processor.process_all(documents_metadata)
        else:
            processed_docs = processor.load_processed_documents()
        
        if not processed_docs:
            return ChunkAndIndexResponse(
                success=False,
                message="No documents to process",
                documents_processed=0,
                chunks_created=0,
                chunks_indexed=0
            )
        
        result = pipeline.process_and_index(
            processed_documents=processed_docs,
            create_new_index=request.create_new_index
        )
        
        return ChunkAndIndexResponse(
            success=True,
            message="Documents processed and indexed successfully",
            documents_processed=result['documents_processed'],
            chunks_created=result['chunks_created'],
            chunks_indexed=result['chunks_indexed']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SEARCH
# =============================================================================

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for documents (vector search only, no LLM)"""
    try:
        retrieval = get_retrieval_service()
        results = retrieval.search(
            query=request.query,
            entitlement=request.entitlement,
            org_id=request.org_id,
            tags=request.tags,
            top_k=request.top_k
        )
        
        return SearchResponse(
            query=request.query,
            results=[SearchResult(
                title=r['title'],
                content=r['content'],
                doc_id=r['doc_id'],
                chunk_id=r.get('chunk_id', ''),
                chunk_index=r['chunk_index'],
                entitlement=r['entitlement'],
                orgId=r.get('org_id', ''),
                tags=r.get('tags', []),
                summary=r.get('summary', ''),
                score=r['score']
            ) for r in results],
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# QUERY (Search + LLM)
# =============================================================================

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Full query pipeline: Search → LLM → Answer"""
    try:
        retrieval = get_retrieval_service()
        
        history = None
        if request.conversation_history:
            history = [turn.model_dump() for turn in request.conversation_history]
        
        result = retrieval.query(
            query=request.query,
            entitlement=request.entitlement,
            org_id=request.org_id,
            tags=request.tags,
            top_k=request.candidates,
            conversation_history=history
        )
        
        source_doc = None
        if result.get('source_document'):
            source_doc = SourceDocument(**result['source_document'])
        
        sources = [SourceDocument(**s) for s in result.get('sources', [])]
        
        return QueryResponse(
            answer=result['answer'],
            source_document=source_doc,
            sources=sources,
            candidates_provided=result['candidates_provided']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SESSIONS
# =============================================================================

@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    """Create a conversation session"""
    try:
        retrieval = get_retrieval_service()
        session_id = retrieval.create_session(
            user_id=request.user_id,
            entitlement=request.entitlement,
            org_id=request.org_id
        )
        return SessionCreateResponse(
            session_id=session_id,
            user_id=request.user_id,
            entitlement=request.entitlement,
            org_id=request.org_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def query_with_session(session_id: str, request: SessionQueryRequest):
    """Query using session for conversation history"""
    try:
        retrieval = get_retrieval_service()
        result = retrieval.query_with_session(
            session_id=session_id,
            query=request.query,
            tags=request.tags,
            top_k=request.candidates,
            history_limit=request.history_limit
        )
        
        source_doc = None
        if result.get('source_document'):
            source_doc = SourceDocument(**result['source_document'])
        
        sources = [SourceDocument(**s) for s in result.get('sources', [])]
        
        return QueryResponse(
            answer=result['answer'],
            source_document=source_doc,
            sources=sources,
            candidates_provided=result['candidates_provided']
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: Optional[int] = None):
    """Get conversation history"""
    try:
        retrieval = get_retrieval_service()
        session = retrieval.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        history = retrieval.get_conversation_history(session_id, limit=limit)
        return {"session_id": session_id, "history": history}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        retrieval = get_retrieval_service()
        if retrieval.clear_session(session_id):
            return {"message": "Session deleted"}
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
