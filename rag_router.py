"""
FastAPI Router for RAG API

All endpoints for document processing, indexing, and retrieval.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import boto3

from models.schemas import (
    # Document Processing
    DocumentMetadataInput,
    ProcessDocumentsRequest,
    ProcessDocumentsResponse,
    
    # Indexing
    IndexCreateResponse,
    IndexStatsResponse,
    ChunkAndIndexRequest,
    ChunkAndIndexResponse,
    
    # Search & Query
    SearchRequest,
    SearchResponse,
    SearchResult,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    
    # Sessions
    SessionCreateRequest,
    SessionCreateResponse,
    SessionQueryRequest,
    
    # Health
    HealthResponse
)

from services.document_processor import DocumentProcessor
from services.chunking_service import ChunkingPipeline
from services.retrieval_service import RetrievalService
from config import settings


router = APIRouter(prefix="/api/v1", tags=["RAG"])


# =============================================================================
# SINGLETON INSTANCES (to persist sessions)
# =============================================================================

_sagemaker_client: boto3.client = None
_retrieval_service: RetrievalService = None


def get_sagemaker_client() -> boto3.client:
    """Get or create SageMaker runtime client"""
    global _sagemaker_client
    if _sagemaker_client is None:
        _sagemaker_client = boto3.client('sagemaker-runtime', region_name=settings.aws_region)
    return _sagemaker_client


def get_document_processor() -> DocumentProcessor:
    """Get document processor instance"""
    return DocumentProcessor(
        raw_docs_path=settings.raw_docs_path,
        processed_docs_path=settings.processed_docs_path
    )


def get_chunking_pipeline() -> ChunkingPipeline:
    """Get chunking pipeline instance"""
    return ChunkingPipeline(
        opensearch_host=settings.opensearch_host,
        opensearch_region=settings.opensearch_region,
        index_name=settings.opensearch_index,
        embedding_endpoint=settings.embedding_endpoint,
        sagemaker_client=get_sagemaker_client(),
        embedding_dimension=settings.embedding_dimension,
        breakpoint_threshold=settings.chunking_breakpoint_threshold,
        min_chunk_size=settings.chunking_min_size,
        max_chunk_size=settings.chunking_max_size
    )


def get_retrieval_service() -> RetrievalService:
    """Get or create retrieval service (singleton for session persistence)"""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService(
            opensearch_host=settings.opensearch_host,
            opensearch_region=settings.opensearch_region,
            index_name=settings.opensearch_index,
            embedding_endpoint=settings.embedding_endpoint,
            llm_endpoint=settings.llm_endpoint,
            sagemaker_client=get_sagemaker_client(),
            llm_max_tokens=settings.llm_max_tokens,
            llm_temperature=settings.llm_temperature
        )
    return _retrieval_service


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health including OpenSearch connection"""
    try:
        retrieval = get_retrieval_service()
        result = retrieval.health_check()
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
        pipeline = get_chunking_pipeline()
        result = pipeline.indexer.create_index(delete_if_exists=delete_if_exists)
        return IndexCreateResponse(
            success=True,
            message=result.get('message', 'Index created')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index", response_model=IndexCreateResponse)
async def delete_index():
    """Delete OpenSearch index"""
    try:
        pipeline = get_chunking_pipeline()
        result = pipeline.indexer.delete_index()
        return IndexCreateResponse(
            success=True,
            message=result.get('message', 'Index deleted')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/stats", response_model=IndexStatsResponse)
async def get_index_stats():
    """Get index statistics"""
    try:
        pipeline = get_chunking_pipeline()
        health = pipeline.health_check()
        return IndexStatsResponse(
            index_name=settings.opensearch_index,
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
    """Process DOCX documents and extract text/tables"""
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
    """
    Process documents, chunk them, and index to OpenSearch.
    
    This is the main endpoint for ingesting new documents.
    """
    try:
        processor = get_document_processor()
        pipeline = get_chunking_pipeline()
        
        # Process documents if metadata provided
        if request.documents:
            documents_metadata = [doc.model_dump() for doc in request.documents]
            processed_docs = processor.process_all(documents_metadata)
        else:
            # Load already processed documents
            processed_docs = processor.load_processed_documents()
        
        if not processed_docs:
            return ChunkAndIndexResponse(
                success=False,
                message="No documents to process",
                documents_processed=0,
                chunks_created=0,
                chunks_indexed=0
            )
        
        # Chunk and index
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
# SEARCH (Vector search only, no LLM)
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
                orgId=r.get('orgId', ''),
                tags=r.get('metadata', {}).get('tags', []),
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
        
        # Convert conversation history if provided
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
        
        sources = []
        if result.get('sources'):
            sources = [SourceDocument(**s) for s in result['sources']]
        
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
        
        sources = []
        if result.get('sources'):
            sources = [SourceDocument(**s) for s in result['sources']]
        
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
    """Get conversation history for a session"""
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
