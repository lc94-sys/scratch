"""
Semantic Chunking and OpenSearch Indexing Service

Performs semantic chunking and indexes to OpenSearch.
Based on notebook_02_chunking.py - UPDATED to use OpenSearch instead of FAISS + BM25
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
import boto3
from sklearn.metrics.pairwise import cosine_similarity
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.helpers import bulk
import uuid
from datetime import datetime


class SemanticChunker:
    """
    Performs semantic chunking based on sentence similarity.
    Same logic as notebook_02, no changes needed.
    """
    
    def __init__(
        self,
        embedding_endpoint: str,
        sagemaker_client: boto3.client,
        breakpoint_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        self.endpoint_name = embedding_endpoint
        self.client = sagemaker_client
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Qwen SageMaker endpoint"""
        params = {
            "inputs": [text], 
            "encoding_format": "float"
        }
        body = json.dumps(params)
        
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        return np.array(output_data[0], dtype='float32')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def semantic_chunk(self, text: str) -> List[str]:
        """Perform semantic chunking based on sentence similarity"""
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text] if text.strip() else []
        
        # Get embeddings
        embeddings = [self.get_embedding(s) for s in sentences]
        
        # Calculate similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1), 
                embeddings[i+1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Find breakpoints
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < self.breakpoint_threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))
        
        # Create chunks
        chunks = []
        for i in range(len(breakpoints) - 1):
            chunk_sentences = sentences[breakpoints[i]:breakpoints[i+1]]
            chunk_text = ' '.join(chunk_sentences)
            
            if len(chunk_text) >= self.min_chunk_size:
                if len(chunk_text) > self.max_chunk_size:
                    # Split large chunks
                    while len(chunk_text) > self.max_chunk_size:
                        chunks.append(chunk_text[:self.max_chunk_size])
                        chunk_text = chunk_text[self.max_chunk_size:]
                    if chunk_text:
                        chunks.append(chunk_text)
                else:
                    chunks.append(chunk_text)
        
        return chunks


class OpenSearchIndexer:
    """
    Indexes document chunks to OpenSearch.
    
    REPLACES: HybridIndexer from notebook_02 (which used FAISS + BM25)
    NOW USES: OpenSearch k-NN for vector search
    """
    
    def __init__(
        self,
        opensearch_host: str,
        opensearch_region: str,
        index_name: str,
        embedding_endpoint: str,
        sagemaker_client: boto3.client,
        embedding_dimension: int = 1024
    ):
        self.host = opensearch_host
        self.region = opensearch_region
        self.index_name = index_name
        self.endpoint_name = embedding_endpoint
        self.sagemaker_client = sagemaker_client
        self.dimension = embedding_dimension
        
        # Initialize OpenSearch client
        self.client = self._create_opensearch_client()
        print(f"✓ OpenSearch client initialized for {self.host}")
    
    def _create_opensearch_client(self) -> OpenSearch:
        """Create OpenSearch client with AWS IAM authentication"""
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, self.region, 'es')
        
        return OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Qwen SageMaker endpoint"""
        params = {
            "inputs": [text], 
            "encoding_format": "float"
        }
        body = json.dumps(params)
        
        response = self.sagemaker_client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=body
        )
        output_data = json.loads(response['Body'].read().decode())
        
        return np.array(output_data[0], dtype='float32')
    
    def create_index(self, delete_if_exists: bool = False) -> Dict:
        """
        Create OpenSearch index with k-NN enabled.
        
        REPLACES: Building FAISS IndexFlatIP
        """
        if delete_if_exists and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            print(f"✓ Deleted existing index: {self.index_name}")
        
        if self.client.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists")
            return {'message': f'Index {self.index_name} already exists'}
        
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "number_of_shards": 2,
                    "number_of_replicas": 1
                }
            },
            "mappings": {
                "properties": {
                    # Vector field - REPLACES FAISS index
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",  # Cosine similarity like FAISS IndexFlatIP
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 256,
                                "m": 48
                            }
                        }
                    },
                    # Text field - REPLACES BM25 index (OpenSearch has built-in BM25)
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    # Metadata fields for filtering
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "entitlement": {"type": "keyword"},
                    "org_id": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "summary": {"type": "text"},
                    "created_at": {"type": "date"}
                }
            }
        }
        
        self.client.indices.create(index=self.index_name, body=index_body)
        print(f"✓ Created index: {self.index_name}")
        return {'message': f'Index {self.index_name} created successfully'}
    
    def build_indexes(self, chunks: List[Dict], batch_size: int = 50) -> Dict:
        """
        Index chunks to OpenSearch with embeddings.
        
        REPLACES: 
        - faiss_index.add(embeddings_array)
        - BM25Okapi(tokenized_corpus)
        - Saving faiss.index, embeddings.npy, bm25.pkl, chunk_metadata.json
        
        NOW: Single OpenSearch index handles both vector and text search
        """
        print("Building OpenSearch index...")
        
        # Ensure index exists
        self.create_index(delete_if_exists=False)
        
        total = len(chunks)
        indexed = 0
        failed = 0
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            actions = []
            
            for chunk in batch:
                try:
                    # Get embedding (same as notebook_02)
                    embedding = self.get_embedding(chunk['content'])
                    
                    # Prepare document for OpenSearch
                    action = {
                        '_index': self.index_name,
                        '_source': {
                            'embedding': embedding.tolist(),  # Vector for k-NN search
                            'content': chunk['content'],      # Text for BM25 search
                            'title': chunk.get('title', ''),
                            'doc_id': chunk.get('doc_id', ''),
                            'chunk_id': chunk.get('chunk_id', str(uuid.uuid4())),
                            'chunk_index': chunk.get('chunk_index', 0),
                            'entitlement': chunk.get('entitlement', ['universal']),
                            'org_id': chunk.get('orgId', chunk.get('org_id', '')),
                            'tags': chunk.get('metadata', {}).get('tags', []),
                            'summary': chunk.get('summary', ''),
                            'created_at': datetime.now().isoformat()
                        }
                    }
                    actions.append(action)
                    
                except Exception as e:
                    failed += 1
                    print(f"✗ Error embedding chunk: {e}")
            
            if actions:
                success, errors = bulk(self.client, actions)
                indexed += success
                if errors:
                    failed += len(errors)
            
            print(f"  Indexed {indexed}/{total} chunks")
        
        print(f"✓ OpenSearch index built with {indexed} vectors")
        print(f"✓ BM25 search available (built-in to OpenSearch)")
        print(f"✓ Indexing complete")
        
        return {'indexed': indexed, 'failed': failed, 'total': total}
    
    def delete_index(self) -> Dict:
        """Delete the index"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            return {'message': f'Index {self.index_name} deleted'}
        return {'message': f'Index {self.index_name} does not exist'}
    
    def get_document_count(self) -> int:
        """Get total documents in index"""
        if self.client.indices.exists(index=self.index_name):
            response = self.client.count(index=self.index_name)
            return response['count']
        return 0
    
    def health_check(self) -> Dict:
        """Check OpenSearch connection"""
        try:
            info = self.client.info()
            return {
                'status': 'healthy',
                'cluster_name': info['cluster_name'],
                'version': info['version']['number'],
                'document_count': self.get_document_count()
            }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}


class ChunkingPipeline:
    """
    Complete pipeline: Load processed docs → Chunk → Index to OpenSearch
    
    REPLACES: The sequential flow in notebook_02
    """
    
    def __init__(
        self,
        opensearch_host: str,
        opensearch_region: str,
        index_name: str,
        embedding_endpoint: str,
        sagemaker_client: boto3.client,
        embedding_dimension: int = 1024,
        breakpoint_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        # Initialize chunker (same as notebook_02)
        self.chunker = SemanticChunker(
            embedding_endpoint=embedding_endpoint,
            sagemaker_client=sagemaker_client,
            breakpoint_threshold=breakpoint_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
        
        # Initialize indexer (OpenSearch instead of FAISS+BM25)
        self.indexer = OpenSearchIndexer(
            opensearch_host=opensearch_host,
            opensearch_region=opensearch_region,
            index_name=index_name,
            embedding_endpoint=embedding_endpoint,
            sagemaker_client=sagemaker_client,
            embedding_dimension=embedding_dimension
        )
    
    def process_and_index(
        self,
        processed_documents: List[Dict],
        create_new_index: bool = False
    ) -> Dict:
        """
        Chunk documents and index to OpenSearch.
        
        Same flow as notebook_02:
        1. Load processed documents
        2. Semantic chunk each document
        3. Build indexes (now OpenSearch instead of FAISS+BM25)
        """
        # Create index if needed
        if create_new_index:
            self.indexer.create_index(delete_if_exists=True)
        
        # Chunk all documents (same as notebook_02)
        all_chunks = []
        chunk_id_counter = 0
        
        for doc in processed_documents:
            text_chunks = self.chunker.semantic_chunk(doc['content'])
            
            for idx, chunk_text in enumerate(text_chunks):
                chunk_obj = {
                    'chunk_id': f"chunk_{chunk_id_counter:06d}",
                    'doc_id': doc['doc_id'],
                    'content': chunk_text,
                    'chunk_index': idx,
                    'entitlement': doc['metadata']['entitlement'],
                    'metadata': doc['metadata']['metadata'],
                    'orgId': doc['metadata']['orgId'],
                    'title': doc['metadata']['title'],
                    'summary': doc['metadata']['summary']
                }
                all_chunks.append(chunk_obj)
                chunk_id_counter += 1
            
            print(f"✓ Chunked {doc['doc_id']}: {len(text_chunks)} chunks")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Index to OpenSearch (replaces FAISS+BM25)
        result = self.indexer.build_indexes(all_chunks)
        
        return {
            'documents_processed': len(processed_documents),
            'chunks_created': len(all_chunks),
            'chunks_indexed': result['indexed'],
            'chunks_failed': result['failed']
        }
    
    def health_check(self) -> Dict:
        """Check OpenSearch health"""
        return self.indexer.health_check()
