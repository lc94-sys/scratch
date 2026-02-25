"""
Semantic Chunking and Indexing Service

Uses EmbeddingModel from models.py and OpenSearchClient from opensearch_client.py
"""

import re
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from services.models import EmbeddingModel
from services.opensearch_client import OpenSearchClient


class SemanticChunker:
    """
    Performs semantic chunking based on sentence similarity.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        breakpoint_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        self.embedding_model = embedding_model
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def semantic_chunk(self, text: str) -> List[str]:
        """Perform semantic chunking based on sentence similarity"""
        sentences = self.split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text] if text.strip() else []
        
        # Get embeddings using EmbeddingModel
        embeddings = [self.embedding_model.get_embedding(s) for s in sentences]
        
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
                    while len(chunk_text) > self.max_chunk_size:
                        chunks.append(chunk_text[:self.max_chunk_size])
                        chunk_text = chunk_text[self.max_chunk_size:]
                    if chunk_text:
                        chunks.append(chunk_text)
                else:
                    chunks.append(chunk_text)
        
        return chunks


class ChunkingPipeline:
    """
    Complete pipeline: Chunk documents → Index to OpenSearch
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        opensearch_client: OpenSearchClient,
        breakpoint_threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        self.embedding_model = embedding_model
        self.opensearch_client = opensearch_client
        
        # Initialize chunker
        self.chunker = SemanticChunker(
            embedding_model=embedding_model,
            breakpoint_threshold=breakpoint_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size
        )
    
    def process_and_index(
        self,
        processed_documents: List[Dict],
        create_new_index: bool = False
    ) -> Dict:
        """Chunk documents and index to OpenSearch"""
        
        # Create index if needed
        if create_new_index:
            self.opensearch_client.create_index(delete_if_exists=True)
        else:
            self.opensearch_client.create_index(delete_if_exists=False)
        
        # Chunk all documents
        all_chunks = []
        chunk_id_counter = 0
        
        for doc in processed_documents:
            text_chunks = self.chunker.semantic_chunk(doc['content'])
            
            for idx, chunk_text in enumerate(text_chunks):
                # Get embedding
                embedding = self.embedding_model.get_embedding(chunk_text)
                
                chunk_obj = {
                    'chunk_id': f"chunk_{chunk_id_counter:06d}",
                    'doc_id': doc['doc_id'],
                    'content': chunk_text,
                    'embedding': embedding,
                    'chunk_index': idx,
                    'entitlement': doc['metadata']['entitlement'],
                    'tags': doc['metadata']['metadata'].get('tags', []),
                    'org_id': doc['metadata']['orgId'],
                    'title': doc['metadata']['title'],
                    'summary': doc['metadata']['summary']
                }
                all_chunks.append(chunk_obj)
                chunk_id_counter += 1
            
            print(f"✓ Chunked {doc['doc_id']}: {len(text_chunks)} chunks")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Index to OpenSearch
        result = self.opensearch_client.bulk_index(all_chunks)
        
        return {
            'documents_processed': len(processed_documents),
            'chunks_created': len(all_chunks),
            'chunks_indexed': result['indexed'],
            'chunks_failed': result['failed']
        }
    
    def health_check(self) -> Dict:
        """Check OpenSearch health"""
        return self.opensearch_client.health_check()
