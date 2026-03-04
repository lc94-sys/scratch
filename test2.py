"""
OpenSearch Client

All OpenSearch connection, index management, and search operations.
"""

from typing import List, Dict
import numpy as np
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.helpers import bulk
import uuid
from datetime import datetime


class OpenSearchClient:
    """
    OpenSearch client with all index and search operations.
    """
    
    def __init__(
        self,
        host: str,
        region: str = 'us-east-1',
        index_name: str = 'documents',
        embedding_dimension: int = 1024
    ):
        self.host = host
        self.region = region
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        
        # Initialize client
        self.client = self._create_client()
    
    def _create_client(self) -> OpenSearch:
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
    
    # =========================================================================
    # HEALTH & INFO
    # =========================================================================
    
    def health_check(self) -> Dict:
        """Check OpenSearch connection health"""
        try:
            # Use info() endpoint
            info = self.client.info()
            count = self.get_document_count()
            return {
                'status': 'healthy',
                'cluster_name': info.get('cluster_name', 'unknown'),
                'version': info.get('version', {}).get('number', 'unknown'),
                'document_count': count
            }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def get_document_count(self) -> int:
        """Get total documents in index"""
        try:
            if self.client.indices.exists(index=self.index_name):
                response = self.client.count(index=self.index_name)
                return response['count']
            return 0
        except Exception:
            return 0
    
    def get_index_info(self) -> Dict:
        """Get detailed index information"""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {'exists': False, 'message': f'Index {self.index_name} does not exist'}
            
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            index_stats = stats['indices'].get(self.index_name, {})
            
            # Get mapping
            mapping = self.client.indices.get_mapping(index=self.index_name)
            
            # Get settings
            settings = self.client.indices.get_settings(index=self.index_name)
            
            # Extract embedding info
            properties = mapping.get(self.index_name, {}).get('mappings', {}).get('properties', {})
            embedding_info = properties.get('embedding', {})
            
            return {
                'exists': True,
                'index_name': self.index_name,
                'document_count': index_stats.get('primaries', {}).get('docs', {}).get('count', 0),
                'size_in_bytes': index_stats.get('primaries', {}).get('store', {}).get('size_in_bytes', 0),
                'embedding_dimension': embedding_info.get('dimension', 'unknown'),
                'embedding_type': embedding_info.get('type', 'unknown'),
                'knn_enabled': settings.get(self.index_name, {}).get('settings', {}).get('index', {}).get('knn', False)
            }
        except Exception as e:
            return {'exists': False, 'error': str(e)}
    
    def get_sample_document(self) -> Dict:
        """Get a sample document to verify structure"""
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": 1,
                    "_source": ["title", "content", "doc_id", "chunk_id", "entitlement", "tags"]
                }
            )
            hits = response.get('hits', {}).get('hits', [])
            if hits:
                return {
                    'found': True,
                    'document': hits[0]['_source']
                }
            return {'found': False, 'message': 'No documents in index'}
        except Exception as e:
            return {'found': False, 'error': str(e)}
    
    def verify_embeddings(self) -> Dict:
        """Verify that embeddings are stored correctly"""
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "size": 1,
                    "_source": ["embedding"]
                }
            )
            hits = response.get('hits', {}).get('hits', [])
            if hits:
                embedding = hits[0]['_source'].get('embedding', [])
                return {
                    'has_embeddings': True,
                    'embedding_dimension': len(embedding),
                    'sample_values': embedding[:5] if embedding else []
                }
            return {'has_embeddings': False, 'message': 'No documents found'}
        except Exception as e:
            return {'has_embeddings': False, 'error': str(e)}
    
    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================
    
    def create_index(self, delete_if_exists: bool = False) -> Dict:
        """Create OpenSearch index with k-NN enabled"""
        
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
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 256,
                                "m": 48
                            }
                        }
                    },
                    "content": {"type": "text", "analyzer": "standard"},
                    "title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
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
    
    def delete_index(self) -> Dict:
        """Delete the index"""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            print(f"✓ Deleted index: {self.index_name}")
            return {'message': f'Index {self.index_name} deleted'}
        return {'message': f'Index {self.index_name} does not exist'}
    
    # =========================================================================
    # INDEXING DOCUMENTS
    # =========================================================================
    
    def index_document(
        self,
        content: str,
        embedding: np.ndarray,
        title: str,
        doc_id: str,
        chunk_id: str = None,
        chunk_index: int = 0,
        entitlement: List[str] = None,
        org_id: str = '',
        tags: List[str] = None,
        summary: str = ''
    ) -> str:
        """Index a single document with embedding"""
        
        doc_body = {
            'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            'content': content,
            'title': title,
            'doc_id': doc_id,
            'chunk_id': chunk_id or str(uuid.uuid4()),
            'chunk_index': chunk_index,
            'entitlement': entitlement or ['universal'],
            'org_id': org_id,
            'tags': tags or [],
            'summary': summary,
            'created_at': datetime.now().isoformat()
        }
        
        response = self.client.index(
            index=self.index_name,
            body=doc_body,
            refresh=True
        )
        
        return response['_id']
    
    def bulk_index(self, documents: List[Dict], batch_size: int = 50) -> Dict:
        """Bulk index documents"""
        total = len(documents)
        indexed = 0
        failed = 0
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            actions = []
            
            for doc in batch:
                try:
                    embedding = doc['embedding']
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()
                    
                    action = {
                        '_index': self.index_name,
                        '_source': {
                            'embedding': embedding,
                            'content': doc['content'],
                            'title': doc.get('title', ''),
                            'doc_id': doc.get('doc_id', ''),
                            'chunk_id': doc.get('chunk_id', str(uuid.uuid4())),
                            'chunk_index': doc.get('chunk_index', 0),
                            'entitlement': doc.get('entitlement', ['universal']),
                            'org_id': doc.get('org_id', ''),
                            'tags': doc.get('tags', []),
                            'summary': doc.get('summary', ''),
                            'created_at': datetime.now().isoformat()
                        }
                    }
                    actions.append(action)
                except Exception as e:
                    failed += 1
                    print(f"✗ Error preparing document: {e}")
            
            if actions:
                success, errors = bulk(self.client, actions)
                indexed += success
                if errors:
                    failed += len(errors)
            
            print(f"  Indexed {indexed}/{total} documents")
        
        print(f"✓ Bulk indexing complete: {indexed} indexed, {failed} failed")
        return {'indexed': indexed, 'failed': failed, 'total': total}
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def vector_search(
        self,
        query_embedding: np.ndarray,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Vector similarity search only (k-NN)"""
        
        filters = self._build_filters(entitlement, org_id, tags)
        
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                                    "k": top_k
                                }
                            }
                        }
                    ],
                    "filter": filters
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        return self._parse_search_results(response)
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid search combining vector (k-NN) + text (BM25).
        
        This replaces the old FAISS + BM25 approach with OpenSearch's built-in capabilities.
        
        Args:
            query_text: Original query text for BM25 search
            query_embedding: Query embedding for vector search
            entitlement: User entitlement for filtering
            org_id: Optional org filter
            tags: Optional tags filter
            top_k: Number of results
            vector_weight: Weight for vector score (default 0.7)
            text_weight: Weight for BM25 text score (default 0.3)
        """
        filters = self._build_filters(entitlement, org_id, tags)
        
        # Hybrid query combining k-NN and BM25
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # Vector search (k-NN) - replaces FAISS
                        {
                            "script_score": {
                                "query": {"bool": {"filter": filters}},
                                "script": {
                                    "source": f"knn_score + {vector_weight}",
                                    "params": {
                                        "field": "embedding",
                                        "query_value": query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                                        "space_type": "cosinesimil"
                                    }
                                }
                            }
                        },
                        # Text search (BM25) - replaces BM25Okapi
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["content^1", "title^2"],
                                "type": "best_fields",
                                "boost": text_weight
                            }
                        }
                    ],
                    "filter": filters,
                    "minimum_should_match": 1
                }
            }
        }
        
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            return self._parse_search_results(response)
        except Exception as e:
            # Fallback to simpler hybrid if script_score not supported
            print(f"Hybrid search fallback: {e}")
            return self._hybrid_search_fallback(query_text, query_embedding, entitlement, org_id, tags, top_k, vector_weight, text_weight)
    
    def _hybrid_search_fallback(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[Dict]:
        """
        Fallback hybrid search using RRF (Reciprocal Rank Fusion).
        
        Runs vector and text search separately, then combines results.
        """
        # Get vector results
        vector_results = self.vector_search(query_embedding, entitlement, org_id, tags, top_k * 2)
        
        # Get text/BM25 results
        text_results = self.text_search(query_text, entitlement, org_id, tags, top_k * 2)
        
        # Combine using RRF (Reciprocal Rank Fusion)
        return self._rrf_combine(vector_results, text_results, vector_weight, text_weight, top_k)
    
    def text_search(
        self,
        query_text: str,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """Text search using BM25 (OpenSearch's default text scoring)"""
        
        filters = self._build_filters(entitlement, org_id, tags)
        
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["content", "title^2"],
                                "type": "best_fields"
                            }
                        }
                    ],
                    "filter": filters
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        return self._parse_search_results(response)
    
    def _rrf_combine(
        self,
        vector_results: List[Dict],
        text_results: List[Dict],
        vector_weight: float,
        text_weight: float,
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each result list
        This is the industry standard method (used by Elasticsearch, Azure AI Search).
        """
        scores = {}
        doc_data = {}
        
        # Score from vector results
        for rank, result in enumerate(vector_results):
            doc_id = result['chunk_id'] or result['doc_id']
            rrf_score = vector_weight * (1.0 / (k + rank + 1))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = result
        
        # Score from text results
        for rank, result in enumerate(text_results):
            doc_id = result['chunk_id'] or result['doc_id']
            rrf_score = text_weight * (1.0 / (k + rank + 1))
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = result
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            result = doc_data[doc_id].copy()
            result['score'] = score
            results.append(result)
        
        return results
    
    def _build_filters(
        self,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None
    ) -> List[Dict]:
        """Build filter clauses"""
        filters = []
        
        filters.append({
            "bool": {
                "should": [
                    {"term": {"entitlement": entitlement}},
                    {"term": {"entitlement": "universal"}}
                ],
                "minimum_should_match": 1
            }
        })
        
        if org_id:
            filters.append({"term": {"org_id": org_id}})
        
        if tags:
            filters.append({"terms": {"tags": tags}})
        
        return filters
    
    def _parse_search_results(self, response: Dict) -> List[Dict]:
        """Parse OpenSearch response"""
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
                'org_id': source.get('org_id', ''),
                'tags': source.get('tags', []),
                'summary': source.get('summary', ''),
                'score': hit['_score']
            })
        
        return results
