"""
OpenSearch Client - Configuration and Operations

All OpenSearch connection, index management, and search operations are centralized here.
"""

import json
from typing import List, Dict, Optional
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
            info = self.client.info()
            count = self.get_document_count()
            return {
                'status': 'healthy',
                'cluster_name': info['cluster_name'],
                'version': info['version']['number'],
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
                    # Vector field for k-NN search
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
                    # Text fields
                    "content": {"type": "text", "analyzer": "standard"},
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
    
    def bulk_index(
        self,
        documents: List[Dict],
        batch_size: int = 50
    ) -> Dict:
        """
        Bulk index documents.
        
        Each document should have:
        - embedding: vector (list or numpy array)
        - content: text
        - title: string
        - doc_id: string
        - chunk_id: string (optional)
        - chunk_index: int (optional)
        - entitlement: list (optional)
        - org_id: string (optional)
        - tags: list (optional)
        - summary: string (optional)
        """
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
        """
        Vector similarity search with filtering.
        """
        # Build filters
        filters = self._build_filters(entitlement, org_id, tags)
        
        # OpenSearch k-NN query
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
    
    def _build_filters(
        self,
        entitlement: str,
        org_id: str = None,
        tags: List[str] = None
    ) -> List[Dict]:
        """Build filter clauses for OpenSearch query"""
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
    
    def _parse_search_results(self, response: Dict) -> List[Dict]:
        """Parse OpenSearch response into list of results"""
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


class OpenSearchClientFactory:
    """
    Factory to create OpenSearch client instances.
    """
    
    _instance: OpenSearchClient = None
    
    @classmethod
    def get_client(
        cls,
        host: str,
        region: str = 'us-east-1',
        index_name: str = 'documents',
        embedding_dimension: int = 1024
    ) -> OpenSearchClient:
        """Get or create OpenSearch client (singleton)"""
        if cls._instance is None:
            cls._instance = OpenSearchClient(
                host=host,
                region=region,
                index_name=index_name,
                embedding_dimension=embedding_dimension
            )
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton instance"""
        cls._instance = None
