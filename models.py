"""
AI Models - Embedding and LLM Invocation

All SageMaker endpoint calls for embeddings and LLM are centralized here.
"""

import json
import re
from typing import List, Dict, Tuple
import numpy as np
import boto3


class EmbeddingModel:
    """
    Qwen Embedding Model via SageMaker endpoint.
    """
    
    def __init__(self, endpoint_name: str, sagemaker_client: boto3.client):
        self.endpoint_name = endpoint_name
        self.client = sagemaker_client
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        params = {
            "inputs": [text], 
            "encoding_format": "float"
        }
        
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(params)
        )
        output_data = json.loads(response['Body'].read().decode())
        
        return np.array(output_data[0], dtype='float32')
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts"""
        return [self.get_embedding(text) for text in texts]


class LLMModel:
    """
    Llama LLM via SageMaker endpoint.
    """
    
    def __init__(
        self,
        endpoint_name: str,
        sagemaker_client: boto3.client,
        max_tokens: int = 512,
        temperature: float = 0.1
    ):
        self.endpoint_name = endpoint_name
        self.client = sagemaker_client
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def invoke(self, prompt: str) -> str:
        """Call LLM with prompt and return response text"""
        params = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=json.dumps(params)
        )
        output_data = json.loads(response['Body'].read().decode())
        
        if isinstance(output_data, list) and len(output_data) > 0:
            return output_data[0].get('generated_text', '')
        elif isinstance(output_data, dict):
            return output_data.get('generated_text', output_data.get('output', ''))
        
        return str(output_data)
    
    def generate_answer_with_document_selection(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        Generate answer and select the best document.
        
        Returns:
            {
                'answer': str,
                'selected_document_number': int,
                'raw_response': str
            }
        """
        # Build numbered document context
        context = "\n\n".join([
            f"[Document {i+1}]: {chunk['title']}\n{chunk['content']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Build conversation history
        history_context = ""
        if conversation_history:
            history_context = "Previous conversation:\n"
            for turn in conversation_history:
                history_context += f"User: {turn['query']}\n"
                history_context += f"Assistant: {turn['answer']}\n\n"
        
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

        # Call LLM
        raw_response = self.invoke(prompt)
        
        # Parse response
        answer_text, selected_doc_num = self._parse_document_selection_response(raw_response)
        
        return {
            'answer': answer_text,
            'selected_document_number': selected_doc_num,
            'raw_response': raw_response
        }
    
    def _parse_document_selection_response(self, response: str) -> Tuple[str, int]:
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


class ModelFactory:
    """
    Factory to create model instances with shared SageMaker client.
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self._sagemaker_client = None
    
    @property
    def sagemaker_client(self) -> boto3.client:
        """Lazy-load SageMaker client"""
        if self._sagemaker_client is None:
            self._sagemaker_client = boto3.client(
                'sagemaker-runtime',
                region_name=self.region
            )
        return self._sagemaker_client
    
    def create_embedding_model(self, endpoint_name: str) -> EmbeddingModel:
        """Create embedding model instance"""
        return EmbeddingModel(
            endpoint_name=endpoint_name,
            sagemaker_client=self.sagemaker_client
        )
    
    def create_llm_model(
        self,
        endpoint_name: str,
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> LLMModel:
        """Create LLM model instance"""
        return LLMModel(
            endpoint_name=endpoint_name,
            sagemaker_client=self.sagemaker_client,
            max_tokens=max_tokens,
            temperature=temperature
        )
