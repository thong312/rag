from typing import Dict, Any, Optional
from .search.bm25 import BM25Search
from .search.vector import VectorSearch
from .search.hybrid import HybridSearch
from .retrieval.reranker import CrossEncoderReranker
from .retrieval.retriever import DocumentRetriever
from .utils.context import ContextFormatter
from config import SIMILARITY_SEARCH_K

class RAGHandler:
    def __init__(self):
        self.vector_search = VectorSearch()
        self.bm25_search = BM25Search()
        self.hybrid_search = HybridSearch(self.bm25_search, self.vector_search)
        self.reranker = CrossEncoderReranker()
        self.retriever = DocumentRetriever()
        self.context_formatter = ContextFormatter()

    def rag_query_hybrid(self, query: str, k: Optional[int] = None, 
                        alpha: float = 0.5, include_sources: bool = True,
                        metadata_filter: Optional[Dict] = None, 
                        use_rerank: bool = True) -> Dict[str, Any]:
        """RAG pipeline with hybrid search"""
        k = k or SIMILARITY_SEARCH_K

        try:
            # Get candidate documents
            candidates = self.hybrid_search.search(
                query=query,
                k=k * 2,
                alpha=alpha,
                metadata_filter=metadata_filter
            )
            documents = [doc for doc, _ in candidates]

            # Rerank if needed
            if use_rerank:
                documents = self.reranker.rerank(query, documents, top_k=k)
            else:
                documents = documents[:k]

            # Format context and get response
            context = self.context_formatter.format_documents(documents)
            response = self.retriever.get_llm_response(query, context)
            
            return {
                "answer": response,
                "sources": self.context_formatter.extract_sources(documents) if include_sources else []
            }

        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }