from typing import List, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Any], top_k: int = 10) -> List[Any]:
        """
        Rerank documents using CrossEncoder
        Args:
            query: user query
            docs: list of documents (langchain Document objects) 
            top_k: number of documents to return after reranking
        """
        if not docs:
            return []

        try:
            # Prepare data for CrossEncoder: [(query, doc_text), ...]
            pairs = [(query, doc.page_content) for doc in docs]

            # Calculate relevance scores
            scores = self.reranker.predict(pairs)

            # Combine scores with docs
            scored = list(zip(docs, scores))

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Return top_k docs
            reranked_docs = [doc for doc, score in scored[:top_k]]

            print(f"Reranking complete: selected {len(reranked_docs)} docs")
            return reranked_docs

        except Exception as e:
            print(f"Error in reranking: {e}")
            return docs[:top_k]