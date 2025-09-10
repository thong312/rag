from rank_bm25 import BM25Okapi
from ..utils.preprocessing import preprocess_text
from ..utils.cache import CacheManager
from typing import List, Tuple, Dict, Any, Optional

class BM25Search:
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.bm25 = None
        self.documents = []
        self._load_or_build_index()

    def _load_or_build_index(self):
        """Load BM25 index from cache or build new"""
        self.bm25, self.documents = self.cache_manager.load_bm25_cache()
        if self.bm25 and self.documents:
            print(f"BM25 index loaded with {len(self.documents)} documents")

    def build_index(self, documents: List[Any]):
        """Build BM25 index from documents"""
        self.documents = documents
        tokenized_docs = [preprocess_text(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.cache_manager.save_bm25_cache(self.bm25, self.documents)
        print(f"BM25 index built with {len(self.documents)} documents")

    def search(self, query: str, k: int = 10, 
              metadata_filter: Optional[Dict] = None) -> List[Tuple[Any, float]]:
        """BM25 search"""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        scored_docs = []
        for i, score in enumerate(scores):
            if i < len(self.documents):
                doc = self.documents[i]
                if self._matches_filter(doc, metadata_filter):
                    scored_docs.append((doc, float(score)))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    def _matches_filter(self, doc: Any, metadata_filter: Optional[Dict]) -> bool:
        if not metadata_filter:
            return True
        return all(doc.metadata.get(k) == v for k, v in metadata_filter.items())