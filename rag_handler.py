# from vector_store import VectorStoreManager
# from config import (
#     SIMILARITY_SEARCH_K,
#     SIMILARITY_THRESHOLD
# )
# from rank_bm25 import BM25Okapi
# import numpy as np

# from sklearn.metrics.pairwise import cosine_similarity
# import re
# import unicodedata
# from typing import List, Dict, Any, Optional, Tuple
# import pickle
# import os
# from models import get_llm, get_rag_prompt
# from sentence_transformers import CrossEncoder
# class RAGHandler:
#     def __init__(self):
#         self.vector_manager = VectorStoreManager()
#         self.vector_store = None
#         self.retriever = None
        
#         # Hybrid search components
#         self.bm25 = None
#         self.documents = []  # Cache documents for BM25
#         self.doc_embeddings = {}  # Cache embeddings
#         self.bm25_cache_path = "bm25_cache.pkl"
#         self.docs_cache_path = "docs_cache.pkl"
        
#         self._initialize_retriever()
#         self._load_or_build_bm25_index()
    
#         self.rag_prompt = get_rag_prompt()
#         self.llm = get_llm()

#         self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#     def _initialize_retriever(self):
#         """Khởi tạo retriever từ vector store"""
#         try:
#             self.vector_store = self.vector_manager.load_vector_store()
#             self.retriever = self.create_retriever()
#             if self.retriever is None:
#                 print("Warning: Retriever could not be created, but vector store exists")
#         except Exception as e:
#             print(f"Error initializing retriever: {e}")
#             print("Attempting to create empty vector store...")
#             try:
#                 self.vector_store = self.vector_manager.load_vector_store()
#                 self.retriever = self.create_retriever()
#                 print("Empty vector store created successfully")
#             except Exception as e2:
#                 print(f"Failed to create empty vector store: {e2}")
#                 self.vector_store = None
#                 self.retriever = None
    
#     def create_retriever(self, vector_store=None):
#         """Tạo retriever từ vector store"""
#         if self.vector_store is None:
#             print("Cannot create retriever: vector store is None")
#             return None
        
#         try:
#             retriever = self.vector_store.as_retriever(
#                 search_type="similarity_score_threshold",
#                 search_kwargs={
#                     "k": SIMILARITY_SEARCH_K,
#                     "score_threshold": SIMILARITY_THRESHOLD
#                 },
#             )
#             print("Retriever created successfully")
#             return retriever
#         except Exception as e:
#             print(f"Error creating retriever: {e}")
#             try:
#                 fallback_retriever = self.vector_store.as_retriever(
#                     search_kwargs={"k": SIMILARITY_SEARCH_K}
#                 )
#                 print("Fallback retriever created")
#                 return fallback_retriever
#             except Exception as e2:
#                 print(f"Fallback retriever also failed: {e2}")
#                 return None
    
#     def _preprocess_text(self, text: str) -> List[str]:
#         """Tiền xử lý text cho BM25"""
#         # Chuẩn hóa Unicode
#         text = unicodedata.normalize('NFKC', text)
#         # Chuyển về lowercase
#         text = text.lower()
#         # Loại bỏ ký tự đặc biệt, chỉ giữ lại chữ, số và khoảng trắng
#         text = re.sub(r'[^\w\s]', ' ', text)
#         # Tách từ
#         tokens = text.split()
#         return tokens
    
#     def _load_or_build_bm25_index(self):
#         """Load BM25 index từ cache hoặc build mới"""
#         try:
#             # Kiểm tra cache
#             if os.path.exists(self.bm25_cache_path) and os.path.exists(self.docs_cache_path):
#                 print("Loading BM25 index from cache...")
#                 with open(self.bm25_cache_path, 'rb') as f:
#                     self.bm25 = pickle.load(f)
#                 with open(self.docs_cache_path, 'rb') as f:
#                     self.documents = pickle.load(f)
#                 print(f"BM25 index loaded with {len(self.documents)} documents")
#             else:
#                 print("Building BM25 index...")
#                 self._build_bm25_index()
#         except Exception as e:
#             print(f"Error loading BM25 cache: {e}")
#             print("Rebuilding BM25 index...")
#             self._build_bm25_index()
    
#     def _build_bm25_index(self):
#         """Build BM25 index từ vector store"""
#         try:
#             if self.vector_store is None:
#                 print("Cannot build BM25: vector store is None")
#                 return
            
#             # Lấy tất cả documents từ vector store
#             collection_info = self.vector_manager.get_collection_info()
#             if collection_info and collection_info.get('points_count', 0) > 0:
#                 # Sử dụng similarity_search với query rộng để lấy nhiều docs
#                 all_docs = self.vector_store.similarity_search("", k=collection_info['points_count'])
                
#                 # Nếu không lấy được, thử với query khác
#                 if not all_docs:
#                     all_docs = self.vector_store.similarity_search("the", k=1000)
                
#                 self.documents = all_docs
                
#                 # Tiền xử lý text cho BM25
#                 tokenized_docs = []
#                 for doc in self.documents:
#                     tokens = self._preprocess_text(doc.page_content)
#                     tokenized_docs.append(tokens)
                
#                 if tokenized_docs:
#                     self.bm25 = BM25Okapi(tokenized_docs)
                    
#                     # Save cache
#                     with open(self.bm25_cache_path, 'wb') as f:
#                         pickle.dump(self.bm25, f)
#                     with open(self.docs_cache_path, 'wb') as f:
#                         pickle.dump(self.documents, f)
                    
#                     print(f"BM25 index built with {len(self.documents)} documents")
#                 else:
#                     print("No documents to build BM25 index")
#             else:
#                 print("No documents in vector store for BM25")
                
#         except Exception as e:
#             print(f"Error building BM25 index: {e}")
    
#     def _bm25_search(self, query: str, k: int = 10, metadata_filter: Optional[Dict] = None) -> List[Tuple[Any, float]]:
#         """BM25 keyword search"""
#         if self.bm25 is None or not self.documents:
#             print("BM25 index not available")
#             return []
        
#         try:
#             # Tiền xử lý query
#             tokenized_query = self._preprocess_text(query)
            
#             # BM25 scoring
#             scores = self.bm25.get_scores(tokenized_query)
            
#             # Lấy top k results với scores
#             scored_docs = []
#             for i, score in enumerate(scores):
#                 if i < len(self.documents):
#                     doc = self.documents[i]
#                     # Apply metadata filter
#                     if metadata_filter:
#                         match = True
#                         for key, value in metadata_filter.items():
#                             if doc.metadata.get(key) != value:
#                                 match = False
#                                 break
#                         if not match:
#                             continue
                    
#                     scored_docs.append((doc, float(score)))
            
#             # Sort by score descending
#             scored_docs.sort(key=lambda x: x[1], reverse=True)
            
#             # Return top k
#             return scored_docs[:k]
            
#         except Exception as e:
#             print(f"Error in BM25 search: {e}")
#             return []
    
#     def _vector_search(self, query: str, k: int = 10, metadata_filter: Optional[Dict] = None) -> List[Tuple[Any, float]]:
#         """Vector semantic search"""
#         try:
#             if metadata_filter:
#                 results = self.vector_store.similarity_search_with_score(
#                     query, k=k, filter=metadata_filter
#                 )
#             else:
#                 results = self.vector_store.similarity_search_with_score(query, k=k)
            
#             # Convert to consistent format (doc, score)
#             return [(doc, float(score)) for doc, score in results]
            
#         except Exception as e:
#             print(f"Error in vector search: {e}")
#             return []
    
    
    
#     # def retrieve_documents_hybrid(self, query, k=None, alpha=0.5, metadata_filter=None):
#     #     """Retrieve documents using hybrid search"""
#     #     k = k or SIMILARITY_SEARCH_K
        
#     #     try:
#     #         scored_docs = self.hybrid_search(
#     #             query=query,
#     #             k=k, 
#     #             alpha=alpha,
#     #             metadata_filter=metadata_filter
#     #         )
            
#     #         # Extract documents (remove scores)
#     #         documents = [doc for doc, score in scored_docs]
            
#     #         print(f"Retrieved {len(documents)} documents via hybrid search")
#     #         return documents
            
#     #     except Exception as e:
#     #         print(f"Error in hybrid retrieve: {e}")
#     #         return []


#     def _rerank_with_cross_encoder(self, query: str, docs: List[Any], top_k: int = 10) -> List[Any]:
#         """
#         Rerank documents using CrossEncoder
#         Args:
#         query: user query
#         docs: list of documents (langchain Document objects)
#         top_k: number of documents to return after reranking
#         """
#         if not docs:
#             return []

#         try:
#             # Chuẩn bị dữ liệu cho CrossEncoder: [(query, doc_text), ...]
#             pairs = [(query, doc.page_content) for doc in docs]

#             # Tính điểm relevance
#             scores = self.reranker.predict(pairs)

#             # Gắn score lại với doc
#             scored = list(zip(docs, scores))

#             # Sắp xếp theo score giảm dần
#             scored.sort(key=lambda x: x[1], reverse=True)

#             # Trả về danh sách doc (top_k)
#             reranked_docs = [doc for doc, score in scored[:top_k]]

#             print(f"Reranking complete: selected {len(reranked_docs)} docs")
#             print(f"docs", [doc.page_content[:50] + "..." for doc in reranked_docs])
#             return reranked_docs

#         except Exception as e:
#             print(f"Error in reranking: {e}")
#             return docs[:top_k]
    
# #HERE
#     def hybrid_search(self, 
#                      query: str, 
#                      k: int = 10, 
#                      alpha: float = 0.5,
#                      metadata_filter: Optional[Dict] = None,
#                      bm25_k: Optional[int] = None,
#                      vector_k: Optional[int] = None) -> List[Tuple[Any, float]]:
#         """
#         Hybrid search combining BM25 and vector search
        
#         Args:
#             query: Search query
#             k: Final number of results
#             alpha: Weight for vector search (0.0 = only BM25, 1.0 = only vector)
#             metadata_filter: Optional metadata filter
#             bm25_k: Number of results from BM25 (default: k*2)
#             vector_k: Number of results from vector search (default: k*2)
#         """
#         if self.vector_store is None:
#             print("Vector store not available")
#             return []
        
#         # Default values
#         if bm25_k is None:
#             bm25_k = max(k * 2, 20)
#         if vector_k is None:
#             vector_k = max(k * 2, 20)
        
#         try:
#             # Get results from both methods
#             bm25_results = self._bm25_search(query, bm25_k, metadata_filter)
#             vector_results = self._vector_search(query, vector_k, metadata_filter)
            
#             print(f"BM25 found {len(bm25_results)} results")
#             print(f"Vector found {len(vector_results)} results")
            
#             # Normalize scores
#             bm25_scores = [score for _, score in bm25_results]
#             vector_scores = [score for _, score in vector_results]
            
#             # Normalize BM25 scores (0-1)
#             if bm25_scores:
#                 max_bm25 = max(bm25_scores) if bm25_scores else 1.0
#                 min_bm25 = min(bm25_scores) if bm25_scores else 0.0
#                 if max_bm25 > min_bm25:
#                     bm25_results = [(doc, (score - min_bm25) / (max_bm25 - min_bm25)) 
#                                    for doc, score in bm25_results]
            
#             # Normalize vector scores (similarity scores, usually 0-1 already)
#             # For some vector stores, higher scores = better similarity
#             # For others (like distance), lower = better. Adjust if needed.
            
#             # Combine results
#             combined_scores = {}
            
#             # Add BM25 scores
#             for doc, score in bm25_results:
#                 doc_key = (doc.page_content, str(doc.metadata))
#                 combined_scores[doc_key] = {
#                     'doc': doc,
#                     'bm25_score': score,
#                     'vector_score': 0.0
#                 }
            
#             # Add vector scores
#             for doc, score in vector_results:
#                 doc_key = (doc.page_content, str(doc.metadata))
#                 if doc_key in combined_scores:
#                     combined_scores[doc_key]['vector_score'] = score
#                 else:
#                     combined_scores[doc_key] = {
#                         'doc': doc,
#                         'bm25_score': 0.0,
#                         'vector_score': score
#                     }
            
#             # Calculate hybrid scores
#             final_results = []
#             for doc_key, scores in combined_scores.items():
#                 hybrid_score = (1 - alpha) * scores['bm25_score'] + alpha * scores['vector_score']
#                 final_results.append((scores['doc'], hybrid_score))
            
#             # Sort by hybrid score
#             final_results.sort(key=lambda x: x[1], reverse=True)
            
#             print(f"Hybrid search returning top {min(k, len(final_results))} results")
#             return final_results[:k]
            
#         except Exception as e:
#             print(f"Error in hybrid search: {e}")
#             return []

#     def rag_query_hybrid(self, query, k=None, alpha=0.5, include_sources=True, metadata_filter=None, use_rerank=True):
#         """RAG pipeline with hybrid search"""
#         k = k or SIMILARITY_SEARCH_K

#         try:
#             candidates = self.hybrid_search(
#                 query=query,
#                 k=k * 2,
#                 alpha=alpha,
#                 metadata_filter=metadata_filter
#             )
#             documents = [doc for doc, score in candidates]

#             if use_rerank:
#                 documents = self._rerank_with_cross_encoder(query, documents, top_k=k)
#             else:
#                 documents = documents[:k]

#             context = self.format_context(documents)
#             prompt = self.rag_prompt.format(
#                 input=query,
#                 context=context
#             )

#             response = self.llm.invoke(prompt)
#             sources = self.extract_sources(documents)

#             # Chỉ trả về câu trả lời và nguồn tham khảo
#             return {
#                 "answer": response.content if hasattr(response, "content") else str(response),
#                 "sources": [
#                     {
#                         "file_name": s["file_name"],
#                         "page": s["page"]
#                     }
#                     for s in sources
#                 ] if include_sources else []
#             }

#         except Exception as e:
#             return {
#                 "answer": "Xin lỗi, có lỗi xảy ra trong quá trình xử lý.",
#                 "sources": []
#             }
    
#     def extract_sources(self, documents):
#         """Extract sources from retrieved documents"""
#         sources = []
        
#         for doc in documents:
#             source_info = {
#                 "source": doc.metadata.get("source", "unknown"),
#                 "file_name": doc.metadata.get("file_name", "unknown"),
#                 "page": doc.metadata.get("page", 0),
#                 "chunk": doc.metadata.get("chunk", 0),
#                 "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
#                 "content_length": len(doc.page_content)
#             }
#             sources.append(source_info)
        
#         return sources
    
#     def format_context(self, documents):
#         """Format retrieved documents as context for LLM"""
#         if not documents:
#             return "Không tìm thấy thông tin liên quan."
        
#         context_parts = []
#         for i, doc in enumerate(documents, 1):
#             file_name = doc.metadata.get("file_name", "unknown")
#             page = doc.metadata.get("page", 0)
            
#             context_part = f"[Nguồn {i}: {file_name}, trang {page}]\n{doc.page_content}"
#             context_parts.append(context_part)
        
#         return "\n\n".join(context_parts)
    
#     def add_pdf(self, file_path):
#         """Add PDF to vector store and rebuild BM25 index"""
#         try:
#             docs_count, chunks_count = self.vector_manager.process_pdf(file_path)
            
#             # Rebuild BM25 index
#             print("Rebuilding BM25 index after adding PDF...")
#             self._build_bm25_index()
            
#             # Refresh retriever
#             self._initialize_retriever()
            
#             return {
#                 "success": True,
#                 "docs_count": docs_count,
#                 "chunks_count": chunks_count,
#                 "message": f"Successfully processed PDF: {docs_count} documents, {chunks_count} chunks"
#             }
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "message": f"Failed to process PDF: {str(e)}"
#             }
    
#     def reset_vector_store(self):
#         """Reset vector store and BM25 index"""
#         try:
#             self.vector_manager.delete_collection()
            
#             # Clear BM25 components
#             self.bm25 = None
#             self.documents = []
            
#             # Remove cache files
#             if os.path.exists(self.bm25_cache_path):
#                 os.remove(self.bm25_cache_path)
#             if os.path.exists(self.docs_cache_path):
#                 os.remove(self.docs_cache_path)
            
#             self._initialize_retriever()
#             return {"success": True, "message": "Vector store and BM25 index reset successfully"}
#         except Exception as e:
#             return {"success": False, "error": str(e)}
    
#     def compare_search_methods(self, query, k=5, metadata_filter=None):
#         """So sánh kết quả của các phương pháp tìm kiếm"""
#         try:
#             results = {}
            
#             # Vector search
#             vector_results = self._vector_search(query, k, metadata_filter)
#             results['vector'] = [(doc.page_content[:100] + "...", score) 
#                                for doc, score in vector_results]
            
#             # BM25 search  
#             bm25_results = self._bm25_search(query, k, metadata_filter)
#             results['bm25'] = [(doc.page_content[:100] + "...", score) 
#                               for doc, score in bm25_results]
            
#             # Hybrid search with different alpha values
#             for alpha in [0.3, 0.5, 0.7]:
#                 hybrid_results = self.hybrid_search(query, k, alpha, metadata_filter)
#                 results[f'hybrid_alpha_{alpha}'] = [(doc.page_content[:100] + "...", score) 
#                                                    for doc, score in hybrid_results]
            
#             return results
            
#         except Exception as e:
#             print(f"Error comparing search methods: {e}")
#             return {}

# # Convenience functions
# def create_hybrid_rag():
#     """Create a new HybridSearchRAGHandler instance"""
#     return RAGHandler()

# def quick_hybrid_query(query, alpha=0.5, k=None):
#     """Quick hybrid RAG query"""
#     rag = create_hybrid_rag()
#     return rag.rag_query_hybrid(query, k=k, alpha=alpha)