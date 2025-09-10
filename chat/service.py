
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from typing import Dict, Any, Optional, List

from rag.handler import RAGHandler
from models import get_llm, get_rag_prompt, get_retriever_prompt
from vector_store import VectorStoreManager
from .history import ChatHistory

class ChatService:
    def __init__(self):
        self.llm = get_llm()
        self.vector_manager = VectorStoreManager()
        self.chat_history = ChatHistory()
        self.rag_handler = RAGHandler()

    def simple_chat(self, query: str) -> str:
        """Simple chat without RAG"""
        print(f"Query: {query}")
        try:
            response = self.llm.invoke(query)
            print(response)
            return response
        except Exception as e:
            print(f"Error in simple chat: {e}")
            return f"Lỗi khi xử lý câu hỏi: {str(e)}"

    def rag_chat(self, query: str) -> Dict[str, Any]:
        """RAG-based chat with history-aware retrieval"""
        print(f"Query: {query}")
        
        try:
            # Verify documents exist
            if not self._verify_documents():
                return self._no_documents_response()

            # Setup retriever and chains
            retriever = self._setup_retriever()
            if not retriever:
                return self._retriever_error_response()

            # Create and execute chain
            result = self._execute_rag_chain(query, retriever)
            
            # Update history
            self.chat_history.add_human_message(query)
            self.chat_history.add_ai_message(result["answer"])
            
            return {
                "answer": result["answer"],
                "sources": self._extract_sources(result),
                "chat_history_length": len(self.chat_history)
            }
            
        except Exception as e:
            print(f"Error in RAG chat: {e}")
            return self._error_response(str(e))

    def rag_chat_simple(self, query: str, k: Optional[int] = None,
                       threshold: Optional[float] = None,
                       metadata_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """Simplified RAG chat using direct RAGHandler query"""
        print(f"Simple RAG Query: {query}")
        
        try:
            result = self.rag_handler.rag_query_hybrid(
                query=query,
                k=k,
                metadata_filter=metadata_filter
            )
            
            return {
                "answer": result["answer"],
                "sources": result["sources"]
            }
            
        except Exception as e:
            print(f"Error in simple RAG chat: {e}")
            return self._error_response(str(e))

    def hybrid_chat(self, query: str, k: Optional[int] = None,
                   alpha: float = 0.5,
                   metadata_filter: Optional[Dict] = None,
                   use_rerank: bool = True) -> Dict[str, Any]:
        """Chat using hybrid search (BM25 + Vector)"""
        print(f"Hybrid chat query: {query}")
        
        try:
            result = self.rag_handler.rag_query_hybrid(
                query=query,
                k=k,
                alpha=alpha,
                metadata_filter=metadata_filter,
                use_rerank=use_rerank
            )
            
            # Update chat history
            self.chat_history.add_human_message(query)
            self.chat_history.add_ai_message(result["answer"])
            
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "chat_history_length": len(self.chat_history)
            }
            
        except Exception as e:
            print(f"Error in hybrid chat: {e}")
            return self._error_response(str(e))

    def chat_with_history(self, query: str, search_type: str = "hybrid",
                         **kwargs) -> Dict[str, Any]:
        """Enhanced chat with choice of search method"""
        if search_type == "hybrid":
            result = self.hybrid_chat(query, **kwargs)
        elif search_type == "rag":
            result = self.rag_chat(query)
        elif search_type == "simple":
            result = {"answer": self.simple_chat(query), "sources": []}
        else:
            return self._error_response(f"Unknown search type: {search_type}")

        return result

    # System management methods
    def add_pdf(self, file_path: str) -> Dict[str, Any]:
        """Add PDF to the system"""
        return self.rag_handler.add_pdf(file_path)

    def reset_system(self) -> Dict[str, Any]:
        """Reset entire system"""
        try:
            reset_result = self.rag_handler.reset_vector_store()
            self.chat_history.clear()
            
            return {
                "success": reset_result["success"],
                "message": "System reset successfully" if reset_result["success"] else reset_result["message"]
            }
        except Exception as e:
            return {"success": False, "message": f"Reset failed: {str(e)}"}

    def get_system_info(self) -> Dict[str, Any]:
        """Get system state information"""
        try:
            collection_info = self.vector_manager.get_collection_info()
            return {
                "vector_store": collection_info,
                "chat_history_length": len(self.chat_history),
                "has_documents": collection_info.get('points_count', 0) > 0 if collection_info else False
            }
        except Exception as e:
            return {
                "error": str(e),
                "chat_history_length": len(self.chat_history),
                "has_documents": False
            }

    # Helper methods
    def _verify_documents(self) -> bool:
        collection_info = self.vector_manager.get_collection_info()
        return collection_info and collection_info.get('points_count', 0) > 0

    def _setup_retriever(self):
        vector_store = self.vector_manager.load_vector_store()
        if self.rag_handler.vector_store is None:
            self.rag_handler.vector_store = vector_store
        return self.rag_handler.create_retriever(vector_store)

    def _execute_rag_chain(self, query: str, retriever) -> Dict[str, Any]:
        retriever_prompt = get_retriever_prompt()
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=retriever_prompt
        )
        
        document_chain = create_stuff_documents_chain(
            self.llm,
            get_rag_prompt()
        )
        
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            document_chain
        )
        
        return retrieval_chain.invoke({
            "input": query,
            "chat_history": self.chat_history.get_messages()
        })

    def _extract_sources(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "context" in result:
            return self.rag_handler.extract_sources(result["context"])
        elif "source_documents" in result:
            return self.rag_handler.extract_sources(result["source_documents"])
        return []

    # Error responses
    def _no_documents_response(self) -> Dict[str, Any]:
        return {
            "answer": "Không có tài liệu nào trong hệ thống. Vui lòng thêm PDF trước khi đặt câu hỏi.",
            "sources": []
        }

    def _retriever_error_response(self) -> Dict[str, Any]:
        return {
            "answer": "Không thể tạo retriever. Vui lòng kiểm tra lại vector store.",
            "sources": []
        }

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            "answer": f"Lỗi: {error_msg}",
            "sources": []
        }