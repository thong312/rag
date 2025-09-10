# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from rag_handler import RAGHandler
# from models import get_llm, get_rag_prompt, get_retriever_prompt
# from vector_store import VectorStoreManager

# class ChatService:
#     def __init__(self):
#         self.llm = get_llm()
#         self.vector_manager = VectorStoreManager()
#         self.chat_history = []
#         self.rag_handler = RAGHandler()
        
#     def simple_chat(self, query):
#         """Simple chat without RAG"""
#         print(f"Query: {query}")
#         try:
#             response = self.llm.invoke(query)
#             print(response)
#             return response
#         except Exception as e:
#             print(f"Error in simple chat: {e}")
#             return f"Lỗi khi xử lý câu hỏi: {str(e)}"
    
#     def rag_chat(self, query):
#         """RAG-based chat with PDF context"""
#         print(f"Query: {query}")
        
#         try:
#             # Check if vector store has documents
#             collection_info = self.vector_manager.get_collection_info()
#             if collection_info and collection_info.get('points_count', 0) == 0:
#                 return {
#                     "answer": "Không có tài liệu nào trong hệ thống. Vui lòng thêm PDF trước khi đặt câu hỏi.",
#                     "sources": []
#                 }
            
#             # Load vector store and create retriever
#             vector_store = self.vector_manager.load_vector_store()
            
#             # Fix: Use the RAGHandler's create_retriever method correctly
#             if self.rag_handler.vector_store is None:
#                 self.rag_handler.vector_store = vector_store
            
#             retriever = self.rag_handler.create_retriever(vector_store)
            
#             if retriever is None:
#                 return {
#                     "answer": "Không thể tạo retriever. Vui lòng kiểm tra lại vector store.",
#                     "sources": []
#                 }
            
#             # Create history-aware retriever
#             retriever_prompt = get_retriever_prompt()
#             history_aware_retriever = create_history_aware_retriever(
#                 llm=self.llm,
#                 retriever=retriever,
#                 prompt=retriever_prompt,
#             )
            
#             # Create document chain
#             rag_prompt = get_rag_prompt()
#             document_chain = create_stuff_documents_chain(
#                 self.llm,
#                 rag_prompt
#             )
            
#             # Create retrieval chain
#             retrieval_chain = create_retrieval_chain(
#                 history_aware_retriever,
#                 document_chain,
#             )
            
#             # Get result
#             result = retrieval_chain.invoke({
#                 "input": query,
#                 "chat_history": self.chat_history
#             })
            
#             print(result["answer"])
            
#             # Update chat history
#             self.chat_history.append(HumanMessage(content=query))
#             self.chat_history.append(AIMessage(content=result["answer"]))
            
#             # Extract sources from retrieved documents
#             sources = []
#             if "context" in result:
#                 # If context documents are available in result
#                 context_docs = result.get("context", [])
#                 sources = self.rag_handler.extract_sources(context_docs)
#             elif "source_documents" in result:
#                 # Alternative key for source documents
#                 sources = self.rag_handler.extract_sources(result["source_documents"])
            
#             return {
#                 "answer": result["answer"],
#                 "sources": sources,
#                 "chat_history_length": len(self.chat_history)
#             }
            
#         except Exception as e:
#             print(f"Error in RAG chat: {e}")
#             return {
#                 "answer": f"Lỗi khi xử lý câu hỏi RAG: {str(e)}",
#                 "sources": []
#             }
    
#     def rag_chat_simple(self, query, k=None, threshold=None, metadata_filter=None):
#         """Simplified RAG chat using RAGHandler's rag_query method"""
#         print(f"Simple RAG Query: {query}")
        
#         try:
#             # Use RAGHandler's rag_query method
#             rag_result = self.rag_handler.rag_query(
#                 query=query, 
#                 k=k, 
#                 threshold=threshold, 
#                 metadata_filter=metadata_filter
#             )
            
#             if not rag_result["context"] or "Không tìm thấy" in rag_result["context"]:
#                 return {
#                     "answer": "Không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
#                     "sources": rag_result["sources"],
#                     "context": rag_result["context"]
#                 }
            
#             # Create prompt with context
#             rag_prompt_template = get_rag_prompt()
            
#             # Format the prompt with context and query
#             formatted_prompt = rag_prompt_template.format(
#                 context=rag_result["context"],
#                 input=query
#             )
            
#             # Get LLM response
#             response = self.llm.invoke(formatted_prompt)
            
#             # Handle different response types
#             if hasattr(response, 'content'):
#                 answer = response.content
#             else:
#                 answer = str(response)
            
#             print(f"Answer: {answer}")
            
#             return {
#                 "answer": answer,
#                 "sources": rag_result["sources"],
#                 "context": rag_result["context"],
#                 "num_documents": rag_result.get("num_documents", 0)
#             }
            
#         except Exception as e:
#             print(f"Error in simple RAG chat: {e}")
#             return {
#                 "answer": f"Lỗi khi xử lý câu hỏi: {str(e)}",
#                 "sources": [],
#                 "context": ""
#             }
    
#     def chat_with_history(self, query, use_rag=True):
#         """Chat with automatic RAG decision and history management"""
#         if use_rag:
#             result = self.rag_chat_simple(query)
#         else:
#             result = {"answer": self.simple_chat(query), "sources": []}
        
#         # Add to chat history
#         self.chat_history.append(HumanMessage(content=query))
#         self.chat_history.append(AIMessage(content=result["answer"]))
        
#         return result
    
#     def clear_history(self):
#         """Clear chat history"""
#         self.chat_history = []
#         print("Chat history cleared")
    
#     def get_chat_history(self):
#         """Get current chat history"""
#         return self.chat_history
    
#     def get_chat_history_formatted(self):
#         """Get formatted chat history for display"""
#         formatted_history = []
#         for i, message in enumerate(self.chat_history):
#             if isinstance(message, HumanMessage):
#                 formatted_history.append({
#                     "type": "human",
#                     "content": message.content,
#                     "index": i
#                 })
#             elif isinstance(message, AIMessage):
#                 formatted_history.append({
#                     "type": "ai", 
#                     "content": message.content,
#                     "index": i
#                 })
#         return formatted_history
    
#     def add_pdf(self, file_path):
#         """Add PDF to the system and refresh RAG handler"""
#         try:
#             result = self.rag_handler.add_pdf(file_path)
#             if result["success"]:
#                 print(f"PDF added successfully: {result['message']}")
#             else:
#                 print(f"Failed to add PDF: {result['message']}")
#             return result
#         except Exception as e:
#             print(f"Error adding PDF: {e}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "message": f"Failed to add PDF: {str(e)}"
#             }
    
#     def reset_system(self):
#         """Reset vector store and chat history"""
#         try:
#             # Reset vector store
#             reset_result = self.rag_handler.reset_vector_store()
            
#             # Clear chat history
#             self.clear_history()
            
#             if reset_result["success"]:
#                 return {
#                     "success": True,
#                     "message": "System reset successfully. Vector store cleared and chat history reset."
#                 }
#             else:
#                 return reset_result
                
#         except Exception as e:
#             print(f"Error resetting system: {e}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "message": f"Failed to reset system: {str(e)}"
#             }
    
#     def get_system_info(self):
#         """Get information about the current system state"""
#         try:
#             collection_info = self.vector_manager.get_collection_info()
#             return {
#                 "vector_store": collection_info,
#                 "chat_history_length": len(self.chat_history),
#                 "has_documents": collection_info.get('points_count', 0) > 0 if collection_info else False
#             }
#         except Exception as e:
#             print(f"Error getting system info: {e}")
#             return {
#                 "error": str(e),
#                 "chat_history_length": len(self.chat_history),
#                 "has_documents": False
#             }

# # Convenience functions for quick usage
# def create_chat_service():
#     """Create a new ChatService instance"""
#     return ChatService()

# def quick_chat(query, use_rag=True):
#     """Quick chat function"""
#     service = create_chat_service()
#     return service.chat_with_history(query, use_rag=use_rag)