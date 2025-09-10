import streamlit as st
import requests

from typing import Dict, Any

# Constants
API_URL = "http://127.0.0.1:8080"  

def send_chat_request(query: str, search_type: str = "hybrid", **kwargs) -> Dict[str, Any]:
    """Send chat request to backend API"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "query": query,
                "search_type": search_type,
                **kwargs
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def upload_pdf(file):
    """Upload PDF file to backend"""
    try:
        files = {"file": file}
        response = requests.post(f"{API_URL}/pdf", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def check_api_connection():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def main():
    st.set_page_config(
        page_title="RAG Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Chat Assistant")

    # Check API connection
    if not check_api_connection():
        st.error("""
        ⚠️ Cannot connect to backend API. Please ensure:
        1. Backend Flask server is running on port 5000
        2. You started backend with `python app.py`
        3. There are no firewall issues blocking the connection
        """)
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # PDF Upload
        st.subheader("Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
        if pdf_file:
            if st.button("Upload PDF"):
                with st.spinner("Uploading and processing PDF..."):
                    result = upload_pdf(pdf_file)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"PDF uploaded successfully! {result['chunks']} chunks created")

        # Search Settings
        st.subheader("Search Settings")
        search_type = st.selectbox(
            "Search Method",
            ["hybrid", "rag", "simple"],
            help="Choose search method"
        )
        
        if search_type == "hybrid":
            alpha = st.slider(
                "Alpha (Vector vs BM25)", 
                0.0, 1.0, 0.5,
                help="1.0 = Pure Vector, 0.0 = Pure BM25"
            )
            use_rerank = st.checkbox("Use Reranking", value=True)
            k = st.number_input("Number of results", min_value=1, value=3)

    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        **Source {idx}:**
                        - File: {source['file_name']}
                        - Page: {source['page']}
                        - Chunk: {source['chunk']}
                        """)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if search_type == "hybrid":
                    response = send_chat_request(
                        prompt,
                        search_type="hybrid",
                        alpha=alpha,
                        k=k,
                        use_rerank=use_rerank
                    )
                else:
                    response = send_chat_request(prompt, search_type=search_type)

                if "error" in response:
                    st.error(response["error"])
                else:
                    st.write(response["answer"])
                    if response.get("sources"):
                        with st.expander("View Sources"):
                            for idx, source in enumerate(response["sources"], 1):
                                st.markdown(f"""
                                **Source {idx}:**
                                - File: {source['file_name']}
                                - Page: {source['page']}
                                - Chunk: {source['chunk']}
                                """)

                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.get("answer", "Error occurred"),
                    "sources": response.get("sources", [])
                })

    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        requests.post(f"{API_URL}/clear_history")
        st.rerun()

if __name__ == "__main__":
    main()