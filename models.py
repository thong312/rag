from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from config import (
    OLLAMA_MODEL, 
    EMBEDDING_MODEL, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP
)

# Initialize LLM
def get_llm():
    return OllamaLLM(model=OLLAMA_MODEL)

# Initialize embeddings
def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialize text splitter
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

# Prompt template for RAG
def get_rag_prompt():
    return PromptTemplate.from_template(
        """ 
        <s>[INST] You are a technical assistant that answers strictly based on the given context. 
        - If the answer to the user's question is in the context, answer it clearly.  
        - If the answer is not in the context, reply exactly: "I don't know".  
        - Do not answer unrelated questions or make assumptions. [/INST] </s>
        
        [INST] Question: {input}
        Context: {context}
        Answer:
        [/INST]
        """
    )



# Prompt template for retriever
def get_retriever_prompt():
    return ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the conversation above, please answer the question based on the provided context."),
    ])
    # return ChatPromptTemplate.from_messages([
    #     ("system", 
    #      "You are an helpful assistant that provides answers based only on the provided context. "
    #      "Format your response striclty as JSON with keys:'answer', 'sources'."
    #      "Do not add any extra text."  ),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human", "{input}"),
    #     ("human", "Given the conversation above, please answer the question based on the provided context."),
    # ])