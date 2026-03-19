import pandas as pd
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Data from Excel
def load_excel_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    documents = []
    for index, row in df.iterrows():
        # Convert row to string: "Column: Value | Column: Value"
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        
        doc = Document(
            page_content=row_text,
            metadata={"source": file_path, "row": index}
        )
        documents.append(doc)
    return documents

# 2. Split Data
def split_documents(documents):
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

# 3. Create Vector Store using LOCAL Ollama Embeddings
def create_vector_store(splits):
    print("Creating vector store using local Llama3 embeddings...")
    # This connects to your local Ollama to generate vectors
    embedding_function = OllamaEmbeddings(model="llama3")
    
    # This might take a minute if you have a lot of data because it runs on CPU
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedding_function,
        persist_directory="./chroma_db_local"
    )
    return vectorstore

def main():
    # Make sure your Excel file is named 'data.xlsx' or change this
    excel_file = "data.xlsx" 
    
    # --- LOAD AND PREPARE ---
    raw_docs = load_excel_data(excel_file)
    splits = split_documents(raw_docs)
    
    # --- CREATE DATABASE ---
    vectorstore = create_vector_store(splits)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # --- INITIALIZE LOCAL LLM ---
    print("Initializing local LLM (Llama 3)...")
    # This connects to the Ollama app running on your computer
    llm = ChatOllama(model="llama3")
    
    # --- PROMPT ---
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, say that you don't know."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # --- CHAIN ---
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("\nSystem Ready! (Using Local Llama 3)")
    print("Ask questions about your Excel data (type 'quit' to exit).\n")
    
    # --- LOOP ---
    while True:
        query = input("Question: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        # When you run this, Ollama will use your CPU/GPU to generate the answer
        result = rag_chain.invoke({"input": query})
        print(f"\nAnswer: {result['answer']}\n")

if __name__ == "__main__":
    main()