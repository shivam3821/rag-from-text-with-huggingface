import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(docs_path = "docs"):
    print("Loading Documents from", docs_path)

    # Check if docs directory exist or not? 

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls= TextLoader
    )

    documents = loader.load() # give a list of langchain document

    if len(documents) == 0:
        raise FileNotFoundError("File not found")

    # Let's see if documents are loaded properly
    # for i, doc in enumerate(documents[:2]):
    #     print(f"\nDocument {i+1}:")
    #     print(f"Source {doc.metadata['source']}:")
    #     print(f"Content length: {len(doc.page_content)} characters")
    #     print(f"Content preview: {doc.page_content[:100]} . . .")
    #     print(f"MetaData: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting Documents into Chunks . . .")

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # if chunks: 
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n---Chunk{i+1}---")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Contents:")
    #         print(chunk.page_content)
    #         print("-"*50)
    #     if len(chunks) > 5:
    #         print(f"\n ... and {len(chunks)-5} more chunks")
    
    empty_docs = [doc for doc in chunks if not doc.page_content.strip()]
    print(len(chunks))

    return chunks

# Create vector DB

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Create Chrome Vector Store
    print("---Creating Vector Store--")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}")
    return vector_store

    
def main():
    
    # 1. Loading the document
    documents = load_documents(docs_path="docs")

    # 2. Chunking the document
    chunks = split_documents(documents)

    # 3. Embedding and Storing in Vector DB
    vector_store = create_vector_store(chunks)

main()
