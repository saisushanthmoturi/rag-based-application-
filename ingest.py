import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def ingest_documents(data_dir="data", db_dir=".chroma_db", collection_name="policies"):
    """
    Ingests PDF documents from data_dir into a local ChromaDB.
    """
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"No documents found in {data_dir}. Please add PDFs to ingest.")
        return False
    
    print(f"Loading documents from {data_dir}...")
    # Load PDFs
    reader = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".pdf"])
    documents = reader.load_data()
    
    if not documents:
        print("No valid PDF documents found.")
        return False

    print(f"Loaded {len(documents)} document pages/chunks.")

    # Setup the local ChromaDB
    print(f"Initializing ChromaDB at {db_dir}...")
    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection(
        collection_name, 
        metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Setup the embedding model
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Setup text splitting (512 tokens)
    print("Splitting text into 512-token chunks...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # Create the index and store in ChromaDB
    print("Creating index and generating embeddings (this may take a moment)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True
    )
    
    print("Ingestion complete. Documents are stored in ChromaDB.")
    return True

if __name__ == "__main__":
    ingest_documents()
