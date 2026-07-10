import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

EMBEDDING_DIM=1536

base_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",dimension=EMBEDDING_DIM)
embedding_file_store = LocalFileStore("./embedding_cache/")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)


# setting up vector store

quadrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=120
)

#collection

def get_collection_name(session_id: str) -> str:
    return f"paper_{session_id.replace('-', '_')}"

def get_vector_store(session_id: str) -> QdrantVectorStore:
    collection_name=get_collection_name(session_id)
    if not quadrant_client.collection_exists(collection_name):
        quadrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=quadrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    
    
    
def add_papers(docs:list[Document],session_id:str)->None:
    get_vector_store(session_id).add_documents(docs)
    
    
def list_papers(session_id:str)->list[str]:
    collection_name=get_collection_name(session_id)
    if not quadrant_client.collection_exists(collection_name):
        return []
    seen:set[str]=set()
    titles:list[str]=[]
    offset=None
    while True:
        points,offset=quadrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True
        )
        for point in points:
            title=point.payload.get("title")
            if title and title not in seen:
                titles.append(title)
                seen.add(title)   
        if offset is None:
            break
    return titles


def search_papers(query:str,session_id:str,k:int=4)->list[Document]:
    return get_vector_store(session_id).similarity_search(query, k=k)

