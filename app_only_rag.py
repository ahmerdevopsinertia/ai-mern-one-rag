import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# 1. Load and split documents
loader = PyPDFLoader("policies/staff_policy_v1.pdf")
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# 2. Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 3. Setup index (384-dim for all-MiniLM-L6-v2)
index_name = "edu-staff-policies-v1"
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)  # Clean up old index

pc.create_index(
    name=index_name,
    dimension=384,  # Critical - must match embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# 4. Create embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 384-dim

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Failed to load embeddings model: {str(e)}")
    raise

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Add if you want to specify CPU/GPU
    encode_kwargs={'normalize_embeddings': True}  # Recommended for cosine similarity
)

# 5. Upload to Pinecone
index = pc.Index(index_name)
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    vectors = [{
        "id": f"vec_{i+j}",
        "values": embeddings.embed_query(doc.page_content),
        "metadata": {"text": doc.page_content}
    } for j, doc in enumerate(batch)]
    index.upsert(vectors=vectors)

print(f"Uploaded {len(chunks)} chunks to {index_name}")

# 2. Query function
def query_pinecone(query: str, index_name: str, top_k: int = 3):
    index = pc.Index(index_name)
    query_embedding = embeddings.embed_query(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    return [match.metadata["text"] for match in results.matches]

# 3. Usage
answers = query_pinecone("What is the leave policy?", "edu-staff-policies-v1")
for answer in answers:
    print(answer)