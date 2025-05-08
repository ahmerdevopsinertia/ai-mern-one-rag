import os
import re
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def extract_section(chunk):
    # Extract section headers like "Section 3.1"
    match = re.search(r"(Section\s?\d+\.\d+)", chunk)
    return match.group(1) if match else "General"

# 1. Load and split documents
loader = PyPDFLoader("policies/staff_policy_v1.pdf")
source = "policies/staff_policy_v1.pdf"
pages = loader.load_and_split()

# 2. Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "edu-staff-policies-v1"

# 3. Setup index (384-dim for all-MiniLM-L6-v2)
# if index_name in pc.list_indexes().names():
#     pc.delete_index(index_name)  # Clean up old index

pc.create_index(
    name=index_name,
    dimension=384,  # Critical - must match embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

if index_name in pc.list_indexes().names():
      print(f"Index {index_name} created successfully.")
else:
    print(f"Failed to create index {index_name}.")

# moved to separate file
# try:
#     embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={'device': 'cpu'},  # Add if you want to specify CPU/GPU
#     encode_kwargs={'normalize_embeddings': True}  # Recommended for cosine similarity
# )
    
#     """Process policy documents and upload to Pinecone"""
#     with open(source) as f:
#         text = f.read()

#     chunks = [
#         {
#             "id": f"{source}-{i}",
#             "values": embeddings.embed_documents([chunk])[0],
#             "metadata": {
#                 "text": chunk,
#                 "source": source,
#                 "section": extract_section(chunk)
#             }
#         } 
#         for i, chunk in enumerate(text_splitter.split_text(text))
#     ]
# except Exception as e:
#     print(f"Failed to load embeddings model: {str(e)}")
#     raise

# # 5. Upload to Pinecone
# index = pc.Index(index_name)
# for i in range(0, len(chunks), 100):
#     index.upsert(vectors=chunks[i:i+100])
#     # batch = chunks[i:i+100]
#     # vectors = [{
#     #     "id": f"vec_{i+j}",
#     #     "values": embeddings.embed_query(doc.page_content),
#     #     "metadata": {"text": doc.page_content}
#     # } for j, doc in enumerate(batch)]
#     # index.upsert(vectors=vectors)

# print(f"Uploaded {len(chunks)} chunks to {index_name}")

# # 2. Query function
# def query_pinecone(query: str, index_name: str, top_k: int = 3):
#     index = pc.Index(index_name)
#     query_embedding = embeddings.embed_query(query)
    
#     results = index.query(
#         vector=query_embedding,
#         top_k=top_k,
#         include_metadata=True
#     )
    
#     return [match.metadata["text"] for match in results.matches]

# # 3. Usage
# answers = query_pinecone("HR policy", "edu-staff-policies-v1")
# for answer in answers:
#     print(answer)