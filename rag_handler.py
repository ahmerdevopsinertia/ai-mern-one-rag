import re
import os
import sys
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import hashlib
from redis import Redis

load_dotenv()  # Loads from .env file automatically

# Initialize connections ONCE
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("edu-staff-policies-v1")

embeddings = HuggingFaceEmbeddings (
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Add if you want to specify CPU/GPU
    encode_kwargs={'normalize_embeddings': True} # Recommended for cosine similarity
)

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\nSection", "\n•", "\n"]
)

def get_hyde_embedding(query):
    hypothetical = f"""Generate a policy snippet that would answer: {query}
    Example format: "Employees may accrue up to 1.25 PTO days per month [Sec 3.1]\""
    """
    return embeddings.embed_query(hypothetical)

def clean_context(context: str) -> str:

        """Remove duplicate sentences and limit length"""
        if not context:
            return context
    
        try:
            # Split into sentences
            sentences = [s.strip() for s in context.split('.') if s.strip()]
            
            # Remove duplicates (case insensitive)
            unique_sentences = []
            seen = set()
            for s in sentences:
                key = re.sub(r'[^a-zA-Z0-9]', '', s.lower())
                if key not in seen and len(key) > 15:  # Ignore short sentences
                    seen.add(key)
                    unique_sentences.append(s)
            
            # Join first 10 sentences (adjust as needed)
            return '. '.join(unique_sentences[:10]) + '.'  # Add final period
        except Exception as e:
            print(f"Context cleaning error: {str(e)}")
            return context  # Return original if cleaning fails

# More flexible citation check
def has_policy_citation(response):
    return any(
        re.search(pattern, response)
        for pattern in [
            r"\[Sec \d+\.\d+\]",  # [Sec 3.1]
            r"Section \d+",        # Section 3
            r"Policy [A-Z]+\d+"    # Policy ABC101
        ]
    )

def validate_response(response, context):	
    # Semantic similarity check
    resp_embed = embeddings.embed_query(response)
    ctx_embed = embeddings.embed_query(context)
    similarity = np.dot(resp_embed, ctx_embed)
    
    
		# Policy citation check
    has_citation = has_policy_citation(response)
    
    return {
        "valid": similarity > 0.7 and has_citation,
        "similarity": float(similarity),
        "missing_citations": 'disabled for now'
    }

def get_cache_key(query):
    query_key = hashlib.sha256(query.encode()).hexdigest()
    return f"hr_rag:{query_key}"

def get_rag_context(query):
  try:
        cache = Redis(host='localhost', port=6379, db=0)
        # cache.set("test", "Redis is running")
        # print(cache.get("test").decode())
        
        # if cached := cache.get(get_cache_key(query)):
        #     print("Cache hit")
        #     return json.loads(cached)

        # 1. Get relevant policy chunks
        original_embedding = embeddings.embed_query(query)
        hyde_embedding = get_hyde_embedding(query)
        blended_embedding = [
            0.7 * o + 0.3 * h  # Weighted average
            for o, h in zip(original_embedding, hyde_embedding)
        ]
        
        results = index.query(
            vector=blended_embedding,
            top_k=3,
            include_metadata=True
        )
        
				# Format and clean the context text
        context = "\n".join([
            f"- {m.metadata['text'].strip()}" 
            for m in results.matches
            if m.metadata.get('text')
        ])

        context = clean_context('\n'.join(m.metadata['text'] for m in results.matches))

        # 2. Format for LLM
        response = {
            "output": context,
            "context": context,
            "sources": list(set(
                m.metadata.get('source', 'policy') 
                for m in results.matches
            )),
            "cache_hit": False # Added for tracking
        }
  
        # Validation
        validation = validate_response(response["output"], context)
        if not validation["valid"]:
            response["output"] = "I cannot confidently answer based on available policies."
                  
        # Cache and return
        cache.set(get_cache_key(query), json.dumps(response), ex=86400)
        return {**response, "cache_hit": False, "validation": validation}
    
  except Exception as e:
        return {"error": str(e)}

def extract_section(chunk):
  match = re.search(r"(Section\s?\d+\.\d+)", chunk)
  return match.group(1) if match else "General"

if __name__ == "__main__":
    try:
        query = json.loads(sys.argv[1])
        result = get_rag_context(query)
        print(json.dumps(result))  # Ensure single JSON output
        sys.exit(0)  # Clean exit
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
        
