import re
import os
import sys
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()  # Loads from .env file automatically

# Initialize connections ONCE
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("edu-staff-policies-v1")
embeddings = HuggingFaceEmbeddings (
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Add if you want to specify CPU/GPU
    encode_kwargs={'normalize_embeddings': True} # Recommended for cosine similarity
)

def clean_context(context: str) -> str:
    # """Remove page artifacts and duplicates"""
    # if not text:
    #     return text
        
    # # Remove page numbers and headers
    # text = re.sub(r'Unit:.*?Edition #:.*?\n', '', text)
    # text = re.sub(r'… continued on following page', '', text)
    
    # # Remove duplicate lines
    # lines = [line.strip() for line in text.split('\n') if line.strip()]
    # seen = set()
    # unique_lines = []
    # for line in lines:
    #     key = re.sub(r'[^a-zA-Z0-9]', '', line.lower())
    #     if key not in seen:
    #         seen.add(key)
    #         unique_lines.append(line)
    
    # return '\n'.join(unique_lines[:15])  # Limit to 15 most relevant lines
    
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

def get_rag_context(query):
    try:
        # 1. Get relevant policy chunks
        query_embedding = embeddings.embed_query(query)
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True
        )
        
                # Clean the context text
        context = "\n".join([
            f"- {m.metadata['text'].strip()}" 
            for m in results.matches
            if m.metadata.get('text')
        ])

        context = clean_context('\n'.join(m.metadata['text'] for m in results.matches))

        # 2. Format for LLM
        return {
            "context": context,
            "sources": list(set(
                m.metadata.get('source', 'policy') 
                for m in results.matches
            ))
        }
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        query = json.loads(sys.argv[1])
        result = get_rag_context(query)
        print(json.dumps(result))  # Ensure single JSON output
        sys.exit(0)  # Clean exit
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)