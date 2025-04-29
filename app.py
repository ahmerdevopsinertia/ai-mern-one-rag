import os
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from llama_cpp import Llama

class HybridAssistant:
    def __init__(self):
        # Your existing Pinecone setup
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index("edu-staff-policies-v1")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize Mistral
        model_path = os.path.expanduser("~/ai/fine-tuning/llama.cpp/models/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=1,      # Increase if you have GPU memory
            n_threads=10,        # M3 Pro has 10-12 cores
            n_batch=512,         # Faster inference
            verbose=False
        )
    
    def is_policy_lookup(self, query: str) -> bool:
        """Determine if query requires exact policy text"""
        policy_keywords = ['policy', 'section', 'document', 'procedure', 'rule']
        return any(keyword in query.lower() for keyword in policy_keywords)
    
    def rag_query(self, query: str) -> str:
        """Your existing Pinecone RAG logic"""
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(vector=query_embedding, top_k=3, include_metadata=True)
        context = "\n".join([match.metadata['text'] for match in results.matches])
        return context
    
    def llm_generate(self, prompt: str, context: str = None) -> str:
            """Generate with Mistral using enhanced prompt engineering"""
            enhanced_prompt = f"""You are an HR assistant for private schools. Use formal tone.
            
    Context: {context if context else 'No specific policy found'}
    
    Question: {prompt}
    
    Answer concisely in bullet points:"""
            
            response = self.llm.create_completion(
                enhanced_prompt,
                max_tokens=256,
                temperature=0.7,
                stop=["\n", "Question:"]
            )
            return response['choices'][0]['text']
    
    def respond(self, user_input: str) -> str:
        try:
          if self.is_policy_lookup(user_input):
                    context = self.rag_query(user_input)
                    if context:
                        return self.llm_generate(
                            prompt=user_input,
                            context=context  # Pass context to the enhanced prompt
                        )
                    else:
                        return "No relevant policies found."
          else:
								 		# For general questions without policy context
                    return self.llm_generate(user_input)
        except Exception as e:
                return f"Error: {str(e)}"

# Usage
assistant = HybridAssistant()
print(assistant.respond("What's the staff policy?"))  # Uses RAG
print(assistant.respond("How do I apply for leave?")) # Uses LLM

