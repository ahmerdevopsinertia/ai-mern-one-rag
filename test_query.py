from rag_handler import get_rag_context
import json

def test_query(query):
    """Test the RAG system with a query and print the results."""
    print(f"\nQuery: '{query}'")
    result = get_rag_context(query)
    
    # Print formatted results
    print("\nResults:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # Example queries to test
    test_queries = [
        "What is the continuing education policy for staff?",
        "How are staff training needs assessed?",
        "Who is responsible for implementing the continuing education plan?",
    ]
    
    for query in test_queries:
        test_query(query)