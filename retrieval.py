import os
import pickle
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from dataset import datasets  # Ensure this is correctly set up

# Configuration
force_embed = False  # Set to True to re-embed documents
vector_store_path = "vector_store.pkl"  # Path to save/load the vector store
language_pairs = ['en-de', 'en-fr', 'en-pt']  # List of language pairs to process


# Define a custom embedding class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text)


# Initialize the embedding model
embedding_model = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')  # Free and lightweight


# Function to create documents from the dataset
def create_documents(language_pairs):
    documents = []
    for lang_pair in language_pairs:
        if lang_pair in datasets:
            train_df = datasets[lang_pair]['train']
            for _, row in train_df.iterrows():
                documents.append(
                    Document(
                        page_content=row['source'],
                        metadata={
                            "source_language": row['source_language'],
                            "target_language": row['target_language'],
                            "reference": row['reference'],
                            "doc_id": row['doc_id'],
                            "client_id": row['client_id'],
                            "sender": row['sender']
                        }
                    )
                )
    return documents


# Function to load or create the vector store
def get_vector_store():
    if not force_embed and os.path.exists(vector_store_path):
        # Load the vector store from disk
        with open(vector_store_path, "rb") as f:
            vector_store = pickle.load(f)
        print("Vector store loaded from disk.")
    else:
        # Create documents and embed them
        documents = create_documents(language_pairs)
        vector_store = FAISS.from_documents(documents, embedding_model)
        # Save the vector store to disk
        with open(vector_store_path, "wb") as f:
            pickle.dump(vector_store, f)
        print("Vector store created and saved to disk.")
    return vector_store


# Function to perform retrieval and remove duplicates
def retrieve_examples(vector_store, query, target_language_filter, top_k=3):
    # Retrieve more than needed to account for potential duplicates
    initial_results = vector_store.similarity_search(
        query,
        k=top_k * 3,  # Retrieve twice as many to ensure enough unique results
        filter={"target_language": target_language_filter}
    )

    unique_results = []
    seen_references = set()

    for result in initial_results:
        reference_translation = result.metadata['reference'].strip()
        if reference_translation not in seen_references:
            seen_references.add(reference_translation)
            unique_results.append(result)
        if len(unique_results) == top_k:
            break

    for i, result in enumerate(unique_results, 1):
        print(f"\nExample {i}:")
        print(f"Source Language: {result.metadata['source_language']}")
        print(f"Target Language: {result.metadata['target_language']}")
        print(f"Source Sentence: {result.page_content}")
        print(f"Reference Translation: {result.metadata['reference']}")
        print(f"Doc ID: {result.metadata['doc_id']}")
        print(f"Client ID: {result.metadata['client_id']}")
        print(f"Sender: {result.metadata['sender']}")


# Main function
def main():
    vector_store = get_vector_store()
    # Test retrieval with a sample English query and filter by target language
    sample_query = "how are you?"
    target_language_filter = "de"  # Specify the language code to filter by
    retrieve_examples(vector_store, sample_query, target_language_filter)


if __name__ == "__main__":
    main()