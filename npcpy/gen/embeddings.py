

from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

try:
    from openai import OpenAI
    import anthropic
except: 
    pass

def get_ollama_embeddings(
    texts: List[str], model: str = "nomic-embed-text"
) -> List[List[float]]:
    """Generate embeddings using Ollama."""
    import ollama

    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings

def get_openai_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Generate embeddings using OpenAI."""
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    return [embedding.embedding for embedding in response.data]

def store_embeddings_for_model(
    texts,
    embeddings,
    chroma_client,
    model,
    provider,
    metadata=None,
):
    collection_name = f"{provider}_{model}_embeddings"
    collection = chroma_client.get_collection(collection_name)

    
    if metadata is None:
        metadata = [{"text_length": len(text)} for text in texts]  
        print(
            "metadata is none, creating metadata for each document as the length of the text"
        )
    
    collection.add(
        ids=[str(i) for i in range(len(texts))],
        embeddings=embeddings,
        metadatas=metadata,  
        documents=texts,
    )

def delete_embeddings_from_collection(collection, ids):
    """Delete embeddings by id from Chroma collection."""
    if ids:
        collection.delete(ids=ids)  

def get_embeddings(
    texts: List[str],
    model: str ,
    provider: str,
) -> List[List[float]]:
    """Generate embeddings using the specified provider and store them in Chroma."""
    if provider == "ollama":
        embeddings = get_ollama_embeddings(texts, model)
    elif provider == "openai":
        embeddings = get_openai_embeddings(texts, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    
    
    return embeddings
