from npcpy.data.load import load_file_contents
from npcpy.data.web import search_web
from npcpy.gen.embeddings import get_ollama_embeddings
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_sysenv import render_markdown
from typing import Any, Dict, List, Optional, Union 
import numpy as np
from datetime import datetime
import traceback
try:
    import chromadb
except ImportError:
    chromadb = None
except Exception as e: 
    print(f"An error occurred: {e}")
    chromadb = None
    

def search_similar_texts(
    query: str,
    embedding_model: str,
    embedding_provider: str,    
    chroma_client = None,
    
    docs_to_embed: Optional[List[str]] = None,
    top_k: int = 15,
) -> List[Dict[str, any]]:
    """
    Search for similar texts using either a Chroma database or direct embedding comparison.
    With duplicate filtering.
    """

    print(f"\nQuery to embed: {query}")
    embedded_search_term = get_ollama_embeddings([query], embedding_model)[0]

    if docs_to_embed is None:
        
        collection_name = f"{embedding_provider}_{embedding_model}_embeddings"
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[embedded_search_term], n_results=top_k * 2  
        )
        
        
        seen_texts = set()
        filtered_results = []
        
        for idx, (id, distance, document) in enumerate(zip(
            results["ids"][0], results["distances"][0], results["documents"][0]
        )):
            
            if document not in seen_texts:
                seen_texts.add(document)
                filtered_results.append({
                    "id": id, 
                    "score": float(distance), 
                    "text": document
                })
                
                
                if len(filtered_results) >= top_k:
                    break
                    
        return filtered_results

    print(f"\nNumber of documents to embed: {len(docs_to_embed)}")

    
    unique_docs = list(dict.fromkeys(docs_to_embed))  
    raw_embeddings = get_ollama_embeddings(unique_docs, embedding_model)

    output_embeddings = []
    unique_doc_indices = []
    
    for idx, emb in enumerate(raw_embeddings):
        if emb:  
            output_embeddings.append(emb)
            unique_doc_indices.append(idx)

    
    doc_embeddings = np.array(output_embeddings)
    query_embedding = np.array(embedded_search_term)

    
    if len(doc_embeddings) == 0:
        raise ValueError("No valid document embeddings found")

    
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_embedding)

    
    if query_norm == 0:
        raise ValueError("Query embedding is zero-length")

    
    cosine_similarities = np.dot(doc_embeddings, query_embedding) / (
        doc_norms.flatten() * query_norm
    )

    
    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]

    return [
        {
            "id": str(unique_doc_indices[idx]),
            "score": float(cosine_similarities[idx]),
            "text": unique_docs[unique_doc_indices[idx]],
        }
        for idx in top_indices
    ]
def execute_search_command(
    command: str,
    messages=None,
    provider: str = None,
):
    """
    Function Description:

    Args:
        command : str : Command
        db_path : str : Database path

    Keyword Args:
        embedding_model : None : Embedding model
        current_npc : None : Current NPC
        text_data : None : Text data
        text_data_embedded : None : Embedded text data
        messages : None : Messages
    Returns:
        dict : dict : Dictionary

    """

    search_command = command.split()
    if any("-p" in s for s in search_command) or any(
        "--provider" in s for s in search_command
    ):
        provider = (
            search_command[search_command.index("-p") + 1]
            if "-p" in search_command
            else search_command[search_command.index("--provider") + 1]
        )
    else:
        provider = None
    if any("-n" in s for s in search_command) or any(
        "--num_results" in s for s in search_command
    ):
        num_results = (
            search_command[search_command.index("-n") + 1]
            if "-n" in search_command
            else search_command[search_command.index("--num_results") + 1]
        )
    else:
        num_results = 5

    
    command = command.replace(f"-p {provider}", "").replace(
        f"--provider {provider}", ""
    )
    result = search_web(command, num_results=num_results, provider=provider)
    if messages is None:
        messages = []
        messages.append({"role": "user", "content": command})

    messages.append(
        {"role": "assistant", "content": result[0] + f" \n Citation Links: {result[1]}"}
    )

    return {
        "messages": messages,
        "output": result[0] + f"\n\n\n Citation Links: {result[1]}",
    }
    

def execute_rag_command(
    command: str,
    vector_db_path: str, 
    embedding_model: str,
    embedding_provider: str,
    top_k: int = 15,
    file_contents=None,  
    **kwargs
) -> dict:
    """
    Execute the RAG command with support for embedding generation.
    When file_contents is provided, it searches those instead of the database.
    """
    
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    
    header = f"\n{BOLD}{BLUE}RAG Query: {RESET}{GREEN}{command}{RESET}\n"
    
    
    if file_contents and len(file_contents) > 0:
        similar_chunks = search_similar_texts(
            command,
            embedding_model,
            embedding_provider,
            chroma_client=None,  
            
            docs_to_embed=file_contents,  
            top_k=top_k
        )
        
        
        file_info = f"{BOLD}{BLUE}Files Processed: {RESET}{YELLOW}{len(file_contents)}{RESET}\n"
        separator = f"{YELLOW}{'-' * 100}{RESET}\n"
        
        
        chunk_results = []
        for i, chunk in enumerate(similar_chunks, 1):
            score = chunk['score']
            text = chunk['text']
            
            
            display_text = text[:150] + ("..." if len(text) > 150 else "")
            chunk_results.append(f"{BOLD}{i:2d}{RESET}. {CYAN}[{score:.2f}]{RESET} {display_text}")
        
        
        file_results = header + file_info + separator + "\n".join(chunk_results)
        render_markdown(f"FILE SEARCH RESULTS:\n{file_results}")
        
        
        plain_chunks = [f"{i+1}. {chunk['text']}" for i, chunk in enumerate(similar_chunks)]
        plain_results = "\n\n".join(plain_chunks)
        
        
        prompt = f"""
        The user asked: {command}
        
        Here are the most relevant sections from the file(s):
        
        {plain_results}
        
        Please respond to the user query based on the above information, integrating the information in an additive way, attempting to always find some possible connection
        between the results and the initial input. do not do this haphazardly, be creative yet cautious.
        """
        
        
        response = get_llm_response(
            prompt,
            **kwargs
        )
        return response
    else:
        return {"output": "RAG without file_contents requires a vector store backend. Provide file_contents or use the application-layer RAG command.", "messages": kwargs.get('messages', [])}
        
