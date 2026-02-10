from npcpy.data.load import load_file_contents
from npcpy.data.web import search_web
from npcpy.memory.command_history import setup_chroma_db
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
    

def get_facts_for_rag(
    kuzu_db_path: str,
    chroma_db_path: str,
    query: str,
    group_filters: Optional[List[str]] = None,
    top_k: int = 10,
) -> str:
    """Get facts for RAG by combining vector and graph search

    Args:
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        query: Search query
        group_filters: Optional list of groups to filter by
        top_k: Number of results to return
        embedding_model: Model to use for embeddings
        provider: Provider for embeddings

    Returns:
        Formatted context string with retrieved facts
    """
    
    kuzu_conn = init_db(kuzu_db_path)
    chroma_client, chroma_collection = setup_chroma_db( 
        "knowledge_graph",
        "Facts extracted from various sources",
        chroma_db_path
    )

    
    results = hybrid_search_with_chroma(
        kuzu_conn=kuzu_conn,
        chroma_collection=chroma_collection,
        query=query,
        group_filter=group_filters,
        top_k=top_k,
    )

    
    context = "Related facts:\n\n"

    
    context += "Most relevant facts:\n"
    vector_matches = [r for r in results if r["source"] == "vector_search"]
    for i, item in enumerate(vector_matches):
        context += f"{i+1}. {item['fact']}\n"

    
    context += "\nRelated concepts:\n"
    graph_matches = [r for r in results if r["source"] != "vector_search"]
    for i, item in enumerate(graph_matches):
        group = item["source"].replace("graph_relation_via_", "")
        context += f"{i+1}. {item['fact']} (related via {group})\n"

    
    kuzu_conn.close()

    return context
def answer_with_rag(
    query: str,
    kuzu_db_path,
    chroma_db_path,
    model,
    provider,    
) -> str:
    """Answer a query using RAG with facts from the knowledge base

    Args:
        query: User query
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        model: LLM model to use
        provider: LLM provider
        embedding_model: Model to use for embeddings

    Returns:
        Answer from the model
    """
    
    context = get_facts_for_rag(
        kuzu_db_path,
        chroma_db_path,
        query,
    )

    
    prompt = f"""
    Answer this question based on the retrieved information.

    Question: {query}

    {context}

    Please provide a comprehensive answer based on the facts above. If the information
    doesn't contain a direct answer, please indicate that clearly but try to synthesize
    from the available facts.
    """

    
    response = get_llm_response(prompt, model=model, provider=provider)

    return response["response"]

    
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
        
        try:
            
            chroma_client, chroma_collection = setup_chroma_db( 
                f"{embedding_provider}_{embedding_model}_embeddings",
                "Conversation embeddings",
                vector_db_path
            )
            
            
            similar_texts = search_similar_texts(
                command, 
                embedding_model,
                embedding_provider,
                chroma_client=chroma_client,
                top_k=top_k,
            )
            
            
            separator = f"{YELLOW}{'-' * 100}{RESET}\n"
            
            
            processed_texts = []
            for i, similar_text in enumerate(similar_texts, 1):
                text = similar_text['text']
                score = similar_text['score']
                
                
                timestamp_str = ""
                try:
                    if 'id' in similar_text and '_' in similar_text['id']:
                        timestamp = datetime.fromisoformat(similar_text['id'].split('_')[1])
                        timestamp_str = f"{YELLOW}({timestamp.strftime('%Y-%m-%d')}){RESET}"
                except (IndexError, ValueError, TypeError):
                    pass
                
                
                text = text.replace('\n', ' ').strip()
                snippet = text[:85] + ("..." if len(text) > 85 else "")
                
                
                processed_texts.append(
                    f"{BOLD}{i:2d}{RESET}. {CYAN}[{score:.2f}]{RESET} {snippet} {timestamp_str}"
                )
            
            
            knowledge_results = header + separator + "\n".join(processed_texts)
            render_markdown(f"KNOWLEDGE BASE: {knowledge_results}")
            
            
            plain_texts = [f"{i+1}. {similar_texts[i]['text']}" for i in range(len(similar_texts))]
            plain_results = "\n\n".join(plain_texts)
            
            
            prompt = f"""
            The user asked: {command}
            
            Here are the most similar texts found in the knowledge base:
            
            {plain_results}
            
            Please respond to the user query based on the above information, integrating the information in an additive way, attempting to always find some possible connection
            between the results and the initial input. do not do this haphazardly, be creative yet cautious.
            """
            
            
            response = get_llm_response(
                prompt,
                **kwargs
            )
            return response
            
        except Exception as e:
            traceback.print_exc()
            return {"output": f"Error searching knowledge base: {e}", "messages": kwargs.get('messages', [])}
        
        
def execute_brainblast_command(
    command: str,
    **kwargs
) -> dict:
    """
    Execute a comprehensive "brainblast" search on command history.
    Breaks the query into words and searches for combinations of those words.
    """
    
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    

    if not 'command_history' in kwargs:
        raise Exception('Command history must be passed as a kwarg to this function')
    command_history = kwargs.get('command_history')
    top_k = kwargs.get('top_k', 10)
    
    
    
    header = f"\n{BOLD}{BLUE}BRAINBLAST Query: {RESET}{GREEN}{command}{RESET}\n"
    separator = f"{YELLOW}{'-' * 100}{RESET}\n"
    
    try:
        
        words = command.split()
        
        if not words:
            return {"output": "Please provide search terms to use brainblast.", "messages": messages or []}
        
        
        all_chunks = []
        
        
        all_chunks.extend(words)
        
        
        if len(words) >= 2:
            for i in range(len(words) - 1):
                all_chunks.append(f"{words[i]} {words[i+1]}")
        
        
        if len(words) >= 4:
            for i in range(len(words) - 3):
                all_chunks.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}")
        
        
        if len(words) > 1:
            all_chunks.append(command)
        
        
        unique_chunks = []
        for chunk in all_chunks:
            if chunk not in unique_chunks:
                unique_chunks.append(chunk)
        
        
        all_results = []
        chunk_results = {}
        
        for chunk in unique_chunks:
            results = command_history.search_conversations(chunk)
            if results:
                chunk_results[chunk] = results[:top_k]  
                all_results.extend(results[:top_k])
        
        
        unique_results = []
        seen_ids = set()
        for result in all_results:
            result_id = result.get('id')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        if not unique_results:
            result_message = f"No matches found for any combination of terms in: {command}"
            render_markdown(f"BRAINBLAST SEARCH: {header}{separator}{result_message}")
            
            
            prompt = f"""
            The user asked for a brainblast search with: {command}
            
            No matching commands were found in the command history for any combination 
            of these search terms. Please let the user know and suggest they try different terms.
            """
            
            response = get_llm_response(
                prompt,
                messages=kwargs.get('messages'),
                **kwargs
            )
            return {'output':response.get('response'), 'messages':response.get('messages') or []}
        
        
        processed_chunks = []
        for chunk, results in chunk_results.items():
            if results:
                chunk_display = f"{BOLD}{BLUE}Results for '{chunk}':{RESET}\n"
                
                for i, result in enumerate(results[:3], 1):  
                    cmd = result.get('content', '')
                    timestamp = result.get('timestamp', '')
                    
                    chunk_display += f"  {i}. {CYAN}[{timestamp}]{RESET} {cmd}\n"
                
                if len(results) > 3:
                    chunk_display += f"  {YELLOW}...and {len(results) - 3} more results{RESET}\n"
                
                processed_chunks.append(chunk_display)
        
        
        
        plain_results = []
        for i, result in enumerate(unique_results[:15], 1):  
            content = result.get('content', '')[0:250]
            timestamp = result.get('timestamp', '')
            location = result.get('directory_path', '')
            
            
            plain_results.append(
                f"{i}. [{timestamp}] Command: {cmd}\n   Location: {location}\n   Output: {content[:150] + ('...' if len(content) > 150 else '')}"
            )
        
        
        term_summary = []
        for chunk, results in chunk_results.items():
            if results:
                term_summary.append(f"Term '{chunk}' matched {len(results)} commands")
        
        
        f=', '.join(term_summary)
        e="\n\n".join(plain_results)
        prompt = f"""
        The user asked for a brainblast search with: {command}
        
        I searched for individual words, pairs of words, groups of 4 words, and the full query.
        Here's a summary of what matched:
        {f}
        
        Here are the top unique matching items from the search:
        
        {e}
        
        Please analyze these results and attempt to generate some novel insight about them in one sentence. think outside the box.
        Provide a summary as well.        
        """
        
        response = get_llm_response(
            prompt,
            **kwargs, 
        )
        return  {"output": response.get('response'), 
                 "messages": response.get('messages',[]) }
        
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing brainblast command: {e}", "messages": kwargs.get('messages',[]) }