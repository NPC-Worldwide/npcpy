

import requests
import os

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

try:
    from googlesearch import search
except:
    pass
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json

try:
    from sentence_transformers import util, SentenceTransformer
except:
    pass





def search_exa(query:str, 
               api_key:str = None, 
               top_k = 5,
               **kwargs):
    from exa_py import Exa
    if api_key is None:
        api_key = os.environ.get('EXA_API_KEY') 
    exa = Exa(api_key)

    results = exa.search_and_contents(
        query, 
        text=True   
    )
    return results.results[0:top_k]


def search_perplexity(
    query: str,
    api_key: str = None,
    model: str = "sonar",
    max_tokens: int = 400,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    if api_key is None:
        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if api_key is None: 
            raise 
        
    
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None,
    }

    
    headers = {"Authorization": f"Bearer {api_key}", 
               "Content-Type": "application/json"}

    response = requests.post(url,
                             json=payload,
                             headers=headers)

    if response.status_code != 200:
        raise Exception(f"Perplexity API error {response.status_code}: {response.text[:200]}")

    data = response.json()
    return [data["choices"][0]["message"]["content"], data.get("citations", [])]


def search_web(
    query: str,
    num_results: int = 5,
    provider: str=None,
    api_key=None,
    perplexity_kwargs: Optional[Dict[str, Any]] = None,
) -> List:
    """
    Function Description:
        This function searches the web for information based on a query.
    Args:
        query: The search query.
    Keyword Args:
        num_results: The number of search results to retrieve.
        provider: The search engine provider to use ('perplexity' or 'duckduckgo').
    Returns:
        A list of dictionaries with 'title', 'link', and 'content' keys.
    """
    if perplexity_kwargs is None:
        perplexity_kwargs = {}
    results = []
    if provider is None:
        provider = 'duckduckgo'

    if provider == "perplexity":
        search_result = search_perplexity(query, api_key=api_key, **perplexity_kwargs)
        
        return search_result

    if provider == "duckduckgo":
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
        }
        ddgs = DDGS(headers=headers)

        try:
            search_results = ddgs.text(query, max_results=num_results)
            urls = [r["href"] for r in search_results]
            results = [
                {"title": r["title"], "link": r["href"], "content": r["body"]}
                for r in search_results
            ]
        except DuckDuckGoSearchException as e:
            print("DuckDuckGo search failed: ", e)
            urls = []
            results = []
    elif provider =='exa':
        return search_exa(query, api_key=api_key, )

    elif provider =='google':  
        urls = list(search(query, num_results=num_results))
        
        
        
        for url in urls:
            try:
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()

                
                soup = BeautifulSoup(response.text, "html.parser")

                
                title = soup.title.string if soup.title else url

                
                content = " ".join([p.get_text() for p in soup.find_all("p")])
                content = " ".join(content.split())  

                results.append(
                    {
                        "title": title,
                        "link": url,
                        "content": (
                            content[:500] + "..." if len(content) > 500 else content
                        ),
                    }
                )

            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                continue

    
    
    content_str = "\n".join(
        [r["content"] + "\n Citation: " + r["link"] + "\n\n\n" for r in results]
    )
    link_str = "\n".join([r["link"] + "\n" for r in results])
    return [content_str, link_str]

