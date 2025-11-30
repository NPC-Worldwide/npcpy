

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
    
    response = response.json()

    return [response["choices"][0]["message"]["content"], response["citations"]]


def _search_wikipedia(query: str, num_results: int = 5) -> List[Dict]:
    """Fallback search using Wikipedia API."""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
            "utf8": 1,
        }
        headers = {
            "User-Agent": "npcsh/1.0 (https://github.com/npcpy; npcsh@example.com) python-requests"
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            results.append({"title": title, "link": link, "content": snippet})
        return results
    except:
        return []


def _search_searxng(query: str, num_results: int = 5) -> List[Dict]:
    """Fallback search using public SearXNG instances."""
    # Public SearXNG instances
    instances = [
        "https://searx.be",
        "https://search.sapti.me",
        "https://searx.tiekoetter.com",
        "https://search.bus-hit.me",
    ]
    import random
    random.shuffle(instances)

    for instance in instances[:2]:  # Try up to 2 instances
        try:
            url = f"{instance}/search"
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            results = []
            for item in data.get("results", [])[:num_results]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "content": item.get("content", ""),
                })
            if results:
                return results
        except:
            continue
    return []


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
        provider: The search engine provider to use ('perplexity', 'duckduckgo', 'wikipedia', 'searxng', 'exa').
    Returns:
        A list with [content_string, links_string].
    """
    import time
    import random

    if perplexity_kwargs is None:
        perplexity_kwargs = {}
    results = []
    if provider is None:
        provider = 'duckduckgo'

    if provider == "perplexity":
        search_result = search_perplexity(query, api_key=api_key, **perplexity_kwargs)
        return search_result

    if provider == "wikipedia":
        results = _search_wikipedia(query, num_results)

    elif provider == "searxng":
        results = _search_searxng(query, num_results)

    elif provider == "duckduckgo":
        # Rotate user agents and use different DDG backends to avoid rate limiting
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
        ]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {"User-Agent": random.choice(user_agents)}

                # Add small random delay to avoid detection
                if attempt > 0:
                    time.sleep(random.uniform(1.0, 3.0))

                ddgs = DDGS(headers=headers)
                search_results = list(ddgs.text(query, max_results=num_results))

                if search_results:
                    results = [
                        {"title": r["title"], "link": r["href"], "content": r["body"]}
                        for r in search_results
                    ]
                    break  # Success, exit retry loop

            except DuckDuckGoSearchException as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(1.0, 3.0)
                    time.sleep(wait_time)
                    continue
                else:
                    # DDG failed, try fallbacks
                    results = _search_searxng(query, num_results)
                    if not results:
                        results = _search_wikipedia(query, num_results)
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1.0, 2.0))
                    continue
                # Try fallbacks
                results = _search_searxng(query, num_results)
                if not results:
                    results = _search_wikipedia(query, num_results)
                break

    elif provider == 'exa':
        return search_exa(query, api_key=api_key)

    elif provider == 'google':
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

    if not results:
        return ["No search results found.", ""]

    content_str = "\n".join(
        [r["content"] + "\n Citation: " + r["link"] + "\n\n\n" for r in results]
    )
    link_str = "\n".join([r["link"] + "\n" for r in results])
    return [content_str, link_str]

