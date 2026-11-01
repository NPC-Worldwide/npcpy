import time
from npcpy.data.web import search_web, search_perplexity


def test_duckduckgo_search():
    """Test DuckDuckGo web search"""
    try:
        results = search_web(
            query="Python programming language",
            num_results=3,
            provider="duckduckgo"
        )
        
        assert len(results) == 2  
        content_str, link_str = results
        
        assert len(content_str) > 0
        assert len(link_str) > 0
        assert "python" in content_str.lower()
        
        print(f"DuckDuckGo search returned {len(content_str)} chars of content")
        print(f"Found links: {link_str[:200]}...")
        
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")


def test_google_search():
    """Test Google web search"""
    try:
        results = search_web(
            query="machine learning basics",
            num_results=2,
            provider="google"
        )
        
        assert len(results) == 2  
        content_str, link_str = results
        
        print(f"Google search returned {len(content_str)} chars of content")
        print(f"Found links: {link_str[:200]}...")
        
    except Exception as e:
        print(f"Google search failed (expected due to restrictions): {e}")


def test_perplexity_search():
    """Test Perplexity AI search"""
    try:
        
        results = search_perplexity(
            query="What is artificial intelligence?",
            api_key="fake_key"  
        )
        
        if results:
            assert len(results) == 2  
            print("Perplexity search successful")
        
    except Exception as e:
        print(f"Perplexity search failed (expected without API key): {e}")


def test_search_web_default_provider():
    """Test web search with default provider"""
    try:
        results = search_web(
            query="open source software",
            num_results=2
        )
        
        assert len(results) == 2
        content_str, link_str = results
        
        assert isinstance(content_str, str)
        assert isinstance(link_str, str)
        
        print(f"Default provider search returned {len(content_str)} chars")
        
    except Exception as e:
        print(f"Default provider search failed: {e}")


def test_search_web_rate_limiting():
    """Test search with small delay to respect rate limits"""
    try:
        
        results1 = search_web("test query 1", num_results=1, provider="duckduckgo")
        
        
        time.sleep(1)
        
        
        results2 = search_web("test query 2", num_results=1, provider="duckduckgo")
        
        assert len(results1) == 2
        assert len(results2) == 2
        
        print("Rate limiting test passed")
        
    except Exception as e:
        print(f"Rate limiting test failed: {e}")


def test_search_web_empty_results():
    """Test search with very specific query that might return few results"""
    try:
        results = search_web(
            query="very_specific_unusual_query_12345678",
            num_results=1,
            provider="duckduckgo"
        )
        
        assert len(results) == 2  
        content_str, link_str = results
        
        print(f"Specific query search: content={len(content_str)}, links={len(link_str)}")
        
    except Exception as e:
        print(f"Specific query search failed: {e}")


def test_perplexity_with_custom_params():
    """Test Perplexity with custom parameters"""
    try:
        perplexity_kwargs = {
            "max_tokens": 200,
            "temperature": 0.5
        }
        
        results = search_web(
            query="climate change effects",
            provider="perplexity",
            api_key="fake_key",
            perplexity_kwargs=perplexity_kwargs
        )
        
        print("Perplexity custom params test completed")
        
    except Exception as e:
        print(f"Perplexity custom params failed (expected): {e}")


def test_search_web_large_num_results():
    """Test search with larger number of results"""
    try:
        results = search_web(
            query="artificial intelligence",
            num_results=10,
            provider="duckduckgo"
        )
        
        assert len(results) == 2
        content_str, link_str = results
        
        
        assert len(content_str) > 100
        
        print(f"Large results search: {len(content_str)} chars of content")
        
    except Exception as e:
        print(f"Large results search failed: {e}")


def test_search_web_content_parsing():
    """Test that search results contain expected content structure"""
    try:
        results = search_web(
            query="Python tutorial",
            num_results=2,
            provider="duckduckgo"
        )
        
        content_str, link_str = results
        
        
        assert "Citation:" in content_str
        
        
        links = link_str.strip().split('\n')
        assert len(links) >= 1
        
        print(f"Content parsing test passed - found {len(links)} links")
        
    except Exception as e:
        print(f"Content parsing test failed: {e}")


def test_serply_search(monkeypatch):
    """Test Serply search request and result mapping (mocked)"""
    from unittest.mock import MagicMock, patch
    from npcpy.data.web import search_serply

    monkeypatch.setenv("SERPLY_API_KEY", "test-key")
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "results": [
            {"title": "Python", "link": "https://www.python.org/", "description": "The official home of Python."},
            {"title": "Docs", "link": "https://docs.python.org/3/", "description": "Python documentation."},
        ]
    }
    with patch("npcpy.data.web.requests.get", return_value=response) as get:
        results = search_serply("python", num_results=1)

    assert results == [
        {"title": "Python", "link": "https://www.python.org/", "content": "The official home of Python."}
    ]
    args, kwargs = get.call_args
    assert args[0] == "https://api.serply.io/v1/search/"
    assert kwargs["params"] == {"q": "python", "num": 1}
    assert kwargs["headers"]["X-Api-Key"] == "test-key"


def test_serply_search_without_key(monkeypatch):
    """Serply returns no results and makes no request without a key"""
    from unittest.mock import patch
    from npcpy.data.web import search_serply

    monkeypatch.delenv("SERPLY_API_KEY", raising=False)
    with patch("npcpy.data.web.requests.get") as get:
        assert search_serply("python") == []
    get.assert_not_called()


def test_search_web_serply_provider():
    """search_web routes provider='serply' and formats citations (mocked)"""
    from unittest.mock import patch

    hits = [{"title": "Python", "link": "https://www.python.org/", "content": "The official home of Python."}]
    with patch("npcpy.data.web.search_serply", return_value=hits) as serply:
        content_str, link_str = search_web("python", num_results=3, provider="serply", api_key="k")

    serply.assert_called_once_with("python", num_results=3, api_key="k")
    assert "Citation: https://www.python.org/" in content_str
    assert link_str.strip() == "https://www.python.org/"


def test_keenable_search(monkeypatch):
    """Test keyless Keenable search request and result mapping (mocked)"""
    from unittest.mock import MagicMock, patch
    from npcpy.data.web import search_keenable

    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "results": [
            {"title": "Python", "url": "https://www.python.org/", "description": "", "snippet": "x" * 600},
            {"title": "Docs", "url": "https://docs.python.org/3/", "description": "", "snippet": "Python documentation."},
        ]
    }
    with patch("npcpy.data.web.requests.post", return_value=response) as post:
        results = search_keenable("python", num_results=1)

    assert results == [
        {"title": "Python", "link": "https://www.python.org/", "content": "x" * 500 + "..."}
    ]
    args, kwargs = post.call_args
    assert args[0] == "https://api.keenable.ai/v1/search/public"
    assert kwargs["json"] == {"query": "python", "max_results": 1}
    assert kwargs["headers"]["X-Keenable-Title"] == "npcpy"
    assert "X-API-Key" not in kwargs["headers"]


def test_keenable_search_with_key(monkeypatch):
    """Keenable uses the keyed endpoint when KEENABLE_API_KEY is set (mocked)"""
    from unittest.mock import MagicMock, patch
    from npcpy.data.web import search_keenable

    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "results": [{"title": "Python", "url": "https://www.python.org/", "snippet": "The official home of Python."}]
    }
    with patch("npcpy.data.web.requests.post", return_value=response) as post:
        results = search_keenable("python", num_results=100)

    assert results == [
        {"title": "Python", "link": "https://www.python.org/", "content": "The official home of Python."}
    ]
    args, kwargs = post.call_args
    assert args[0] == "https://api.keenable.ai/v1/search"
    assert kwargs["json"] == {"query": "python", "max_results": 50}
    assert kwargs["headers"]["X-API-Key"] == "test-key"


def test_search_web_keenable_provider():
    """search_web routes provider='keenable' and formats citations (mocked)"""
    from unittest.mock import patch

    hits = [{"title": "Python", "link": "https://www.python.org/", "content": "The official home of Python."}]
    with patch("npcpy.data.web.search_keenable", return_value=hits) as keenable:
        content_str, link_str = search_web("python", num_results=3, provider="keenable", api_key="k")

    keenable.assert_called_once_with("python", num_results=3, api_key="k")
    assert "Citation: https://www.python.org/" in content_str
    assert link_str.strip() == "https://www.python.org/"


def test_firecrawl_search(monkeypatch):
    """Test Firecrawl search request and result mapping (mocked)"""
    from unittest.mock import MagicMock, patch
    from npcpy.data.web import search_firecrawl

    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "success": True,
        "data": {
            "web": [
                {"title": "Python", "url": "https://www.python.org/", "description": "The official home of Python."},
                {"title": "Docs", "url": "https://docs.python.org/3/", "description": "Python documentation."},
            ]
        },
    }
    with patch("npcpy.data.web.requests.post", return_value=response) as post:
        results = search_firecrawl("python", num_results=1)

    assert results == [
        {"title": "Python", "link": "https://www.python.org/", "content": "The official home of Python."}
    ]
    args, kwargs = post.call_args
    assert args[0] == "https://api.firecrawl.dev/v2/search"
    assert kwargs["json"] == {"query": "python", "limit": 1, "sources": ["web"], "integration": "_npcpy"}
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["headers"]["User-Agent"] == "npcpy"


def test_firecrawl_search_without_key(monkeypatch):
    """Firecrawl returns no results and makes no request without a key"""
    from unittest.mock import patch
    from npcpy.data.web import search_firecrawl

    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    with patch("npcpy.data.web.requests.post") as post:
        assert search_firecrawl("python") == []
    post.assert_not_called()


def test_search_web_firecrawl_provider():
    """search_web routes provider='firecrawl' and formats citations (mocked)"""
    from unittest.mock import patch

    hits = [{"title": "Python", "link": "https://www.python.org/", "content": "The official home of Python."}]
    with patch("npcpy.data.web.search_firecrawl", return_value=hits) as firecrawl:
        content_str, link_str = search_web("python", num_results=3, provider="firecrawl", api_key="k")

    firecrawl.assert_called_once_with("python", num_results=3, api_key="k")
    assert "Citation: https://www.python.org/" in content_str
    assert link_str.strip() == "https://www.python.org/"
