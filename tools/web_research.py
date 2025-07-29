# -*- coding: utf-8 -*-
"""
Advanced Web Research Tools for AI Research Agent
Provides comprehensive web search, scraping, and content analysis capabilities
"""

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import wikipedia
import arxiv
from typing import List, Dict, Any, Optional
from langchain.tools import Tool
import re
import time
from urllib.parse import urljoin, urlparse
import json

class WebResearchTool:
    """Advanced web research capabilities"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web using DuckDuckGo"""
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', ''),
                        'source': 'web_search'
                    })
                
                return {
                    'query': query,
                    'results': results,
                    'total_found': len(results)
                }
        except Exception as e:
            return {
                'query': query,
                'results': [],
                'error': f"Web search failed: {str(e)}"
            }
    
    def scrape_webpage(self, url: str, extract_text: bool = True) -> Dict[str, Any]:
        """Scrape and analyze a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract metadata
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title"
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''
            
            # Extract main content
            content = ""
            if extract_text:
                # Try to find main content areas
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
                
                if main_content:
                    content = main_content.get_text(separator=' ', strip=True)
                else:
                    content = soup.get_text(separator=' ', strip=True)
                
                # Clean up content
                content = re.sub(r'\s+', ' ', content)
                content = content[:5000]  # Limit content length
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True)[:10]:  # Limit to 10 links
                href = urljoin(url, link['href'])
                link_text = link.get_text(strip=True)
                if link_text and len(link_text) > 3:
                    links.append({'url': href, 'text': link_text})
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'content': content,
                'links': links,
                'word_count': len(content.split()) if content else 0,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'url': url,
                'error': f"Failed to scrape webpage: {str(e)}",
                'status': 'error'
            }
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search Wikipedia for information"""
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=max_results)
            
            articles = []
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    summary = wikipedia.summary(title, sentences=3)
                    
                    articles.append({
                        'title': page.title,
                        'url': page.url,
                        'summary': summary,
                        'content': page.content[:2000],  # First 2000 chars
                        'categories': page.categories[:5],  # First 5 categories
                        'source': 'wikipedia'
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        summary = wikipedia.summary(e.options[0], sentences=3)
                        articles.append({
                            'title': page.title,
                            'url': page.url,
                            'summary': summary,
                            'content': page.content[:2000],
                            'categories': page.categories[:5],
                            'source': 'wikipedia'
                        })
                    except:
                        continue
                except:
                    continue
            
            return {
                'query': query,
                'articles': articles,
                'total_found': len(articles)
            }
            
        except Exception as e:
            return {
                'query': query,
                'articles': [],
                'error': f"Wikipedia search failed: {str(e)}"
            }
    
    def search_arxiv(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search arXiv for academic papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary,
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'categories': result.categories,
                    'source': 'arxiv'
                })
            
            return {
                'query': query,
                'papers': papers,
                'total_found': len(papers)
            }
            
        except Exception as e:
            return {
                'query': query,
                'papers': [],
                'error': f"arXiv search failed: {str(e)}"
            }

class NewsResearchTool:
    """News and current events research"""
    
    def __init__(self):
        self.session = requests.Session()
        
    def search_news(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for recent news articles"""
        try:
            with DDGS() as ddgs:
                results = []
                # Search for news specifically
                news_query = f"{query} news recent"
                
                for result in ddgs.text(news_query, max_results=max_results):
                    # Try to determine if it's a news article
                    url = result.get('href', '')
                    is_news = any(news_domain in url.lower() for news_domain in 
                                ['news', 'reuters', 'bbc', 'cnn', 'guardian', 'times', 'post'])
                    
                    results.append({
                        'title': result.get('title', ''),
                        'url': url,
                        'snippet': result.get('body', ''),
                        'is_likely_news': is_news,
                        'source': 'news_search'
                    })
                
                return {
                    'query': query,
                    'articles': results,
                    'total_found': len(results)
                }
                
        except Exception as e:
            return {
                'query': query,
                'articles': [],
                'error': f"News search failed: {str(e)}"
            }

def get_web_research_tools():
    """Get comprehensive web research tools"""
    
    web_tool = WebResearchTool()
    news_tool = NewsResearchTool()
    
    def web_search_tool(query: str) -> str:
        """Search the web for information"""
        try:
            results = web_tool.search_web(query, max_results=5)
            
            if 'error' in results:
                return f"Web search error: {results['error']}"
            
            formatted_output = f"Web search results for: {query}\n"
            formatted_output += "=" * 50 + "\n"
            
            for i, result in enumerate(results['results'], 1):
                formatted_output += f"{i}. {result['title']}\n"
                formatted_output += f"   URL: {result['url']}\n"
                formatted_output += f"   Summary: {result['snippet'][:200]}...\n\n"
            
            return formatted_output
            
        except Exception as e:
            return f"Web search failed: {str(e)}"
    
    def webpage_scraper_tool(url: str) -> str:
        """Scrape and analyze a specific webpage"""
        try:
            result = web_tool.scrape_webpage(url)
            
            if result['status'] == 'error':
                return f"Scraping error: {result['error']}"
            
            formatted_output = f"Webpage Analysis: {result['title']}\n"
            formatted_output += "=" * 50 + "\n"
            formatted_output += f"URL: {result['url']}\n"
            formatted_output += f"Description: {result['description']}\n"
            formatted_output += f"Word Count: {result['word_count']}\n\n"
            formatted_output += f"Content Preview:\n{result['content'][:1000]}...\n\n"
            
            if result['links']:
                formatted_output += "Related Links:\n"
                for link in result['links'][:5]:
                    formatted_output += f"- {link['text']}: {link['url']}\n"
            
            return formatted_output
            
        except Exception as e:
            return f"Webpage scraping failed: {str(e)}"
    
    def wikipedia_search_tool(query: str) -> str:
        """Search Wikipedia for comprehensive information"""
        try:
            results = web_tool.search_wikipedia(query)
            
            if 'error' in results:
                return f"Wikipedia search error: {results['error']}"
            
            formatted_output = f"Wikipedia search results for: {query}\n"
            formatted_output += "=" * 50 + "\n"
            
            for i, article in enumerate(results['articles'], 1):
                formatted_output += f"{i}. {article['title']}\n"
                formatted_output += f"   URL: {article['url']}\n"
                formatted_output += f"   Summary: {article['summary']}\n"
                formatted_output += f"   Categories: {', '.join(article['categories'][:3])}\n\n"
            
            return formatted_output
            
        except Exception as e:
            return f"Wikipedia search failed: {str(e)}"
    
    def arxiv_search_tool(query: str) -> str:
        """Search arXiv for academic papers"""
        try:
            results = web_tool.search_arxiv(query)
            
            if 'error' in results:
                return f"arXiv search error: {results['error']}"
            
            formatted_output = f"arXiv search results for: {query}\n"
            formatted_output += "=" * 50 + "\n"
            
            for i, paper in enumerate(results['papers'], 1):
                formatted_output += f"{i}. {paper['title']}\n"
                formatted_output += f"   Authors: {', '.join(paper['authors'][:3])}\n"
                formatted_output += f"   Published: {paper['published']}\n"
                formatted_output += f"   Categories: {', '.join(paper['categories'][:2])}\n"
                formatted_output += f"   URL: {paper['url']}\n"
                formatted_output += f"   Summary: {paper['summary'][:300]}...\n\n"
            
            return formatted_output
            
        except Exception as e:
            return f"arXiv search failed: {str(e)}"
    
    def news_search_tool(query: str) -> str:
        """Search for recent news and current events"""
        try:
            results = news_tool.search_news(query)
            
            if 'error' in results:
                return f"News search error: {results['error']}"
            
            formatted_output = f"News search results for: {query}\n"
            formatted_output += "=" * 50 + "\n"
            
            for i, article in enumerate(results['articles'], 1):
                news_indicator = "📰" if article['is_likely_news'] else "🔍"
                formatted_output += f"{i}. {news_indicator} {article['title']}\n"
                formatted_output += f"   URL: {article['url']}\n"
                formatted_output += f"   Summary: {article['snippet'][:200]}...\n\n"
            
            return formatted_output
            
        except Exception as e:
            return f"News search failed: {str(e)}"
    
    return [
        Tool(
            name="web_search",
            description="Search the web for general information using DuckDuckGo. Use for current information, facts, and general research.",
            func=web_search_tool
        ),
        Tool(
            name="scrape_webpage",
            description="Scrape and analyze a specific webpage. Input should be a valid URL. Use to get detailed content from specific pages.",
            func=webpage_scraper_tool
        ),
        Tool(
            name="wikipedia_search",
            description="Search Wikipedia for comprehensive, encyclopedic information. Best for background information, definitions, and established facts.",
            func=wikipedia_search_tool
        ),
        Tool(
            name="arxiv_search",
            description="Search arXiv for academic papers and research. Use for scientific research, technical papers, and academic sources.",
            func=arxiv_search_tool
        ),
        Tool(
            name="news_search",
            description="Search for recent news articles and current events. Use for latest developments, breaking news, and recent updates.",
            func=news_search_tool
        )
    ]