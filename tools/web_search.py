# -*- coding: utf-8 -*-
"""
Web search tools for research agent
Provides web search capabilities for gathering external information
"""

from langchain.tools import Tool
import requests
from typing import List, Dict, Any
import json

class WebSearchTool:
    """Simple web search tool using DuckDuckGo API"""
    
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str, max_results: int = 5) -> str:
        """Search the web for information"""
        try:
            # Use DuckDuckGo instant answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            results = []
            
            # Get abstract if available
            if data.get('Abstract'):
                results.append(f"Summary: {data['Abstract']}")
            
            # Get definition if available
            if data.get('Definition'):
                results.append(f"Definition: {data['Definition']}")
            
            # Get related topics
            if data.get('RelatedTopics'):
                topics = []
                for topic in data['RelatedTopics'][:3]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        topics.append(topic['Text'])
                if topics:
                    results.append(f"Related: {'; '.join(topics)}")
            
            # Get answer if available
            if data.get('Answer'):
                results.append(f"Answer: {data['Answer']}")
            
            if results:
                return f"Web search results for '{query}':\n" + "\n".join(results)
            else:
                return f"No detailed results found for '{query}'. Try rephrasing your search."
                
        except Exception as e:
            return f"Web search error: {str(e)}. Try a different search term."

def get_web_search_tool():
    """Get web search tool for research agent"""
    searcher = WebSearchTool()
    
    return Tool(
        name="web_search",
        description="Search the web for current information, definitions, and facts. Use this when you need external information not in memory.",
        func=searcher.search
    )

# Alternative simple search function for testing
def simple_web_search(query: str) -> str:
    """Simple fallback search that returns a structured response"""
    return f"""
Web search simulation for: {query}

This is a placeholder response. In a production environment, this would:
1. Query real search APIs (Google, Bing, DuckDuckGo)
2. Extract relevant information from results
3. Summarize findings for the research agent

For now, this confirms the search functionality is integrated.
Search query processed: {query}
"""

def get_simple_search_tool():
    """Get a simple search tool for testing"""
    return Tool(
        name="web_search",
        description="Search the web for information (test mode)",
        func=simple_web_search
    )