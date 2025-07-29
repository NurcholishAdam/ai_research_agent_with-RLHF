# -*- coding: utf-8 -*-
"""
Enhanced LangMem tools for research agent
Provides semantic memory management with research-specific features
"""

from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain.tools import Tool
from typing import Dict, Any, List
import json

def get_memory_tools():
    """Get enhanced memory tools for research agent"""
    
    # Create base LangMem tools
    manage_tool = create_manage_memory_tool(name="save_memory")
    search_tool = create_search_memory_tool(name="search_memory")
    
    # Enhanced search tool with better formatting
    def enhanced_search(query: str) -> str:
        """Enhanced memory search with better result formatting"""
        try:
            # Use the base search tool
            results = search_tool.invoke({"query": query})
            
            if not results or results == "No relevant memories found.":
                return f"No relevant information found in memory for: {query}"
            
            # Format results for better readability
            formatted_results = f"Memory search results for '{query}':\n"
            formatted_results += "-" * 40 + "\n"
            formatted_results += str(results)
            
            return formatted_results
            
        except Exception as e:
            return f"Error searching memory: {str(e)}"
    
    # Enhanced save tool with metadata support
    def enhanced_save(content: str, metadata: Dict[str, Any] = None) -> str:
        """Enhanced memory save with metadata support"""
        try:
            # Prepare content with metadata if provided
            if metadata:
                enriched_content = f"{content}\n[Metadata: {json.dumps(metadata)}]"
            else:
                enriched_content = content
            
            # Use the base manage tool
            result = manage_tool.invoke({"content": enriched_content})
            
            return f"Successfully saved to memory: {content[:100]}..."
            
        except Exception as e:
            return f"Error saving to memory: {str(e)}"
    
    # Create enhanced tools
    enhanced_search_tool = Tool(
        name="search_memory",
        description="Search semantic memory for relevant information. Use this to find previously stored research findings, facts, or insights.",
        func=enhanced_search
    )
    
    enhanced_save_tool = Tool(
        name="save_memory", 
        description="Save important information to semantic memory. Use this to store research findings, insights, or facts for future reference.",
        func=lambda content: enhanced_save(content)
    )
    
    return [enhanced_save_tool, enhanced_search_tool]

def create_research_memory_tool():
    """Create a specialized tool for research-specific memory operations"""
    
    def research_memory_operation(operation: str, content: str = "", query: str = "") -> str:
        """Perform research-specific memory operations"""
        tools = get_memory_tools()
        
        if operation == "save_finding":
            return tools[0].invoke(f"Research Finding: {content}")
        elif operation == "search_topic":
            return tools[1].invoke(f"Topic: {query}")
        elif operation == "save_hypothesis":
            return tools[0].invoke(f"Research Hypothesis: {content}")
        elif operation == "search_evidence":
            return tools[1].invoke(f"Evidence for: {query}")
        else:
            return "Invalid operation. Use: save_finding, search_topic, save_hypothesis, search_evidence"
    
    return Tool(
        name="research_memory",
        description="Specialized memory tool for research operations. Operations: save_finding, search_topic, save_hypothesis, search_evidence",
        func=lambda input_str: research_memory_operation(*input_str.split("|", 2))
    )