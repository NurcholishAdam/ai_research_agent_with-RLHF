# -*- coding: utf-8 -*-
"""
Research Tools Manager for AI Research Agent
Integrates and manages all Phase 3 research tools
"""

from typing import List, Dict, Any
from langchain.tools import Tool
from tools.web_research import get_web_research_tools
from tools.document_processor import get_document_processing_tools
from tools.data_visualization import get_visualization_tools

class ResearchToolsManager:
    """Manages all research tools and provides intelligent tool selection"""
    
    def __init__(self):
        self.web_tools = get_web_research_tools()
        self.document_tools = get_document_processing_tools()
        self.visualization_tools = get_visualization_tools()
        
        # Combine all tools
        self.all_tools = (
            self.web_tools + 
            self.document_tools + 
            self.visualization_tools
        )
        
        # Create tool categories
        self.tool_categories = {
            'web_research': [tool.name for tool in self.web_tools],
            'document_processing': [tool.name for tool in self.document_tools],
            'data_visualization': [tool.name for tool in self.visualization_tools]
        }
    
    def get_all_tools(self) -> List[Tool]:
        """Get all available research tools"""
        return self.all_tools
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get tools by category"""
        if category not in self.tool_categories:
            return []
        
        tool_names = self.tool_categories[category]
        return [tool for tool in self.all_tools if tool.name in tool_names]
    
    def suggest_tools_for_query(self, query: str) -> Dict[str, List[str]]:
        """Suggest appropriate tools based on research query"""
        query_lower = query.lower()
        suggestions = {
            'recommended': [],
            'optional': [],
            'visualization': []
        }
        
        # Web research suggestions
        if any(keyword in query_lower for keyword in ['current', 'recent', 'latest', 'news', 'today']):
            suggestions['recommended'].extend(['web_search', 'news_search'])
        
        if any(keyword in query_lower for keyword in ['academic', 'research', 'paper', 'study']):
            suggestions['recommended'].append('arxiv_search')
        
        if any(keyword in query_lower for keyword in ['definition', 'what is', 'explain', 'background']):
            suggestions['recommended'].append('wikipedia_search')
        
        # Document processing suggestions
        if any(keyword in query_lower for keyword in ['pdf', 'document', 'paper', 'article']):
            suggestions['optional'].extend(['process_pdf_url', 'analyze_document_structure'])
        
        # Visualization suggestions
        if any(keyword in query_lower for keyword in ['trend', 'timeline', 'over time', 'history']):
            suggestions['visualization'].append('create_timeline_visualization')
        
        if any(keyword in query_lower for keyword in ['relationship', 'connection', 'network', 'concept']):
            suggestions['visualization'].append('create_concept_network')
        
        if any(keyword in query_lower for keyword in ['analysis', 'frequency', 'common', 'popular']):
            suggestions['visualization'].append('create_word_frequency_chart')
        
        # Default suggestions if none match
        if not any(suggestions.values()):
            suggestions['recommended'] = ['web_search', 'wikipedia_search']
            suggestions['optional'] = ['arxiv_search']
        
        return suggestions
    
    def get_tool_usage_guide(self) -> str:
        """Get comprehensive guide for using research tools"""
        guide = """
🛠️ Research Tools Arsenal - Usage Guide
=====================================

📡 WEB RESEARCH TOOLS:
• web_search - General web search using DuckDuckGo
• wikipedia_search - Encyclopedic information and definitions
• arxiv_search - Academic papers and research
• news_search - Recent news and current events
• scrape_webpage - Detailed analysis of specific webpages

📄 DOCUMENT PROCESSING TOOLS:
• process_pdf_url - Download and analyze PDFs from URLs
• process_local_pdf - Analyze local PDF files
• analyze_document_structure - Extract sections and structure from text

📊 DATA VISUALIZATION TOOLS:
• create_timeline_visualization - Timeline charts from chronological data
• create_concept_network - Network graphs of concept relationships
• create_metrics_dashboard - Comprehensive metrics dashboards
• create_word_frequency_chart - Word frequency analysis
• analyze_research_data - Suggest appropriate visualizations

🎯 TOOL SELECTION TIPS:
• Start with web_search for general information
• Use wikipedia_search for background and definitions
• Use arxiv_search for academic/scientific topics
• Use news_search for current events and recent developments
• Use document tools when you have specific PDFs to analyze
• Use visualization tools to create charts and graphs from your findings

🔄 WORKFLOW SUGGESTIONS:
1. Begin with web_search or wikipedia_search for overview
2. Use arxiv_search for academic depth
3. Process relevant PDFs with document tools
4. Create visualizations to present findings
5. Use news_search for latest developments
"""
        return guide

def get_research_tools_manager():
    """Get the research tools manager instance"""
    return ResearchToolsManager()

def get_all_research_tools():
    """Get all Phase 3 research tools"""
    manager = ResearchToolsManager()
    
    # Add tool suggestion capability
    def suggest_tools_tool(query: str) -> str:
        """Suggest appropriate research tools for a query"""
        try:
            suggestions = manager.suggest_tools_for_query(query)
            
            output = f"Tool Suggestions for: '{query}'\n"
            output += "=" * 50 + "\n"
            
            if suggestions['recommended']:
                output += "🎯 RECOMMENDED TOOLS:\n"
                for tool in suggestions['recommended']:
                    output += f"  • {tool}\n"
                output += "\n"
            
            if suggestions['optional']:
                output += "🔧 OPTIONAL TOOLS:\n"
                for tool in suggestions['optional']:
                    output += f"  • {tool}\n"
                output += "\n"
            
            if suggestions['visualization']:
                output += "📊 VISUALIZATION TOOLS:\n"
                for tool in suggestions['visualization']:
                    output += f"  • {tool}\n"
                output += "\n"
            
            output += "💡 Use 'get_tools_guide' for detailed usage instructions"
            
            return output
            
        except Exception as e:
            return f"Tool suggestion failed: {str(e)}"
    
    def get_tools_guide_tool(dummy_input: str = "") -> str:
        """Get comprehensive research tools usage guide"""
        return manager.get_tool_usage_guide()
    
    # Add the management tools
    management_tools = [
        Tool(
            name="suggest_research_tools",
            description="Suggest appropriate research tools based on your research query. Input your research question or topic.",
            func=suggest_tools_tool
        ),
        Tool(
            name="get_tools_guide",
            description="Get comprehensive guide for using all research tools. No input required.",
            func=get_tools_guide_tool
        )
    ]
    
    return manager.get_all_tools() + management_tools