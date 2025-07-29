# -*- coding: utf-8 -*-
"""
Document Processing Tools for AI Research Agent
Handles PDF ingestion, text extraction, and document analysis
"""

import PyPDF2
import requests
from typing import Dict, List, Any, Optional
from langchain.tools import Tool
import re
import os
from urllib.parse import urlparse
import tempfile
import json
from datetime import datetime

class DocumentProcessor:
    """Advanced document processing capabilities"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']
        self.temp_dir = tempfile.gettempdir()
    
    def download_pdf(self, url: str) -> str:
        """Download PDF from URL to temporary location"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create temporary file
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or "document.pdf"
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            temp_path = os.path.join(self.temp_dir, f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            return temp_path
            
        except Exception as e:
            raise Exception(f"Failed to download PDF: {str(e)}")
    
    def extract_pdf_text(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {}
                if pdf_reader.metadata:
                    metadata = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                    }
                
                # Extract text from all pages
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        page_texts.append({
                            'page_number': page_num + 1,
                            'text': page_text,
                            'word_count': len(page_text.split())
                        })
                        full_text += page_text + "\n"
                    except Exception as e:
                        page_texts.append({
                            'page_number': page_num + 1,
                            'text': f"Error extracting page: {str(e)}",
                            'word_count': 0
                        })
                
                # Clean up text
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                
                # Extract key information
                key_info = self._extract_key_information(full_text)
                
                return {
                    'file_path': file_path,
                    'total_pages': len(pdf_reader.pages),
                    'metadata': metadata,
                    'full_text': full_text,
                    'page_texts': page_texts,
                    'word_count': len(full_text.split()),
                    'character_count': len(full_text),
                    'key_information': key_info,
                    'status': 'success'
                }
                
        except Exception as e:
            return {
                'file_path': file_path,
                'error': f"Failed to extract PDF text: {str(e)}",
                'status': 'error'
            }
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and extract sections"""
        try:
            # Find potential sections/headings
            lines = text.split('\n')
            sections = []
            current_section = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if line might be a heading
                is_heading = (
                    len(line) < 100 and  # Short lines
                    (line.isupper() or  # All caps
                     re.match(r'^\d+\.?\s+[A-Z]', line) or  # Numbered sections
                     re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', line))  # Title case
                )
                
                if is_heading:
                    if current_section:
                        sections.append(current_section)
                    current_section = {
                        'title': line,
                        'start_line': i,
                        'content': []
                    }
                elif current_section:
                    current_section['content'].append(line)
            
            # Add last section
            if current_section:
                sections.append(current_section)
            
            # Extract other structural elements
            references = self._extract_references(text)
            figures_tables = self._extract_figures_tables(text)
            
            return {
                'sections': sections[:10],  # Limit to first 10 sections
                'total_sections': len(sections),
                'references': references,
                'figures_tables': figures_tables,
                'has_bibliography': 'references' in text.lower() or 'bibliography' in text.lower()
            }
            
        except Exception as e:
            return {
                'error': f"Failed to analyze document structure: {str(e)}"
            }
    
    def _extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from document text"""
        try:
            # Extract emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            
            # Extract URLs
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            
            # Extract DOIs
            dois = re.findall(r'10\.\d{4,}/[^\s]+', text)
            
            # Extract years (potential publication years)
            years = re.findall(r'\b(19|20)\d{2}\b', text)
            recent_years = [year for year in years if int(year) >= 2000]
            
            # Extract potential author names (simple heuristic)
            author_patterns = re.findall(r'\b[A-Z][a-z]+,?\s+[A-Z]\.(?:\s+[A-Z]\.)?', text)
            
            return {
                'emails': list(set(emails))[:5],
                'urls': list(set(urls))[:10],
                'dois': list(set(dois))[:5],
                'recent_years': list(set(recent_years))[:10],
                'potential_authors': list(set(author_patterns))[:10]
            }
            
        except Exception as e:
            return {'error': f"Failed to extract key information: {str(e)}"}
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references from document"""
        try:
            # Look for reference section
            ref_section_match = re.search(r'(?i)(references|bibliography)(.*?)(?=\n\n[A-Z]|\Z)', text, re.DOTALL)
            
            if ref_section_match:
                ref_text = ref_section_match.group(2)
                # Split by lines and filter
                references = []
                for line in ref_text.split('\n'):
                    line = line.strip()
                    if len(line) > 20 and (re.search(r'\d{4}', line) or 'doi:' in line.lower()):
                        references.append(line)
                
                return references[:20]  # Limit to 20 references
            
            return []
            
        except Exception:
            return []
    
    def _extract_figures_tables(self, text: str) -> Dict[str, List[str]]:
        """Extract figure and table references"""
        try:
            figures = re.findall(r'(?i)figure\s+\d+[:\.]?[^\n]*', text)
            tables = re.findall(r'(?i)table\s+\d+[:\.]?[^\n]*', text)
            
            return {
                'figures': figures[:10],
                'tables': tables[:10]
            }
            
        except Exception:
            return {'figures': [], 'tables': []}
    
    def cleanup_temp_files(self, file_path: str):
        """Clean up temporary files"""
        try:
            if os.path.exists(file_path) and file_path.startswith(self.temp_dir):
                os.remove(file_path)
        except Exception:
            pass  # Ignore cleanup errors

def get_document_processing_tools():
    """Get document processing tools"""
    
    processor = DocumentProcessor()
    
    def process_pdf_url_tool(url: str) -> str:
        """Download and process PDF from URL"""
        try:
            # Download PDF
            temp_path = processor.download_pdf(url)
            
            # Extract text and analyze
            result = processor.extract_pdf_text(temp_path)
            
            if result['status'] == 'error':
                processor.cleanup_temp_files(temp_path)
                return f"PDF processing error: {result['error']}"
            
            # Analyze structure
            structure = processor.analyze_document_structure(result['full_text'])
            
            # Format output
            formatted_output = f"PDF Analysis: {result['metadata'].get('title', 'Unknown Title')}\n"
            formatted_output += "=" * 60 + "\n"
            formatted_output += f"Source URL: {url}\n"
            formatted_output += f"Pages: {result['total_pages']}\n"
            formatted_output += f"Word Count: {result['word_count']}\n"
            
            if result['metadata'].get('author'):
                formatted_output += f"Author: {result['metadata']['author']}\n"
            
            formatted_output += f"\nDocument Structure:\n"
            if structure.get('sections'):
                formatted_output += f"- {len(structure['sections'])} sections identified\n"
                for section in structure['sections'][:5]:
                    formatted_output += f"  • {section['title']}\n"
            
            if structure.get('references'):
                formatted_output += f"- {len(structure['references'])} references found\n"
            
            # Include first part of content
            formatted_output += f"\nContent Preview:\n"
            formatted_output += result['full_text'][:2000] + "...\n"
            
            # Key information
            key_info = result['key_information']
            if key_info.get('dois'):
                formatted_output += f"\nDOIs found: {', '.join(key_info['dois'][:3])}\n"
            
            # Cleanup
            processor.cleanup_temp_files(temp_path)
            
            return formatted_output
            
        except Exception as e:
            return f"PDF processing failed: {str(e)}"
    
    def process_local_pdf_tool(file_path: str) -> str:
        """Process local PDF file"""
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            result = processor.extract_pdf_text(file_path)
            
            if result['status'] == 'error':
                return f"PDF processing error: {result['error']}"
            
            # Analyze structure
            structure = processor.analyze_document_structure(result['full_text'])
            
            # Format output (similar to URL version)
            formatted_output = f"PDF Analysis: {result['metadata'].get('title', os.path.basename(file_path))}\n"
            formatted_output += "=" * 60 + "\n"
            formatted_output += f"File Path: {file_path}\n"
            formatted_output += f"Pages: {result['total_pages']}\n"
            formatted_output += f"Word Count: {result['word_count']}\n"
            
            if result['metadata'].get('author'):
                formatted_output += f"Author: {result['metadata']['author']}\n"
            
            formatted_output += f"\nContent Preview:\n"
            formatted_output += result['full_text'][:2000] + "...\n"
            
            return formatted_output
            
        except Exception as e:
            return f"Local PDF processing failed: {str(e)}"
    
    def extract_document_sections_tool(content: str) -> str:
        """Analyze document structure and extract sections"""
        try:
            structure = processor.analyze_document_structure(content)
            
            if 'error' in structure:
                return f"Structure analysis error: {structure['error']}"
            
            formatted_output = "Document Structure Analysis\n"
            formatted_output += "=" * 40 + "\n"
            
            if structure.get('sections'):
                formatted_output += f"Sections Found ({len(structure['sections'])}):\n"
                for i, section in enumerate(structure['sections'][:10], 1):
                    formatted_output += f"{i}. {section['title']}\n"
                    if section['content']:
                        preview = ' '.join(section['content'][:3])[:200]
                        formatted_output += f"   Preview: {preview}...\n"
                    formatted_output += "\n"
            
            if structure.get('references'):
                formatted_output += f"References ({len(structure['references'])}):\n"
                for ref in structure['references'][:5]:
                    formatted_output += f"- {ref[:100]}...\n"
            
            if structure.get('figures_tables'):
                figs = structure['figures_tables'].get('figures', [])
                tables = structure['figures_tables'].get('tables', [])
                if figs:
                    formatted_output += f"\nFigures: {len(figs)} found\n"
                if tables:
                    formatted_output += f"Tables: {len(tables)} found\n"
            
            return formatted_output
            
        except Exception as e:
            return f"Document structure analysis failed: {str(e)}"
    
    return [
        Tool(
            name="process_pdf_url",
            description="Download and analyze PDF from URL. Extract text, metadata, and structure. Input should be a valid PDF URL.",
            func=process_pdf_url_tool
        ),
        Tool(
            name="process_local_pdf",
            description="Process and analyze local PDF file. Input should be a valid file path to a PDF.",
            func=process_local_pdf_tool
        ),
        Tool(
            name="analyze_document_structure",
            description="Analyze document structure and extract sections from text content. Input should be document text.",
            func=extract_document_sections_tool
        )
    ]