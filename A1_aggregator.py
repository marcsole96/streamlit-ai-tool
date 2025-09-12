# Agent 1 Aggregator

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import io
import torch
import requests
import base64
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from datetime import datetime
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import textwrap
from sklearn.preprocessing import normalize
import PyPDF2
import pickle
import uuid
import time
import hdbscan
from sklearn.metrics import silhouette_score
import nltk 
nltk.download('punkt_tab')
# Try to import graphviz layout - not available on all systems
try:
    from networkx.drawing.nx_agraph import graphviz_layout
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

import fitz  # PyMuPDF
import pdfplumber
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
from itertools import chain

# Try to download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


#############################################
# CLASS DEFINITIONS
#############################################

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)



class EmbeddingCalculator:
    """Class for calculating and managing embeddings"""
    def __init__(self):
        self.model_name = "jinaai/jina-embeddings-v3"
        self.model = None
        self.tokenizer = None
        self.embeddings = {}  # Store embeddings: {id: embedding_vector}
        self.element_map = {}  # Map element IDs to full elements
        self.similarity_matrix = None
        self.clusters = []  # List of clusters (each cluster is a list of element IDs)
        
    def load_model(self, status_callback=None):
        """Load the embedding model and tokenizer"""
        if status_callback:
            status_callback(f"Loading embedding model: {self.model_name}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            
            if status_callback:
                status_callback("Embedding model loaded successfully")
            
            return True
        except Exception as e:
            if status_callback:
                status_callback(f"Error loading embedding model: {str(e)}")
            return False
    
    def compute_embeddings(self, elements, status_callback=None):
        """
        Compute embeddings for a list of elements
        
        Args:
            elements: List of element dictionaries
            status_callback: Function to call with status updates
        """
        if not self.model or not self.tokenizer:
            if not self.load_model(status_callback):
                return False
        
        if status_callback:
            status_callback(f"Computing embeddings for {len(elements)} elements...")
        
        # Clear previous embeddings
        self.embeddings = {}
        self.element_map = {}
        
        # Process elements in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(elements), batch_size):
            batch = elements[i:i+batch_size]
            
            # Extract element names and IDs
            element_ids = [element['id'] for element in batch]
            element_names = [element['name'] for element in batch]
            
            # Store elements in the element map
            for element in batch:
                self.element_map[element['id']] = element
            
            # Tokenize and compute embeddings
            inputs = self.tokenizer(element_names, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                model_output = self.model(**inputs)
                # Use the mean of the last hidden state as the sentence embedding
                embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
            
            # Store embeddings by element ID
            for j, element_id in enumerate(element_ids):
                self.embeddings[element_id] = embeddings[j].numpy()
            
            if status_callback:
                status_callback(f"Processed {min(i+batch_size, len(elements))}/{len(elements)} elements")
        
        if status_callback:
            status_callback("Embeddings computation completed")
        
        return True
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenizer
        
        Returns:
            Tensor with sentence embeddings
        """
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def compute_similarity_matrix(self, element_ids=None, status_callback=None):
        """
        Compute similarity matrix between elements
        
        Args:
            element_ids: Optional list of element IDs to include (if None, use all)
            status_callback: Function to call with status updates
        
        Returns:
            DataFrame with similarity scores
        """
        if not self.embeddings:
            if status_callback:
                status_callback("No embeddings available. Please compute embeddings first.")
            return None
        
        if status_callback:
            status_callback("Computing similarity matrix...")
        
        # Get all element IDs or filter to requested ones
        if element_ids is None:
            element_ids = list(self.embeddings.keys())
        else:
            # Only include elements that have embeddings
            element_ids = [id for id in element_ids if id in self.embeddings]
        
        n_elements = len(element_ids)
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_elements, n_elements))
        
        # Compute pairwise cosine similarities
        for i in range(n_elements):
            for j in range(i, n_elements):
                id1 = element_ids[i]
                id2 = element_ids[j]
                
                # Calculate cosine similarity (1 - cosine distance)
                sim = 1 - cosine(self.embeddings[id1], self.embeddings[id2])
                
                # Store in the matrix (symmetric)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Create DataFrame for better readability
        df = pd.DataFrame(similarity_matrix, index=element_ids, columns=element_ids)
        
        # Store for later use
        self.similarity_matrix = df
        
        if status_callback:
            status_callback("Similarity matrix computed successfully")
        
        return df
    
    def get_embedding_matrix(self, element_ids=None):
        """
        Get a matrix of embeddings for specified element IDs
        
        Args:
            element_ids: Optional list of element IDs to include (if None, use all)
            
        Returns:
            Tuple of (embedding_matrix, element_ids)
        """
        if not self.embeddings:
            return None, None
        
        # Get all element IDs or filter to requested ones
        if element_ids is None:
            element_ids = list(self.embeddings.keys())
        else:
            # Only include elements that have embeddings
            element_ids = [id for id in element_ids if id in self.embeddings]
        
        # Stack embeddings
        embedding_matrix = np.array([self.embeddings[id] for id in element_ids])
        
        return embedding_matrix, element_ids
    
    def perform_hdbscan_clustering(self, element_ids=None, min_cluster_size=3, min_samples=2, 
                                cluster_selection_epsilon=0.5, status_callback=None):
        """
        Perform HDBSCAN clustering on embeddings
        
        Args:
            element_ids: Optional list of element IDs to include (if None, use all)
            min_cluster_size: Minimum size of clusters (number of points)
            min_samples: Number of samples in a neighborhood for a point to be a core point
            cluster_selection_epsilon: Distance threshold for cluster extraction
            status_callback: Function to call with status updates
            
        Returns:
            Tuple of (element_ids, cluster_labels, clusterer)
        """
        if not self.embeddings:
            if status_callback:
                status_callback("No embeddings available. Please compute embeddings first.")
            return None, None, None
        
        if status_callback:
            status_callback("Performing HDBSCAN clustering...")
        
        # Get embedding matrix
        embedding_matrix, element_ids = self.get_embedding_matrix(element_ids)
        
        if embedding_matrix is None or len(embedding_matrix) < min_cluster_size:
            if status_callback:
                status_callback("Not enough elements for clustering")
            return None, None, None
        
        # Normalize embeddings to unit length for cosine similarity effect
        if status_callback:
            status_callback("Normalizing embeddings for cosine similarity...")
        
        normalized_matrix = normalize(embedding_matrix, norm='l2', axis=1)
        
        # Create and fit HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean',  # Still using euclidean, but on normalized vectors
            prediction_data=True
        )
        
        cluster_labels = clusterer.fit_predict(normalized_matrix)
        
        # Calculate number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if status_callback:
            status_callback(f"Found {n_clusters} clusters with {n_noise} noise points")
            
            # Calculate silhouette score if more than one cluster and no noise points
            if n_clusters > 1 and n_noise == 0:
                try:
                    silhouette_avg = silhouette_score(normalized_matrix, cluster_labels)
                    status_callback(f"Silhouette score: {silhouette_avg:.3f} (higher is better)")
                except:
                    pass  # Skip silhouette score if it fails
                
        return element_ids, cluster_labels, clusterer

    def get_similar_elements(self, element_id, top_n=5, threshold=0.6):
        """
        Get the most similar elements to a given element
        
        Args:
            element_id: The ID of the element to find similarities for
            top_n: Number of similar elements to return
            threshold: Minimum similarity score
            
        Returns:
            List of dictionaries with similar elements and their scores
        """
        if element_id not in self.embeddings:
            return []
        
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
        
        # Get similarities for this element
        if element_id in self.similarity_matrix.index:
            similarities = self.similarity_matrix[element_id].sort_values(ascending=False)
            
            # Filter by threshold and exclude self
            similarities = similarities[similarities > threshold]
            similarities = similarities[similarities.index != element_id]
            
            # Take top N
            similarities = similarities.head(top_n)
            
            # Format results
            results = []
            for similar_id, score in similarities.items():
                element = self.element_map.get(similar_id, {})
                results.append({
                    'id': similar_id,
                    'name': element.get('name', 'Unknown'),
                    'similarity': score,
                    'element': element
                })
            
            return results
        
        return []


class RAGManager:
    """Enhanced class for managing Retrieval-Augmented Generation (RAG) resources"""
    def __init__(self):
        self.taxonomy_sources = {}  # {source_id: {"name": name, "content": taxonomy_data}}
        self.literature_sources = {}  # {source_id: {"name": name, "content": text, "chunks": chunks, "embeddings": embeddings}}
        self.babelnet_cache = {}  # {term: response_data}
        self.custom_sources = {}  # {source_id: {"name": name, "content": text}}
        self.embedding_model = None
        self.chunk_stats = {}  # Statistics about chunking for each source
        self.retrieval_stats = {}  # Statistics about retrieval operations
        
    def add_taxonomy_source(self, name, file_content):
        """Add a taxonomy source file"""
        source_id = f"tax_{uuid.uuid4().hex[:8]}"
        try:
            # Parse the JSON content
            if isinstance(file_content, str):
                taxonomy_data = json.loads(file_content)
            else:
                taxonomy_data = json.load(file_content)
            
            self.taxonomy_sources[source_id] = {
                "name": name,
                "content": taxonomy_data,
                "date_added": datetime.now().isoformat()
            }
            return source_id
        except Exception as e:
            st.error(f"Error adding taxonomy source: {str(e)}")
            return None
    
    def add_literature_source(self, name, file_content):
        """Enhanced method to add a literature source (PDF) with improved extraction"""
        source_id = f"lit_{uuid.uuid4().hex[:8]}"
        try:
            # Create a BytesIO object if file_content is bytes
            if isinstance(file_content, bytes):
                pdf_file = io.BytesIO(file_content)
            else:
                # Read file-like object into bytes
                pdf_file = io.BytesIO(file_content.read())
                pdf_file.seek(0)  # Reset position to beginning
            
            # Extract text and structure using PyMuPDF (fitz)
            pdf_text, toc, metadata, page_texts = self._extract_pdf_text_and_structure(pdf_file)
            
            # Create document map/outline
            document_map = self._create_document_map(pdf_text, toc, metadata)
            
            # Create semantic chunks with hierarchy preservation
            chunks, chunk_metadata = self._semantic_chunking(pdf_text, page_texts, toc)
            
            # Generate section and document summaries
            summaries = self._generate_summaries(chunks, chunk_metadata)
            
            # Store all the information
            self.literature_sources[source_id] = {
                "name": name,
                "content": pdf_text,
                "chunks": chunks,
                "chunk_metadata": chunk_metadata,
                "toc": toc,
                "metadata": metadata,
                "document_map": document_map,
                "summaries": summaries,
                "embeddings": None,  # Will be computed when needed
                "section_embeddings": None,  # Section-level embeddings
                "document_embedding": None,  # Document-level embedding
                "date_added": datetime.now().isoformat()
            }
            
            # Store chunking statistics
            self.chunk_stats[source_id] = {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk) for chunk in chunks) / max(1, len(chunks)),
                "section_coverage": len(set(m.get("section", "") for m in chunk_metadata)) if chunk_metadata else 0,
                "semantic_boundaries": sum(1 for m in chunk_metadata if m.get("semantic_boundary", False)) if chunk_metadata else 0
            }
            
            return source_id
        except Exception as e:
            st.error(f"Error adding literature source: {str(e)}")
            return None
    
    def _extract_pdf_text_and_structure(self, pdf_file):
        """
        Extract text and structure from PDF using PyMuPDF and pdfplumber for better accuracy
        
        Returns:
            tuple: (full_text, table_of_contents, metadata, page_texts)
        """
        pdf_text = ""
        toc = []
        metadata = {}
        page_texts = []
        
        # Reset file position to beginning
        pdf_file.seek(0)
        
        # Try extraction with PyMuPDF first
        try:
            with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                # Extract metadata
                metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "keywords": doc.metadata.get("keywords", ""),
                    "page_count": len(doc),
                    "producer": doc.metadata.get("producer", "")
                }
                
                # Extract table of contents
                toc = doc.get_toc()
                
                # Extract text with structure preservation
                for page_num, page in enumerate(doc):
                    # Get text with blocks info to preserve layout
                    blocks = page.get_text("dict")["blocks"]
                    page_text = ""
                    
                    # Process each block to preserve structure
                    for block in blocks:
                        if block.get("type") == 0:  # Text block
                            for line in block.get("lines", []):
                                line_text = ""
                                for span in line.get("spans", []):
                                    # Check if it's a heading by font size
                                    font_size = span.get("size", 0)
                                    text = span.get("text", "").strip()
                                    
                                    if font_size > 12:  # Likely a heading
                                        line_text += f"\n## {text} ##\n"
                                    else:
                                        line_text += text + " "
                                
                                page_text += line_text.strip() + "\n"
                        
                        elif block.get("type") == 1:  # Image block
                            page_text += "\n[IMAGE]\n"
                    
                    page_texts.append(page_text)
                    pdf_text += page_text + "\n\n"
        
        except Exception as e:
            # Fall back to pdfplumber if PyMuPDF fails
            pdf_file.seek(0)
            
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    # Basic metadata
                    metadata = {
                        "page_count": len(pdf.pages),
                    }
                    
                    # Extract text page by page
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text() or ""
                        
                        # Try to detect headings by looking at first lines of pages or lines with few words but all caps
                        lines = page_text.split('\n')
                        for j, line in enumerate(lines):
                            if j == 0 and i > 0 and len(line.strip()) > 0 and len(line.strip()) < 100:
                                # First line of a page might be a header
                                lines[j] = f"\n## {line.strip()} ##\n"
                            elif line.isupper() and 2 <= len(line.split()) <= 7:
                                # Likely a heading if it's all uppercase and has few words
                                lines[j] = f"\n## {line.strip()} ##\n"
                        
                        processed_text = '\n'.join(lines)
                        page_texts.append(processed_text)
                        pdf_text += processed_text + "\n\n"
                        
                        # Look for tables
                        tables = page.extract_tables()
                        if tables:
                            table_text = "\n[TABLE]\n"
                            page_texts[-1] += table_text
                            pdf_text += table_text
            
            except Exception as inner_e:
                # Last resort: try simple PyPDF2-like extraction
                pdf_file.seek(0)
                
                try:
                    with fitz.open(stream=pdf_file, filetype="pdf") as doc:
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            page_text = page.get_text("text")
                            page_texts.append(page_text)
                            pdf_text += page_text + "\n\n"
                        
                        metadata = {"page_count": len(doc)}
                        
                except Exception as final_e:
                    # If all else fails, return empty with error message
                    st.error(f"Failed to extract PDF content: {str(final_e)}")
                    return "", [], {}, []
        
        return pdf_text, toc, metadata, page_texts
    
    def _create_document_map(self, pdf_text, toc, metadata):
        """
        Create a structured map/outline of the document for easier navigation
        
        Returns:
            dict: A hierarchical representation of the document structure
        """
        document_map = {
            "title": metadata.get("title", "Untitled Document"),
            "metadata": metadata,
            "structure": []
        }
        
        # Use table of contents if available
        if toc:
            for item in toc:
                level, title, page = item
                document_map["structure"].append({
                    "level": level,
                    "title": title,
                    "page": page,
                    "type": "toc_entry"
                })
            return document_map
        
        # If no TOC, try to infer structure from extracted headings
        heading_pattern = r"\n##\s+(.+?)\s+##\n"
        headings = re.findall(heading_pattern, pdf_text)
        
        current_level = 1
        for heading in headings:
            # Simple heuristic: shorter headings with all caps might be higher level
            if heading.isupper() and len(heading) < 30:
                level = 1
            else:
                level = 2
            
            document_map["structure"].append({
                "level": level,
                "title": heading,
                "page": None,  # We don't know the page without more complex analysis
                "type": "extracted_heading"
            })
        
        # If we still have no structure, create artificial sections based on pages
        if not document_map["structure"] and "page_count" in metadata:
            page_count = metadata["page_count"]
            # Create sections for every few pages
            section_size = min(5, max(1, page_count // 10))
            for i in range(0, page_count, section_size):
                end_page = min(i + section_size - 1, page_count - 1)
                document_map["structure"].append({
                    "level": 1,
                    "title": f"Section: Pages {i+1}-{end_page+1}",
                    "page": i,
                    "type": "auto_section"
                })
        
        return document_map
    
    def _semantic_chunking(self, pdf_text, page_texts, toc):
        """
        Create semantic chunks that respect document structure and logical units
        
        Returns:
            tuple: (chunks, chunk_metadata)
        """
        chunks = []
        chunk_metadata = []
        
        # Identify semantic boundaries: headings, page breaks, sections
        # Pattern for headings we inserted during extraction
        heading_pattern = r"\n##\s+(.+?)\s+##\n"
        
        # If we have a structured TOC, use it to identify main section boundaries
        toc_sections = {}
        if toc:
            for level, title, page in toc:
                if page < len(page_texts):
                    toc_sections[page] = (level, title)
        
        # First, split by headings and page boundaries
        current_chunk = ""
        current_section = ""
        current_page = 0
        section_level = 0
        
        # Process with proper semantic boundaries
        for page_idx, page_text in enumerate(page_texts):
            # Check if this page starts a new TOC section
            if page_idx in toc_sections:
                # If we have accumulated text, save it as a chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    chunk_metadata.append({
                        "section": current_section,
                        "section_level": section_level,
                        "start_page": current_page,
                        "end_page": page_idx - 1,
                        "semantic_boundary": True,
                        "boundary_type": "section",
                        "word_count": len(current_chunk.split())
                    })
                
                # Start a new chunk with this section
                level, title = toc_sections[page_idx]
                current_section = title
                section_level = level
                current_chunk = f"Section: {title}\n\n{page_text}"
                current_page = page_idx
                continue
            
            # Process headings within the page
            headings = re.findall(heading_pattern, page_text)
            if headings:
                # Split by headings
                parts = re.split(heading_pattern, page_text)
                
                # First part may be text before any heading
                if parts[0].strip():
                    current_chunk += "\n\n" + parts[0].strip()
                
                # Process each heading and its content
                for i in range(1, len(parts), 2):  # heading, content, heading, content...
                    heading = parts[i].strip()
                    content = parts[i+1].strip() if i+1 < len(parts) else ""
                    
                    # Save current chunk if it's substantial
                    if current_chunk and len(current_chunk.split()) > 50:
                        chunks.append(current_chunk.strip())
                        chunk_metadata.append({
                            "section": current_section,
                            "section_level": section_level,
                            "start_page": current_page,
                            "end_page": page_idx,
                            "semantic_boundary": True,
                            "boundary_type": "heading",
                            "word_count": len(current_chunk.split())
                        })
                    
                    # Start a new chunk with this heading
                    current_chunk = f"{heading}\n\n{content}"
                    current_page = page_idx
            else:
                # No headings, append to current chunk
                current_chunk += "\n\n" + page_text
            
            # Check if the chunk is getting too large, break by paragraphs if needed
            if len(current_chunk.split()) > 800:  # About 500-800 words is a good chunk size
                # Try to find a good paragraph break
                paragraphs = re.split(r'\n\s*\n', current_chunk)
                
                if len(paragraphs) > 1:
                    # Find a good breaking point around halfway
                    midpoint = len(paragraphs) // 2
                    
                    # Create the first chunk
                    first_chunk = "\n\n".join(paragraphs[:midpoint]).strip()
                    chunks.append(first_chunk)
                    chunk_metadata.append({
                        "section": current_section,
                        "section_level": section_level,
                        "start_page": current_page,
                        "end_page": page_idx,
                        "semantic_boundary": False,
                        "boundary_type": "size_limit",
                        "word_count": len(first_chunk.split())
                    })
                    
                    # Start a new chunk with the remaining paragraphs
                    current_chunk = "\n\n".join(paragraphs[midpoint:]).strip()
                    current_page = page_idx
                else:
                    # If we can't find paragraph breaks, just take the first 500 words
                    words = current_chunk.split()
                    first_chunk = " ".join(words[:500])
                    chunks.append(first_chunk)
                    chunk_metadata.append({
                        "section": current_section,
                        "section_level": section_level,
                        "start_page": current_page,
                        "end_page": page_idx,
                        "semantic_boundary": False,
                        "boundary_type": "size_limit_harsh",
                        "word_count": 500
                    })
                    
                    # Start new chunk with overlap for context preservation
                    current_chunk = " ".join(words[400:])  # 100 words overlap
                    current_page = page_idx
        
        # Add the final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            chunk_metadata.append({
                "section": current_section,
                "section_level": section_level,
                "start_page": current_page,
                "end_page": len(page_texts) - 1,
                "semantic_boundary": True,
                "boundary_type": "document_end",
                "word_count": len(current_chunk.split())
            })
        
        return chunks, chunk_metadata
    
    def _generate_summaries(self, chunks, chunk_metadata):
        """
        Generate summaries at different levels of the document
        
        Returns:
            dict: Summaries for document, sections, and chunks
        """
        summaries = {
            "document": "",
            "sections": {},
            "chunks": {}
        }
        
        try:
            # Group chunks by section
            sections = {}
            for i, metadata in enumerate(chunk_metadata):
                section = metadata.get("section", "")
                if section not in sections:
                    sections[section] = []
                sections[section].append(chunks[i])
            
            # Create section summaries using extractive summarization
            for section, section_chunks in sections.items():
                if not section:
                    continue
                
                # Combine all chunks in this section
                section_text = " ".join(section_chunks)
                
                # Generate extractive summary
                summary = self._extractive_summarize(section_text, 3)  # 3 sentences
                summaries["sections"][section] = summary
            
            # Create document summary from section summaries
            if sections:
                all_section_summaries = " ".join(summaries["sections"].values())
                document_summary = self._extractive_summarize(all_section_summaries, 5)  # 5 sentences
                summaries["document"] = document_summary
            
            # Create chunk summaries
            for i, chunk in enumerate(chunks):
                # Only summarize if chunk is substantial
                if len(chunk.split()) > 100:
                    summaries["chunks"][i] = self._extractive_summarize(chunk, 2)  # 2 sentences
                else:
                    # For small chunks, just use the first sentence
                    first_sentence = sent_tokenize(chunk)
                    summaries["chunks"][i] = first_sentence[0] if first_sentence else chunk[:100]
        
        except Exception as e:
            # If summarization fails, log error and return empty summaries
            print(f"Error generating summaries: {str(e)}")
        
        return summaries
    
    def _extractive_summarize(self, text, num_sentences=3):
        """
        Generate an extractive summary by selecting the most important sentences
        
        Args:
            text: Text to summarize
            num_sentences: Number of sentences to include in summary
            
        Returns:
            str: Extractive summary
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # If text is too short, return it as is
        if len(sentences) <= num_sentences:
            return text
        
        try:
            # Get stopwords
            stop_words = set(stopwords.words('english'))
            
            # Calculate word frequencies
            word_frequencies = {}
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word not in stop_words and word.isalnum():
                        word_frequencies[word] = word_frequencies.get(word, 0) + 1
            
            # Normalize frequencies
            max_frequency = max(word_frequencies.values()) if word_frequencies else 1
            for word in word_frequencies:
                word_frequencies[word] = word_frequencies[word] / max_frequency
            
            # Score sentences based on word frequencies
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                for word in word_tokenize(sentence.lower()):
                    if word in word_frequencies:
                        if i not in sentence_scores:
                            sentence_scores[i] = 0
                        sentence_scores[i] += word_frequencies[word]
            
            # Get top sentences
            top_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            top_indices.sort()  # Sort to maintain original order
            
            # Construct summary
            summary = ' '.join([sentences[i] for i in top_indices])
            return summary
        
        except Exception as e:
            # If anything fails, return first few sentences
            return ' '.join(sentences[:num_sentences])
    
    def add_custom_source(self, name, text_content):
        """Add a custom text source"""
        source_id = f"cus_{uuid.uuid4().hex[:8]}"
        try:
            self.custom_sources[source_id] = {
                "name": name,
                "content": text_content,
                "date_added": datetime.now().isoformat()
            }
            return source_id
        except Exception as e:
            st.error(f"Error adding custom source: {str(e)}")
            return None
    


    def query_babelnet(self, term, api_key=None):
        """Query BabelNet API for term relations using direct approach"""
        if term in self.babelnet_cache:
            return self.babelnet_cache[term]
        
        if not api_key:
            st.warning("BabelNet API key not provided")
            return None
        
        try:
            # Use v5 API endpoint to match BabelNet version 5.3
            url = "https://babelnet.io/v5/getSynsetIds"
            params = {
                "lemma": term,
                "searchLang": "EN",
                "key": api_key
            }
            
            # Log what we're doing
            st.session_state.status_message = f"Querying BabelNet for '{term}'"
            
            # Make request with a timeout
            response = requests.get(url, params=params, timeout=10)
            
            # Check status code explicitly
            if response.status_code != 200:
                st.error(f"BabelNet API returned status code {response.status_code}")
                return None
            
            # Try to parse the response
            data = response.json()
            
            # Check if response is an error message
            if isinstance(data, dict) and 'message' in data:
                st.error(f"BabelNet API error: {data['message']}")
                return None
            
            # Check for valid list response
            if not isinstance(data, list):
                st.warning(f"BabelNet returned unexpected format: {type(data)}")
                return None
            
            # Cache the successful response
            self.babelnet_cache[term] = data
            return data
            
        except json.JSONDecodeError:
            st.error("BabelNet returned invalid JSON response")
            return None
        except requests.exceptions.Timeout:
            st.error("BabelNet API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error making request to BabelNet: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error querying BabelNet: {str(e)}")
            return None

    
    def compute_literature_embeddings(self, source_id):
        """
        Enhanced method to compute embeddings at multiple levels for a literature source
        - Document level embedding
        - Section level embeddings
        - Chunk level embeddings
        """
        if source_id not in self.literature_sources:
            return False
        
        # Skip if embeddings already exist
        lit_source = self.literature_sources[source_id]
        if lit_source.get("embeddings") is not None:
            return True
        
        # Load model if not loaded
        if self.embedding_model is None:
            try:
                # Use the same embedding model as in EmbeddingCalculator
                self.embedding_model = {
                    "tokenizer": AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3"),
                    "model": AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
                }
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return False
        
        # Get data to embed
        chunks = lit_source["chunks"]
        chunk_metadata = lit_source.get("chunk_metadata", [{}] * len(chunks))
        
        # Get document and section texts
        document_text = lit_source.get("summaries", {}).get("document", "")
        if not document_text and "content" in lit_source:
            # Use first 1000 words if no summary
            document_text = " ".join(lit_source["content"].split()[:1000])
        
        section_texts = {}
        for i, metadata in enumerate(chunk_metadata):
            section = metadata.get("section", "")
            if section and section not in section_texts:
                # Use section summary if available, otherwise use first chunk from section
                section_texts[section] = lit_source.get("summaries", {}).get("sections", {}).get(section, chunks[i])
        
        try:
            # Compute embeddings for chunks
            chunk_embeddings = []
            
            batch_size = 8  # Process in small batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Tokenize
                inputs = self.embedding_model["tokenizer"](batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                
                # Compute embeddings
                with torch.no_grad():
                    model_output = self.embedding_model["model"](**inputs)
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Convert to numpy and add to list
                chunk_embeddings.extend(batch_embeddings.numpy())
            
            # Compute section embeddings
            section_embeddings = {}
            if section_texts:
                section_list = list(section_texts.keys())
                section_content_list = [section_texts[section] for section in section_list]
                
                # Process in batches if many sections
                section_batch_size = 8
                for i in range(0, len(section_list), section_batch_size):
                    batch_sections = section_list[i:i+section_batch_size]
                    batch_contents = section_content_list[i:i+section_batch_size]
                    
                    # Tokenize
                    inputs = self.embedding_model["tokenizer"](batch_contents, padding=True, truncation=True, 
                                                               max_length=512, return_tensors="pt")
                    
                    # Compute embeddings
                    with torch.no_grad():
                        model_output = self.embedding_model["model"](**inputs)
                        # Mean pooling
                        attention_mask = inputs['attention_mask']
                        token_embeddings = model_output[0]
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Store section embeddings
                    for j, section in enumerate(batch_sections):
                        section_embeddings[section] = batch_embeddings[j].numpy()
            
            # Compute document embedding
            document_embedding = None
            if document_text:
                inputs = self.embedding_model["tokenizer"]([document_text], padding=True, truncation=True, 
                                                           max_length=512, return_tensors="pt")
                with torch.no_grad():
                    model_output = self.embedding_model["model"](**inputs)
                    attention_mask = inputs['attention_mask']
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    document_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    document_embedding = document_embedding[0].numpy()
            
            # Store all embeddings
            self.literature_sources[source_id]["embeddings"] = chunk_embeddings
            self.literature_sources[source_id]["section_embeddings"] = section_embeddings
            self.literature_sources[source_id]["document_embedding"] = document_embedding
            
            return True
        
        except Exception as e:
            st.error(f"Error computing literature embeddings: {str(e)}")
            return False
    
    def retrieve_relevant_context(self, query, sources=None, max_results=3, retrieval_mode="hybrid"):
        """
        Enhanced method to retrieve relevant context for a query using a more sophisticated approach
        
        Args:
            query: The query text
            sources: List of source_ids to search (if None, search all)
            max_results: Maximum number of results to return
            retrieval_mode: "semantic", "keyword", or "hybrid"
            
        Returns:
            List of relevant context passages with metadata
        """
        results = []
        retrieval_stats = {
            "query": query,
            "mode": retrieval_mode,
            "matches_by_source": {},
            "timing": {"start": time.time()}
        }
        
        # STEP 1: Query Preprocessing
        # Break down complex queries into components
        query_components = self._decompose_query(query)
        retrieval_stats["query_components"] = query_components
        
        # STEP 2: Ensure embedding model is loaded
        if self.embedding_model is None:
            try:
                self.embedding_model = {
                    "tokenizer": AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3"),
                    "model": AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
                }
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                retrieval_stats["error"] = f"Error loading embedding model: {str(e)}"
                self.retrieval_stats[query] = retrieval_stats
                return results
        
        # STEP 3: Encode the query for semantic search
        try:
            # Create embeddings for each query component
            query_embeddings = []
            for component in query_components:
                inputs = self.embedding_model["tokenizer"]([component], padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    model_output = self.embedding_model["model"](**inputs)
                    attention_mask = inputs['attention_mask']
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    component_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    query_embeddings.append(component_embedding.numpy()[0])
            
            # Primary query embedding (original query)
            primary_query_embedding = query_embeddings[0]
            
        except Exception as e:
            st.error(f"Error encoding query: {str(e)}")
            retrieval_stats["error"] = f"Error encoding query: {str(e)}"
            self.retrieval_stats[query] = retrieval_stats
            return results
        
        # STEP 4: Determine which sources to search
        if sources is None:
            literature_sources = list(self.literature_sources.keys())
        else:
            literature_sources = [s for s in sources if s in self.literature_sources]
        
        # STEP 5: Multi-level hierarchical search
        all_matches = []
        
        # First, find matching documents and sections
        matching_sources = {}
        for source_id in literature_sources:
            source = self.literature_sources[source_id]
            source_name = source["name"]
            
            # Compute embeddings if not already done
            if source.get("embeddings") is None:
                if not self.compute_literature_embeddings(source_id):
                    continue
            
            # Check document-level match first
            document_embedding = source.get("document_embedding")
            if document_embedding is not None:
                doc_similarity = cosine_similarity([primary_query_embedding], [document_embedding])[0][0]
                if doc_similarity > 0.5:  # Document is relevant
                    matching_sources[source_id] = {
                        "score": doc_similarity,
                        "source_name": source_name,
                        "matching_sections": []
                    }
            else:
                # If no document embedding, include by default
                matching_sources[source_id] = {
                    "score": 0.7,  # Default score
                    "source_name": source_name,
                    "matching_sections": []
                }
            
            # Check section-level matches
            section_embeddings = source.get("section_embeddings", {})
            for section, embedding in section_embeddings.items():
                # Check similarity with all query components
                section_similarities = [cosine_similarity([component_emb], [embedding])[0][0] 
                                       for component_emb in query_embeddings]
                max_similarity = max(section_similarities)
                
                if max_similarity > 0.6:  # Section is relevant
                    if source_id in matching_sources:
                        matching_sources[source_id]["matching_sections"].append({
                            "section": section,
                            "score": max_similarity
                        })
        
        # STEP 6: For relevant documents and sections, retrieve specific chunks
        for source_id, source_match in matching_sources.items():
            source = self.literature_sources[source_id]
            chunks = source["chunks"]
            chunk_metadata = source.get("chunk_metadata", [{}] * len(chunks))
            embeddings = source["embeddings"]
            
            # Track matches for this source
            source_matches = []
            
            # Find chunks in matching sections first
            matching_sections = {s["section"] for s in source_match["matching_sections"]}
            
            # Hybrid search: combine semantic + keyword
            for i, chunk in enumerate(chunks):
                if i >= len(embeddings) or i >= len(chunk_metadata):
                    continue
                
                chunk_section = chunk_metadata[i].get("section", "")
                chunk_embedding = embeddings[i]
                
                # Higher priority if in matching section
                section_boost = 0.1 if chunk_section in matching_sections else 0
                
                # Semantic search score - check against all query components
                semantic_similarities = [cosine_similarity([component_emb], [chunk_embedding])[0][0] 
                                      for component_emb in query_embeddings]
                semantic_score = max(semantic_similarities)
                
                # Keyword search score if using hybrid mode
                keyword_score = 0
                if retrieval_mode in ["keyword", "hybrid"]:
                    # Count exact keyword matches
                    chunk_lower = chunk.lower()
                    query_terms = set(w.lower() for w in query.split() if len(w) > 3)
                    matched_terms = sum(1 for term in query_terms if term in chunk_lower)
                    keyword_score = matched_terms / max(1, len(query_terms)) * 0.8
                
                # Combine scores based on retrieval mode
                if retrieval_mode == "semantic":
                    final_score = semantic_score + section_boost
                elif retrieval_mode == "keyword":
                    final_score = keyword_score + section_boost
                else:  # hybrid
                    final_score = (semantic_score * 0.7) + (keyword_score * 0.3) + section_boost
                
                # Only keep if score is high enough
                if final_score > 0.5:
                    source_matches.append({
                        "source_id": source_id,
                        "source_name": source_match["source_name"],
                        "chunk_index": i,
                        "text": chunk,
                        "score": final_score,
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_score,
                        "section": chunk_section,
                        "metadata": chunk_metadata[i]
                    })
            
            # Sort matches for this source
            source_matches.sort(key=lambda x: x["score"], reverse=True)
            all_matches.extend(source_matches)
            
            # Track stats
            retrieval_stats["matches_by_source"][source_id] = {
                "total_chunks": len(chunks),
                "matching_chunks": len(source_matches),
                "top_score": source_matches[0]["score"] if source_matches else 0
            }
        
        # STEP 7: Check taxonomy sources
        if sources is None or any(s in self.taxonomy_sources for s in sources):
            for source_id, source in self.taxonomy_sources.items():
                if sources is not None and source_id not in sources:
                    continue
                
                # Extract relevant taxonomy info for the query
                relevance_score = 0
                taxonomy_info = f"Taxonomy: {source['name']}\n"
                
                # Simple keyword matching for taxonomy data
                if "categories" in source["content"]:
                    query_terms = set(w.lower() for w in query.split() if len(w) > 3)
                    matching_categories = []
                    
                    for category in source["content"]["categories"]:
                        category_name = category.get("name", "").lower()
                        if any(term in category_name for term in query_terms):
                            matching_categories.append(category)
                            relevance_score = 0.7  # Fixed score for matching categories
                    
                    if matching_categories:
                        # Include matched categories
                        taxonomy_info += "Matching Categories: " + ", ".join([c.get("name", "") for c in matching_categories[:5]]) + "\n"
                        all_matches.append({
                            "source_id": source_id,
                            "source_name": source["name"],
                            "text": taxonomy_info,
                            "score": relevance_score,
                            "semantic_score": 0,
                            "keyword_score": relevance_score,
                            "section": "Taxonomy",
                            "metadata": {"type": "taxonomy", "matches": len(matching_categories)}
                        })
        
        # STEP 8: Check custom sources
        if sources is None or any(s in self.custom_sources for s in sources):
            for source_id, source in self.custom_sources.items():
                if sources is not None and source_id not in sources:
                    continue
                
                custom_text = source["content"]
                
                # Compute similarity for custom text
                try:
                    inputs = self.embedding_model["tokenizer"]([custom_text], padding=True, truncation=True, 
                                                               max_length=512, return_tensors="pt")
                    with torch.no_grad():
                        model_output = self.embedding_model["model"](**inputs)
                        attention_mask = inputs['attention_mask']
                        token_embeddings = model_output[0]
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        custom_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        custom_embedding = custom_embedding[0].numpy()
                    
                    semantic_score = cosine_similarity([primary_query_embedding], [custom_embedding])[0][0]
                    
                    # Keyword matching for hybrid approach
                    keyword_score = 0
                    if retrieval_mode in ["keyword", "hybrid"]:
                        custom_lower = custom_text.lower()
                        query_terms = set(w.lower() for w in query.split() if len(w) > 3)
                        matched_terms = sum(1 for term in query_terms if term in custom_lower)
                        keyword_score = matched_terms / max(1, len(query_terms)) * 0.8
                    
                    # Combine scores
                    if retrieval_mode == "semantic":
                        final_score = semantic_score
                    elif retrieval_mode == "keyword":
                        final_score = keyword_score
                    else:  # hybrid
                        final_score = (semantic_score * 0.7) + (keyword_score * 0.3)
                    
                    if final_score > 0.5:
                        # Truncate very long content
                        display_text = custom_text
                        if len(display_text) > 1500:
                            # Find a good breaking point
                            break_point = display_text[:1500].rfind('.')
                            if break_point > 0:
                                display_text = display_text[:break_point+1] + "..."
                            else:
                                display_text = display_text[:1500] + "..."
                        
                        all_matches.append({
                            "source_id": source_id,
                            "source_name": source["name"],
                            "text": display_text,
                            "score": final_score,
                            "semantic_score": semantic_score,
                            "keyword_score": keyword_score,
                            "section": "Custom Knowledge",
                            "metadata": {"type": "custom"}
                        })
                except Exception as e:
                    # If embedding fails, still try keyword matching
                    if retrieval_mode in ["keyword", "hybrid"]:
                        custom_lower = custom_text.lower()
                        query_terms = set(w.lower() for w in query.split() if len(w) > 3)
                        matched_terms = sum(1 for term in query_terms if term in custom_lower)
                        keyword_score = matched_terms / max(1, len(query_terms)) * 0.8
                        
                        if keyword_score > 0.5:
                            # Truncate very long content
                            display_text = custom_text[:1500] + "..." if len(custom_text) > 1500 else custom_text
                            
                            all_matches.append({
                                "source_id": source_id,
                                "source_name": source["name"],
                                "text": display_text,
                                "score": keyword_score,
                                "semantic_score": 0,
                                "keyword_score": keyword_score,
                                "section": "Custom Knowledge",
                                "metadata": {"type": "custom", "fallback": "keyword_only"}
                            })
        
        # STEP 9: Sort by score, apply re-ranking, and deduplicate
        if all_matches:
            # First sort by score
            all_matches.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply re-ranking to promote diversity
            results = self._rerank_results(all_matches, max_results)
        
        # Record timing
        retrieval_stats["timing"]["end"] = time.time()
        retrieval_stats["timing"]["duration"] = retrieval_stats["timing"]["end"] - retrieval_stats["timing"]["start"]
        retrieval_stats["results_count"] = len(results)
        
        # Store stats
        self.retrieval_stats[query] = retrieval_stats
        
        return results
    
    def _decompose_query(self, query):
        """Break down a complex query into component parts for better retrieval"""
        # Start with the original query
        components = [query]
        
        # If query is very long, break it down
        if len(query.split()) > 15:
            # Try to split by question marks
            parts = [p.strip() for p in query.split('?') if p.strip()]
            if len(parts) > 1:
                # Add each question as a component
                for part in parts:
                    if part not in components and len(part.split()) > 3:
                        components.append(part + '?')
            
            # Extract key noun phrases
            try:
                # Very basic noun phrase extraction
                # In a real implementation, you might use a proper NLP library
                words = query.lower().split()
                # Filter out stop words and short words
                filtered_words = [w for w in words if w not in stopwords.words('english') and len(w) > 3]
                # Add most frequent non-stop words as components
                word_counts = Counter(filtered_words)
                top_words = [word for word, count in word_counts.most_common(3) if count > 1]
                for word in top_words:
                    if word not in components:
                        components.append(word)
            except:
                # Fallback if NLP processing fails
                pass
        
        return components
    
    def _rerank_results(self, matches, max_results):
        """
        Re-rank results to promote diversity and quality
        
        Args:
            matches: List of matched passages sorted by score
            max_results: Maximum number of results to return
            
        Returns:
            List of re-ranked results
        """
        if not matches:
            return []
        
        # Start with the top result
        results = [matches[0]]
        matches_left = matches[1:]
        
        # Set to track content already included to avoid near-duplicates
        included_content = set([self._get_content_signature(matches[0]["text"])])
        
        # Track which sources we've already included
        included_sources = {matches[0]["source_id"]: 1}
        
        # Add remaining results with diversity in mind
        while len(results) < max_results and matches_left:
            best_match = None
            best_score = -1
            best_idx = -1
            
            for i, match in enumerate(matches_left):
                # Skip if too similar to already included content
                content_sig = self._get_content_signature(match["text"])
                if content_sig in included_content:
                    continue
                
                # Calculate diversity score
                source_count = included_sources.get(match["source_id"], 0)
                source_diversity_factor = 1.0 if source_count == 0 else (0.7 ** source_count)
                
                # Adjusted score considering diversity
                adjusted_score = match["score"] * source_diversity_factor
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_match = match
                    best_idx = i
            
            # If no diverse match found, just take the next highest scoring
            if best_match is None and matches_left:
                best_match = matches_left[0]
                best_idx = 0
            
            # Break if no more matches
            if best_match is None:
                break
            
            # Add to results
            results.append(best_match)
            included_content.add(self._get_content_signature(best_match["text"]))
            included_sources[best_match["source_id"]] = included_sources.get(best_match["source_id"], 0) + 1
            
            # Remove from candidates
            matches_left.pop(best_idx)
        
        return results
    
    def _get_content_signature(self, text):
        """Generate a signature for text to detect near-duplicates"""
        # This is a simple approach - for production, consider using more robust methods
        # like MinHash or other fingerprinting techniques
        words = text.lower().split()
        if len(words) <= 10:
            return text.lower()  # For short texts, use entire text
        
        # For longer texts, use selective words
        filtered_words = []
        for i, word in enumerate(words):
            if i % 5 == 0 and len(word) > 3:  # Sample every 5th substantial word
                filtered_words.append(word)
        
        return " ".join(filtered_words)
    
    def get_retrieval_stats(self):
        """Get statistics about retrieval operations"""
        return self.retrieval_stats
    
    def get_chunk_stats(self):
        """Get statistics about document chunking"""
        return self.chunk_stats
    
    def get_document_map(self, source_id):
        """Get the document map for a given source"""
        if source_id in self.literature_sources:
            return self.literature_sources[source_id].get("document_map", {})
        return {}
    
    def explain_retrieval(self, query, results, detailed=False):
        """Generate a human-readable explanation of retrieval results"""
        explanation = {
            "query": query,
            "results_count": len(results),
            "retrieval_strategy": "Hierarchical Hybrid Search",
            "sources_used": set(),
            "match_types": [],
            "score_range": {"min": 1.0, "max": 0.0}
        }
        
        # Gather information from results
        for result in results:
            source_id = result["source_id"]
            explanation["sources_used"].add(source_id)
            
            # Track score range
            score = result["score"]
            explanation["score_range"]["min"] = min(explanation["score_range"]["min"], score)
            explanation["score_range"]["max"] = max(explanation["score_range"]["max"], score)
            
            # Track match types
            match_type = "Unknown"
            metadata = result.get("metadata", {})
            if "type" in metadata:
                match_type = metadata["type"]
            elif "section" in result:
                section = result["section"]
                if section:
                    match_type = f"Section: {section}"
                else:
                    match_type = "Document Chunk"
            
            if match_type not in explanation["match_types"]:
                explanation["match_types"].append(match_type)
        
        # Format explanation
        if detailed:
            # Include timing and stats if available
            if query in self.retrieval_stats:
                stats = self.retrieval_stats[query]
                explanation["timing"] = stats.get("timing", {}).get("duration", 0)
                explanation["components"] = stats.get("query_components", [query])
                explanation["matches_by_source"] = stats.get("matches_by_source", {})
        
        return explanation


class LLMManager:
    """Class for managing Local LLM interactions"""
    def __init__(self, api_base="http://127.0.0.1:1234/v1"):
        self.api_base = api_base
        self.client = None
        self.available_models = []
        self.selected_model = None
    
    def connect(self):
        """Connect to the LLM server"""
        try:
            self.client = OpenAI(base_url=self.api_base, api_key="not-needed")
            return True
        except Exception as e:
            st.error(f"Error connecting to LLM server: {str(e)}")
            return False
    
    def get_available_models(self):
        """Get available models from the server"""
        if not self.client:
            if not self.connect():
                return []
        
        try:
            models_response = self.client.models.list()
            self.available_models = [model.id for model in models_response.data]
            return self.available_models
        except Exception as e:
            st.error(f"Error getting models: {str(e)}")
            return []
    
    def set_model(self, model_id):
        """Set the model to use"""
        if model_id in self.available_models:
            self.selected_model = model_id
            return True
        else:
            st.warning(f"Model {model_id} not available")
            return False
    
    def generate_completion(self, prompt, max_tokens=1000, temperature=0.7):
        """Generate text completion"""
        if not self.client:
            if not self.connect():
                return "Error: Unable to connect to LLM server"
        
        if not self.selected_model:
            available_models = self.get_available_models()
            if available_models:
                self.selected_model = available_models[0]
            else:
                return "Error: No models available"
        
        try:
            response = self.client.completions.create(
                model=self.selected_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text
        except Exception as e:
            st.error(f"Error generating completion: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_chat_completion(self, messages, max_tokens=1000, temperature=0.7):
        """Generate chat completion"""
        if not self.client:
            if not self.connect():
                return "Error: Unable to connect to LLM server"
        
        if not self.selected_model:
            available_models = self.get_available_models()
            if available_models:
                self.selected_model = available_models[0]
            else:
                return "Error: No models available"
        
        try:
            response = self.client.chat.completions.create(
                model=self.selected_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating chat completion: {str(e)}")
            return f"Error: {str(e)}"

    def generate_taxonomy_suggestion(self, element, potential_parents, context=None):
        """
        Generate a suggestion for where to place an element in the taxonomy
        
        Args:
            element: The element to place
            potential_parents: List of potential parent nodes
            context: Additional context for the LLM
            
        Returns:
            Dictionary with suggestion information
        """
        if not self.client:
            if not self.connect():
                return {"error": "Unable to connect to LLM server"}
        
        # Prepare the prompt
        system_message = """You are an AI assistant helping to build a taxonomy for software engineering effort estimation.
    Your task is to suggest where to place an element in the taxonomy structure. You will be given:
    1. Information about the element to be placed
    2. A list of potential parent nodes
    3. Any additional context that might be relevant

    Please analyze the semantics and relationships to determine the most appropriate parent node.
    Provide a clear explanation for your recommendation and a confidence score (0-100).
    """
        
        # Format the element information
        element_info = f"Element to place: {element['name']}\n"
        if 'description' in element:
            element_info += f"Description: {element['description']}\n"
        if 'type' in element:
            element_info += f"Type: {element['type']}\n"
        if 'source' in element:
            element_info += f"Source: {element['source']}\n"
        
        # Format potential parents
        parents_info = "Potential parent nodes:\n"
        for i, parent in enumerate(potential_parents):
            parents_info += f"{i+1}. {parent['name']}"
            if 'description' in parent:
                parents_info += f" - {parent['description']}"
            parents_info += "\n"
        
        # Add context if provided
        context_info = ""
        if context:
            context_info = "Additional context:\n" + context
        
        # Build the user message
        user_message = f"{element_info}\n{parents_info}\n{context_info}\n\nPlease suggest the most appropriate parent node for this element, explain your reasoning, and provide a confidence score (0-100). Please respond with your recommendation in this format: 'I recommend placing this under: [EXACT PARENT NAME]' "
        
        # Generate completion
        try:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = self.client.chat.completions.create(
                model=self.selected_model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            
            # Parse the response to extract parent suggestion and confidence
            suggested_parent = None
            confidence = 0
            explanation = response_text
            
            # NEW: First try to find the recommended pattern
            recommend_match = re.search(r"I recommend placing this under:\s*([^\n.]+)", response_text, re.IGNORECASE)
            if recommend_match:
                parent_name = recommend_match.group(1).strip()
                # Look for an exact or close match in potential_parents
                for parent in potential_parents:
                    # Check for exact match, containment, or significant word overlap
                    if (parent['name'].lower() == parent_name.lower() or 
                        parent['name'].lower() in parent_name.lower() or 
                        parent_name.lower() in parent['name'].lower()):
                        suggested_parent = parent
                        break
                        
                    # If no direct match, try word similarity approach
                    if not suggested_parent:
                        parent_words = set(parent['name'].lower().split())
                        recommendation_words = set(parent_name.lower().split())
                        
                        # If enough words match, consider it a match
                        if len(parent_words.intersection(recommendation_words)) >= max(1, len(parent_words) * 0.5):
                            suggested_parent = parent
                            break
            
            # If recommended pattern not found, try the older methods
            if not suggested_parent:
                # Try to extract parent number (e.g., "I recommend parent 2")
                parent_match = re.search(r"parent (\d+)", response_text.lower())
                if parent_match:
                    parent_num = int(parent_match.group(1))
                    if 1 <= parent_num <= len(potential_parents):
                        suggested_parent = potential_parents[parent_num-1]
                
                # If no parent number found, try to match by name
                if not suggested_parent:
                    for parent in potential_parents:
                        # Try exact match first
                        if parent['name'].lower() in response_text.lower():
                            suggested_parent = parent
                            break
                        
                        # Try partial match (split words and look for significant matches)
                        parent_words = set(parent['name'].lower().split())
                        response_words = set(response_text.lower().split())
                        
                        # If enough words match, consider it a match
                        if len(parent_words.intersection(response_words)) >= max(1, len(parent_words) * 0.5):
                            suggested_parent = parent
                            break
            
            # Default to first parent if no match found
            if not suggested_parent and potential_parents:
                suggested_parent = potential_parents[0]
            
            # Try to extract confidence score
            confidence_match = re.search(r"confidence(?:\s+score)?(?:\s+of)?(?:\s+is)?(?:\s*:)?\s*(\d+)", response_text.lower())
            if confidence_match:
                confidence = int(confidence_match.group(1))
                confidence = min(100, max(0, confidence))  # Ensure between 0-100
            
            return {
                "parent": suggested_parent,
                "confidence": confidence,
                "explanation": explanation
            }
        
        except Exception as e:
            st.error(f"Error generating taxonomy suggestion: {str(e)}")
            return {"error": str(e)}
class TaxonomyBuilder:
    """Class for building and managing the taxonomy structure"""
    def __init__(self):
        self.nodes = []  # List of node dictionaries
        self.edges = []  # List of edge dictionaries
        self.next_node_id = 1
        
    def initialize_taxonomy(self, root_name="Effort Estimation Taxonomy"):
        """Initialize a new taxonomy with a root node"""
        self.nodes = [
            {
                "id": "root",
                "name": root_name,
                "type": "root",
                "description": "Root node of the taxonomy",
                "source": "user-created"
            }
        ]
        self.edges = []
        self.next_node_id = 1
        return "root"
    
    def add_node(self, name, node_type="category", parent_id="root", description=None, source=None, original_id=None, metadata=None):
        """Add a node to the taxonomy"""
        node_id = f"n{self.next_node_id}"
        self.next_node_id += 1
        
        # Create the node
        node = {
            "id": node_id,
            "name": name,
            "type": node_type,
            "description": description,
            "source": source,
            "original_id": original_id
        }
        
        # Add additional metadata if provided
        if metadata:
            node.update(metadata)
        
        # Add to nodes list
        self.nodes.append(node)
        
        # Create edge to parent
        edge = {
            "source": parent_id,
            "target": node_id
        }
        self.edges.append(edge)
        
        return node_id
    
    def update_node(self, node_id, updates):
        """Update a node's properties"""
        for i, node in enumerate(self.nodes):
            if node["id"] == node_id:
                self.nodes[i].update(updates)
                return True
        return False
    
    def delete_node(self, node_id):
        """Delete a node and its descendants"""
        # Can't delete root
        if node_id == "root":
            return False
        
        # Get all descendants
        descendants = self.get_descendants(node_id)
        descendant_ids = [node_id] + [d["id"] for d in descendants]
        
        # Remove nodes
        self.nodes = [n for n in self.nodes if n["id"] not in descendant_ids]
        
        # Remove edges
        self.edges = [e for e in self.edges if e["source"] not in descendant_ids and e["target"] not in descendant_ids]
        
        return True
    
    def move_node(self, node_id, new_parent_id):
        """Move a node to a new parent"""
        # Can't move root
        if node_id == "root":
            return False
        
        # Check if new parent exists
        if not any(n["id"] == new_parent_id for n in self.nodes):
            return False
        
        # Check if move would create a cycle
        if new_parent_id == node_id or self.is_descendant(node_id, new_parent_id):
            return False
        
        # Find and update the edge
        for i, edge in enumerate(self.edges):
            if edge["target"] == node_id:
                self.edges[i]["source"] = new_parent_id
                return True
        
        # Edge not found, create it
        self.edges.append({
            "source": new_parent_id,
            "target": node_id
        })
        
        return True
    
    def get_descendants(self, node_id):
        """Get all descendants of a node"""
        descendants = []
        
        # Get immediate children
        children = [n for n in self.nodes if any(e["source"] == node_id and e["target"] == n["id"] for e in self.edges)]
        
        # Add children and their descendants
        for child in children:
            descendants.append(child)
            descendants.extend(self.get_descendants(child["id"]))
        
        return descendants
    
    def is_descendant(self, ancestor_id, node_id):
        """Check if node_id is a descendant of ancestor_id"""
        descendants = self.get_descendants(ancestor_id)
        return any(d["id"] == node_id for d in descendants)
    
    def get_parent(self, node_id):
        """Get the parent of a node"""
        for edge in self.edges:
            if edge["target"] == node_id:
                return next((n for n in self.nodes if n["id"] == edge["source"]), None)
        return None
    
    def get_children(self, node_id):
        """Get the children of a node"""
        return [n for n in self.nodes if any(e["source"] == node_id and e["target"] == n["id"] for e in self.edges)]
    
    def get_node(self, node_id):
        """Get a node by ID"""
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        return None
    
    def export_taxonomy(self, format="json"):
        """Export the taxonomy in the specified format"""
        if format == "json":
            return json.dumps({
                "nodes": self.nodes,
                "edges": self.edges,
                "metadata": {
                    "created_date": datetime.now().isoformat(),
                    "node_count": len(self.nodes),
                    "edge_count": len(self.edges)
                }
            }, indent=2, cls=NumpyEncoder)  # Add the cls parameter here
        
        elif format == "csv":
            # Create a buffer to store CSV data
            output = io.StringIO()
            
            # Write nodes CSV
            nodes_df = pd.DataFrame(self.nodes)
            nodes_csv = nodes_df.to_csv(index=False)
            output.write("NODES CSV:\n")
            output.write(nodes_csv)
            output.write("\n\nEDGES CSV:\n")
            
            # Write edges CSV
            edges_df = pd.DataFrame(self.edges)
            edges_csv = edges_df.to_csv(index=False)
            output.write(edges_csv)
            
            return output.getvalue()
        
        else:
            return "Unsupported format"
    
    def visualize_taxonomy(self, selected_node=None, hover_node=None):
        """Create a visualization of the taxonomy structure using Plotly"""
        # Create a graph using NetworkX
        G = nx.DiGraph()
        
        # Add nodes with their properties
        for node in self.nodes:
            G.add_node(
                node["id"],
                name=node["name"],
                type=node["type"],
                description=node.get("description", ""),
                selected=(node["id"] == selected_node),
                hover=(node["id"] == hover_node)
            )
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge["source"], edge["target"])
        
        # Create hierarchical layout
        try:
            if HAS_GRAPHVIZ:
                pos = graphviz_layout(G, prog="dot")
            else:
                # Create a custom hierarchical layout
                pos = {}
                
                # Start with the root node
                root = "root"
                pos[root] = (0, 0)
                
                # BFS to position nodes level by level
                levels = {root: 0}
                width_per_level = {}
                current_width = {}
                
                # First pass: determine levels
                queue = [root]
                visited = {root}
                
                while queue:
                    node = queue.pop(0)
                    level = levels[node]
                    
                    # Initialize level counter if not exists
                    if level not in width_per_level:
                        width_per_level[level] = 0
                        current_width[level] = 0
                    
                    # Increment width for this level
                    width_per_level[level] += 1
                    
                    # Process all children
                    children = list(G.successors(node))
                    for child in children:
                        if child not in visited:
                            visited.add(child)
                            queue.append(child)
                            levels[child] = level + 1
                
                # Second pass: position nodes by level
                for node, level in levels.items():
                    if node == root:
                        continue
                    
                    # Calculate x position based on the node's position in level
                    x = (current_width[level] - width_per_level[level]/2) * 100
                    current_width[level] += 1
                    
                    # Set y position based on level (negative to make root at top)
                    y = -level * 100
                    
                    pos[node] = (x, y)
        
        except Exception as e:
            st.error(f"Error creating layout: {str(e)}")
            # Fall back to spring layout
            pos = nx.spring_layout(G)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Define node colors by type and state
        base_colors = {
            "root": "#3498db",      # Blue
            "category": "#2ecc71",  # Green
            "element": "#e74c3c",   # Red
            "characteristic": "#9b59b6"  # Purple
        }
        
        # Add edges as lines
        edge_x = []
        edge_y = []
        
        for edge in self.edges:
            if edge["source"] in pos and edge["target"] in pos:
                x0, y0 = pos[edge["source"]]
                x1, y1 = pos[edge["target"]]
                
                # Create orthogonal edges (right angle)
                if y0 != y1:  # Only for edges that change levels
                    # First segment: horizontal move
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y0, None])
                    
                    # Second segment: vertical move
                    edge_x.extend([x1, x1, None])
                    edge_y.extend([y0, y1, None])
                else:
                    # Direct line for same-level connections
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes by type
        for node_type in ["root", "category", "element", "characteristic"]:
            # Get nodes of this type
            type_nodes = [n for n in self.nodes if n.get("type", "") == node_type]
            
            if not type_nodes:
                continue
            
            # Prepare node data
            node_x = []
            node_y = []
            node_text = []
            node_hover = []
            node_colors = []
            node_sizes = []
            node_ids = []
            
            for node in type_nodes:
                if node["id"] not in pos:
                    continue
                
                x, y = pos[node["id"]]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node["name"])
                
                # Hover text with more details
                hover_text = f"<b>{node['name']}</b><br>"
                if "description" in node and node["description"]:
                    hover_text += f"Description: {node['description']}<br>"
                if "source" in node and node["source"]:
                    hover_text += f"Source: {node['source']}<br>"
                node_hover.append(hover_text)
                
                # Set color based on selection/hover state
                if node["id"] == selected_node:
                    node_colors.append('#ff9900')  # Orange for selected
                elif node["id"] == hover_node:
                    node_colors.append('#f39c12')  # Light orange for hover
                else:
                    node_colors.append(base_colors.get(node_type, '#95a5a6'))  # Default color
                
                # Set size based on type
                size_map = {"root": 25, "category": 20, "element": 15, "characteristic": 15}
                node_sizes.append(size_map.get(node_type, 15))
                
                # Store node ID for interactivity
                node_ids.append(node["id"])
            
            # Add trace for this node type
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=1, color='white')
                ),
                text=node_text,
                textposition="top center",
                hovertext=node_hover,
                hoverinfo='text',
                name=node_type,
                customdata=node_ids  # Store node IDs for callbacks
            ))
        
        # Update layout
        fig.update_layout(
            title="Taxonomy Structure",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            height=700,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig


#############################################
# HELPER FUNCTIONS
#############################################

def initialize_state():
    """Initialize session state variables"""
    if 'embedding_calculator' not in st.session_state:
        st.session_state.embedding_calculator = EmbeddingCalculator()
    
    if 'categories' not in st.session_state:
        st.session_state.categories = []
    
    if 'characteristics' not in st.session_state:
        st.session_state.characteristics = []
    
    if 'category_clusters' not in st.session_state:
        st.session_state.category_clusters = None
    
    if 'characteristic_clusters' not in st.session_state:
        st.session_state.characteristic_clusters = None
    
    if 'taxonomy_graph' not in st.session_state:
        st.session_state.taxonomy_graph = None
    
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Ready"
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "Step 1: Load Data"
    
    # Add RAG manager
    if 'rag_manager' not in st.session_state:
        st.session_state.rag_manager = RAGManager()
    
    # Add LLM manager
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = LLMManager()
    
    # Add taxonomy builder
    if 'taxonomy_builder' not in st.session_state:
        st.session_state.taxonomy_builder = TaxonomyBuilder()
        # Initialize with a root node
        st.session_state.taxonomy_builder.initialize_taxonomy()
    
    # Building interface state
    if 'selected_element' not in st.session_state:
        st.session_state.selected_element = None
    
    if 'selected_node' not in st.session_state:
        st.session_state.selected_node = None
    
    if 'hover_node' not in st.session_state:
        st.session_state.hover_node = None
    
    if 'unassigned_elements' not in st.session_state:
        st.session_state.unassigned_elements = []
    
    # Track user context
    if 'user_context' not in st.session_state:
        st.session_state.user_context = ""


def load_data(uploaded_file):
    """Load taxonomy data from a JSON file"""
    try:
        content = uploaded_file.read().decode('utf-8')
        data = json.loads(content)
        
        # Extract categories and characteristics
        categories = data.get('categories', [])
        characteristics = data.get('characteristics', [])
        
        # Update session state
        st.session_state.categories = categories
        st.session_state.characteristics = characteristics
        
        # Create unassigned elements list for taxonomy building
        st.session_state.unassigned_elements = []
        for category in categories:
            st.session_state.unassigned_elements.append({
                "id": category["id"],
                "name": category["name"],
                "type": "category",
                "source": category.get("source", "Unknown"),
                "description": category.get("description", "")
            })
        
        for characteristic in characteristics:
            st.session_state.unassigned_elements.append({
                "id": characteristic["id"],
                "name": characteristic["name"],
                "type": "characteristic",
                "source": characteristic.get("source", "Unknown"),
                "description": characteristic.get("description", "")
            })
        
        # Success message
        st.success(f"Loaded {len(categories)} categories and {len(characteristics)} characteristics")
        
        # Show preview - side by side instead of tabs
        st.subheader("Data Preview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Categories (Inner Nodes)**")
            if categories:
                st.dataframe(pd.DataFrame(categories), height=400)
            else:
                st.write("No categories found in the data")
        
        with col2:
            st.write("**Characteristics (Leaves)**")
            if characteristics:
                st.dataframe(pd.DataFrame(characteristics), height=400)
            else:
                st.write("No characteristics found in the data")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return False
    
    return True


def compute_embeddings():
    """Compute embeddings for all elements"""
    # Combine categories and characteristics for embedding calculation
    elements = st.session_state.categories + st.session_state.characteristics
    
    if not elements:
        st.error("No elements to process. Please load data first.")
        return False
    
    # Compute embeddings
    with st.spinner("Computing embeddings..."):
        success = st.session_state.embedding_calculator.compute_embeddings(
            elements,
            lambda msg: st.session_state.update(status_message=msg)
        )
    
    if success:
        st.success("Embeddings computed successfully!")
        
        # The issue with the visualization could be happening because of parent containers
        # Let's modify how we display the visualization for full width
                
        # Display embeddings information
        st.markdown("## Embeddings Overview")
        embedding_dim = next(iter(st.session_state.embedding_calculator.embeddings.values())).shape[0]
        st.write(f"Generated {len(st.session_state.embedding_calculator.embeddings)} embeddings with dimension {embedding_dim}")
        
        # Create visual embeddings plot with t-SNE for better visualization
        st.markdown("## Improved Embedding Visualization")
        
        # Get all embeddings
        embedding_matrix = np.array(list(st.session_state.embedding_calculator.embeddings.values()))
        element_ids = list(st.session_state.embedding_calculator.embeddings.keys())
        
        # Apply t-SNE for better visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(embedding_matrix)-1)))
        embeddings_2d = tsne.fit_transform(embedding_matrix)
        
        # Create a DataFrame for the plot
        plot_data = pd.DataFrame({
            'ID': element_ids,
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'Type': [st.session_state.embedding_calculator.element_map.get(id, {"type": "Unknown"})['type'] for id in element_ids],
            'Name': [st.session_state.embedding_calculator.element_map.get(id, {"name": "Unknown"})['name'] for id in element_ids],
            'Source': [st.session_state.embedding_calculator.element_map.get(id, {"source": "Unknown"}).get('source', "Unknown") for id in element_ids]
        })
        
        # Use Plotly for interactive visualization - explicitly making it large
        fig = px.scatter(
            plot_data, x='x', y='y', color='Type', text='Name',
            hover_data=['Name', 'Source', 'Type'],
            title='t-SNE Projection of Element Embeddings',
            # Set an explicit width larger than the typical container width
            width=1200,
            height=700
        )
        
        # Customize appearance
        fig.update_traces(
            marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
            textposition='top center',
            textfont=dict(size=10)
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title=None, showticklabels=False),
            yaxis=dict(title=None, showticklabels=False),
            # Ensure no margins constrain the visualization
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Show the plot - make sure it uses the full container width
        st.plotly_chart(fig, use_container_width=True)
        
        return True
    else:
        st.error("Failed to compute embeddings")
        return False


def cluster_categories(min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.5):
    """Cluster categories using HDBSCAN"""
    # Get category IDs
    category_ids = [cat['id'] for cat in st.session_state.categories]
    
    if not category_ids:
        st.warning("No categories to cluster")
        return False
    
    # Perform clustering
    with st.spinner("Clustering categories..."):
        element_ids, cluster_labels, clusterer = st.session_state.embedding_calculator.perform_hdbscan_clustering(
            category_ids,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            status_callback=lambda msg: st.session_state.update(status_message=msg)
        )
    
    if element_ids is None:
        st.error("Clustering failed")
        return False
    
    # Store results
    st.session_state.category_clusters = {
        'element_ids': element_ids,
        'cluster_labels': cluster_labels,
        'clusterer': clusterer,
        'probabilities': clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
    }
    
    # Success message
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    st.success(f"Categories clustered into {n_clusters} groups with {n_noise} unclustered items")
    
    # Display cluster analysis with improved visualization
    st.subheader("Category Clustering Analysis")
    
    # Create tabs for different visualizations
    cluster_tabs = st.tabs(["Cluster Groups", "2D Visualization", "3D Visualization"])
    
    with cluster_tabs[0]:
        st.write("### Category Clusters")
        st.write("Elements grouped by semantic similarity:")
        
        # Create a mapping from cluster label to element names
        category_map = {cat['id']: cat['name'] for cat in st.session_state.categories}
        cluster_to_elements = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in cluster_to_elements:
                cluster_to_elements[cluster_label] = []
            
            element_id = element_ids[i]
            element_name = category_map.get(element_id, element_id)
            element_source = next((cat.get('source', 'Unknown') for cat in st.session_state.categories if cat['id'] == element_id), 'Unknown')
            element_probability = clusterer.probabilities_[i] if hasattr(clusterer, 'probabilities_') else None
            
            cluster_to_elements[cluster_label].append({
                'id': element_id,
                'name': element_name,
                'source': element_source,
                'probability': element_probability
            })
        
        # Display each cluster with expanders for better organization
        for cluster_label, elements in sorted(cluster_to_elements.items()):
            # Create a name for the cluster
            if cluster_label == -1:
                cluster_title = f"Unclustered Points ({len(elements)})"
            elif len(elements) == 1:
                cluster_title = f"Cluster {cluster_label}: {elements[0]['name']}"
            else:
                # Take first element and show count
                cluster_title = f"Cluster {cluster_label}: {elements[0]['name']} and {len(elements) - 1} others"
            
            with st.expander(cluster_title, expanded=True):
                # Create a colorful dataframe
                elements_df = pd.DataFrame([{
                    'Name': e['name'],
                    'Source': e['source'],
                    'ID': e['id'],
                    'Probability': f"{e['probability']:.2f}" if e['probability'] is not None else "N/A"
                } for e in elements])
                
                st.dataframe(elements_df, use_container_width=True)
    
    with cluster_tabs[1]:
        st.write("""
        ### 2D Cluster Visualization

        This visualization shows how categories form clusters in semantic space.

        **About the axes:** The X and Y coordinates don't represent specific features but are calculated by t-SNE to preserve semantic relationships from high-dimensional space. 

        **What to look for:**
        - **Proximity**: Points close together represent semantically similar concepts
        - **Clusters**: Natural groupings suggest related taxonomic categories
        - **Outliers**: Distant points represent conceptually unique elements

        The absolute position on any axis has no inherent meaning; what matters is the relative distance between points.
        """)
        
        # Apply t-SNE to get 2D projection of embeddings
        embedding_matrix, _ = st.session_state.embedding_calculator.get_embedding_matrix(category_ids)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(embedding_matrix)-1)))
        embeddings_2d = tsne.fit_transform(embedding_matrix)
        
        # Create DataFrame for visualization
        plot_data = pd.DataFrame({
            'ID': element_ids,
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'Name': [category_map.get(id, id) for id in element_ids],
            'Cluster': [f"Cluster {label}" if label != -1 else "Noise" for label in cluster_labels]
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_data, x='x', y='y',
            color='Cluster', text='Name', 
            hover_data=['Name', 'Cluster'],
            width=None, height=600
        )
        
        # Update marker appearance
        fig.update_traces(
            marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
            textposition='top center'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with cluster_tabs[2]:
        st.write("### 3D Cluster Visualization")
        st.write("This interactive 3D visualization shows how categories form clusters in semantic space.")
        
        # Apply PCA to get 3D projection of embeddings
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embedding_matrix)
        
        # Create DataFrame for visualization
        plot3d_data = pd.DataFrame({
            'ID': element_ids,
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'Name': [category_map.get(id, id) for id in element_ids],
            'Cluster': [f"Cluster {label}" if label != -1 else "Noise" for label in cluster_labels]
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            plot3d_data, x='x', y='y', z='z',
            color='Cluster', text='Name', 
            hover_data=['Name', 'Cluster'],
            width=None, height=600
        )
        
        # Update marker appearance
        fig.update_traces(
            marker=dict(size=5, opacity=0.8),
            selector=dict(mode='markers+text')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    return True



def cluster_characteristics(min_cluster_size=3, min_samples=2, cluster_selection_epsilon=0.5):
    """Cluster characteristics using HDBSCAN"""
    # Get characteristic IDs
    characteristic_ids = [char['id'] for char in st.session_state.characteristics]
    
    if not characteristic_ids:
        st.warning("No characteristics to cluster")
        return False
    
    # Perform clustering
    with st.spinner("Clustering characteristics..."):
        element_ids, cluster_labels, clusterer = st.session_state.embedding_calculator.perform_hdbscan_clustering(
            characteristic_ids,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            status_callback=lambda msg: st.session_state.update(status_message=msg)
        )
    
    if element_ids is None:
        st.error("Clustering failed")
        return False
    
    # Store results
    st.session_state.characteristic_clusters = {
        'element_ids': element_ids,
        'cluster_labels': cluster_labels,
        'clusterer': clusterer,
        'probabilities': clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
    }
    
    # Success message
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    st.success(f"Characteristics clustered into {n_clusters} groups with {n_noise} unclustered items")
    
    # Display cluster analysis with improved visualization
    st.subheader("Characteristic Clustering Analysis")
    
    # Create tabs for different visualizations
    cluster_tabs = st.tabs(["Cluster Groups", "2D Visualization", "3D Visualization"])
    
    with cluster_tabs[0]:
        st.write("### Characteristic Clusters")
        st.write("Elements grouped by semantic similarity:")
        
        # Create a mapping from cluster label to element names
        characteristic_map = {char['id']: char['name'] for char in st.session_state.characteristics}
        cluster_to_elements = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in cluster_to_elements:
                cluster_to_elements[cluster_label] = []
            
            element_id = element_ids[i]
            element_name = characteristic_map.get(element_id, element_id)
            element_source = next((char.get('source', 'Unknown') for char in st.session_state.characteristics if char['id'] == element_id), 'Unknown')
            element_probability = clusterer.probabilities_[i] if hasattr(clusterer, 'probabilities_') else None
            
            cluster_to_elements[cluster_label].append({
                'id': element_id,
                'name': element_name,
                'source': element_source,
                'probability': element_probability
            })
        
        # Display each cluster with expanders for better organization
        for cluster_label, elements in sorted(cluster_to_elements.items()):
            # Create a name for the cluster
            if cluster_label == -1:
                cluster_title = f"Unclustered Points ({len(elements)})"
            elif len(elements) == 1:
                cluster_title = f"Cluster {cluster_label}: {elements[0]['name']}"
            else:
                # Take first element and show count
                cluster_title = f"Cluster {cluster_label}: {elements[0]['name']} and {len(elements) - 1} others"
            
            with st.expander(cluster_title, expanded=True):
                # Create a colorful dataframe
                elements_df = pd.DataFrame([{
                    'Name': e['name'],
                    'Source': e['source'],
                    'ID': e['id'],
                    'Probability': f"{e['probability']:.2f}" if e['probability'] is not None else "N/A"
                } for e in elements])
                
                st.dataframe(elements_df, use_container_width=True)
    
    with cluster_tabs[1]:
        st.write("""
        ### 2D Cluster Visualization

        This visualization shows how categories form clusters in semantic space.

        **About the axes:** The X and Y coordinates don't represent specific features but are calculated by t-SNE to preserve semantic relationships from high-dimensional space. 

        **What to look for:**
        - **Proximity**: Points close together represent semantically similar concepts
        - **Clusters**: Natural groupings suggest related taxonomic categories
        - **Outliers**: Distant points represent conceptually unique elements

        The absolute position on any axis has no inherent meaning; what matters is the relative distance between points.
        """)
        
        # Apply t-SNE to get 2D projection of embeddings
        embedding_matrix, _ = st.session_state.embedding_calculator.get_embedding_matrix(characteristic_ids)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(embedding_matrix)-1)))
        embeddings_2d = tsne.fit_transform(embedding_matrix)
        
        # Create DataFrame for visualization
        plot_data = pd.DataFrame({
            'ID': element_ids,
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'Name': [characteristic_map.get(id, id) for id in element_ids],
            'Cluster': [f"Cluster {label}" if label != -1 else "Noise" for label in cluster_labels]
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_data, x='x', y='y',
            color='Cluster', text='Name', 
            hover_data=['Name', 'Cluster'],
            width=None, height=600
        )
        
        # Update marker appearance
        fig.update_traces(
            marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
            textposition='top center'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with cluster_tabs[2]:
        st.write("### 3D Cluster Visualization")
        st.write("This interactive 3D visualization shows how characteristics form clusters in semantic space.")
        
        # Apply PCA to get 3D projection of embeddings
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embedding_matrix)
        
        # Create DataFrame for visualization
        plot3d_data = pd.DataFrame({
            'ID': element_ids,
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'Name': [characteristic_map.get(id, id) for id in element_ids],
            'Cluster': [f"Cluster {label}" if label != -1 else "Noise" for label in cluster_labels]
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            plot3d_data, x='x', y='y', z='z',
            color='Cluster', text='Name', 
            hover_data=['Name', 'Cluster'],
            width=None, height=600
        )
        
        # Update marker appearance
        fig.update_traces(
            marker=dict(size=5, opacity=0.8),
            selector=dict(mode='markers+text')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    return True

def test_babelnet_api():
    """Test function for BabelNet API connection"""
    babelnet_key = st.session_state.get('temp_babelnet_key', '')
    test_term = st.session_state.get('test_term', 'effort estimation')
    
    if not babelnet_key:
        st.warning("Please enter an API key first")
        return
    
    with st.spinner("Querying BabelNet..."):
        # Make request directly instead of using the method
        url = "https://babelnet.io/v5/getSynsetIds"
        params = {
            "lemma": test_term,
            "searchLang": "EN",
            "key": babelnet_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Check if it's an error message
                    if isinstance(data, dict) and 'message' in data:
                        st.error(f"BabelNet API error: {data['message']}")
                    # Check if it's a valid list response
                    elif isinstance(data, list):
                        st.success("BabelNet query successful!")
                        st.json(data)
                        # Cache the successful response
                        st.session_state.rag_manager.babelnet_cache[test_term] = data
                    else:
                        st.warning(f"BabelNet returned unexpected format")
                        st.json(data)
                except json.JSONDecodeError:
                    st.error("BabelNet returned invalid JSON")
                    st.text(response.text[:500])
            else:
                st.error(f"BabelNet API returned status {response.status_code}")
                st.text(response.text[:500])
        except Exception as e:
            st.error(f"Error querying BabelNet: {str(e)}")

def select_element(element_id):
    """
    Select an unassigned element and show cluster information
    to guide placement decisions
    """
    st.session_state.selected_element = element_id
    
    # Generate relevant context for this element
    element = next((e for e in st.session_state.unassigned_elements if e["id"] == element_id), None)
    if element:
        # Get potential parent nodes for AI suggestions
        potential_parents = st.session_state.taxonomy_builder.nodes
        
        # Get similar elements
        similar_elements = st.session_state.embedding_calculator.get_similar_elements(element_id, top_n=5)
        
        # Get cluster information for this element
        cluster_info = get_element_cluster_info(element_id)
        
        # Update display information in session state
        st.session_state.element_similar_elements = similar_elements
        st.session_state.element_potential_parents = potential_parents
        st.session_state.element_cluster_info = cluster_info


def add_element_to_taxonomy(element_id, parent_id):
    """Add an element to the taxonomy"""
    # Find the element in unassigned elements
    element = next((e for e in st.session_state.unassigned_elements if e["id"] == element_id), None)
    if element:
        # Add to taxonomy
        node_id = st.session_state.taxonomy_builder.add_node(
            name=element["name"],
            node_type=element["type"],
            parent_id=parent_id,
            description=element.get("description", ""),
            source=element.get("source", ""),
            original_id=element["id"]
        )
        
        # Remove from unassigned elements
        st.session_state.unassigned_elements = [e for e in st.session_state.unassigned_elements if e["id"] != element_id]
        
        # Clear selection
        st.session_state.selected_element = None
        
        return node_id
    
    return None


def get_element_cluster_info(element_id):
    """
    Get information about which cluster an element belongs to
    and what other elements are in the same cluster
    """
    # Default response if no cluster info found
    cluster_info = {
        "in_cluster": False,
        "cluster_label": None,
        "cluster_elements": []
    }
    
    # Check if this is a category
    if 'category_clusters' in st.session_state and st.session_state.category_clusters:
        category_ids = st.session_state.category_clusters['element_ids']
        category_cluster_labels = st.session_state.category_clusters['cluster_labels']
        
        # Find if this element is in the category clusters
        if element_id in category_ids:
            idx = category_ids.index(element_id)
            cluster_label = category_cluster_labels[idx]
            
            # Find all elements in this cluster
            cluster_elements = []
            for i, label in enumerate(category_cluster_labels):
                if label == cluster_label and category_ids[i] != element_id:
                    cat_id = category_ids[i]
                    cat = next((c for c in st.session_state.categories if c['id'] == cat_id), None)
                    if cat:
                        cluster_elements.append({
                            'id': cat_id,
                            'name': cat['name'],
                            'type': 'category'
                        })
            
            # If we found cluster elements, update the result
            if cluster_elements:
                cluster_info = {
                    "in_cluster": True,
                    "cluster_label": cluster_label,
                    "cluster_elements": cluster_elements
                }
    
    # Check if this is a characteristic
    if not cluster_info["in_cluster"] and 'characteristic_clusters' in st.session_state and st.session_state.characteristic_clusters:
        char_ids = st.session_state.characteristic_clusters['element_ids']
        char_cluster_labels = st.session_state.characteristic_clusters['cluster_labels']
        
        # Find if this element is in the characteristic clusters
        if element_id in char_ids:
            idx = char_ids.index(element_id)
            cluster_label = char_cluster_labels[idx]
            
            # Find all elements in this cluster
            cluster_elements = []
            for i, label in enumerate(char_cluster_labels):
                if label == cluster_label and char_ids[i] != element_id:
                    char_id = char_ids[i]
                    char = next((c for c in st.session_state.characteristics if c['id'] == char_id), None)
                    if char:
                        cluster_elements.append({
                            'id': char_id,
                            'name': char['name'],
                            'type': 'characteristic'
                        })
            
            # If we found cluster elements, update the result
            if cluster_elements:
                cluster_info = {
                    "in_cluster": True,
                    "cluster_label": cluster_label,
                    "cluster_elements": cluster_elements
                }
    
    return cluster_info


def generate_ai_suggestion(element_id):
    """Generate AI suggestion with direct document references and BabelNet insights"""
    # Find the element
    element = next((e for e in st.session_state.unassigned_elements if e["id"] == element_id), None)
    if not element:
        return None, [], None  # Added None for BabelNet data
    
    # Get potential parent nodes (excluding root for more specific suggestions)
    potential_parents = [n for n in st.session_state.taxonomy_builder.nodes if n["type"] in ["category", "root"]]
    
    # Only suggest if we have potential parents
    if not potential_parents:
        return None, [], None
    
    # Direct search for element mentions in documents
    element_name = element["name"]
    
    direct_mentions = []
    for source_id, source in st.session_state.rag_manager.literature_sources.items():
        source_name = source["name"]
        chunks = source.get("chunks", [])
        
        # Search for direct mentions
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            element_name_lower = element_name.lower()
            
            if element_name_lower in chunk_lower:
                # Extract sentences containing the element name
                sentences = chunk.split('.')
                relevant_sentences = []
                highlighted_sentences = []
                
                for sentence in sentences:
                    if element_name_lower in sentence.lower():
                        relevant_sentences.append(sentence)
                        
                        # Create highlighted version of the sentence
                        # (We'll use {{HIGHLIGHT}} as a marker to be replaced with HTML later)
                        sentence_lower = sentence.lower()
                        start_idx = sentence_lower.find(element_name_lower)
                        if start_idx != -1:
                            highlighted = (
                                sentence[:start_idx] + 
                                "{{HIGHLIGHT}}" + sentence[start_idx:start_idx+len(element_name)] + "{{/HIGHLIGHT}}" + 
                                sentence[start_idx+len(element_name):]
                            )
                            highlighted_sentences.append(highlighted)
                        else:
                            highlighted_sentences.append(sentence)
                
                if relevant_sentences:
                    direct_mentions.append({
                        "source": source_name,
                        "text": '. '.join(relevant_sentences) + '.',
                        "highlighted_text": '. '.join(highlighted_sentences) + '.',
                        "chunk_idx": i
                    })
    
    # Get BabelNet insights for the element
    babelnet_insights = None
    
    # Get BabelNet API key from the temporary storage
    babelnet_key = st.session_state.get('temp_babelnet_key', '')
    
    if babelnet_key:
        # Use the key directly without saving it permanently
        babelnet_data = st.session_state.rag_manager.query_babelnet(element_name, api_key=babelnet_key)
        
        # Process BabelNet data to extract useful insights
        if babelnet_data:
            # Structure to store processed insights
            babelnet_insights = {
                "synsets": [],
                "pos": set(),
                "source": "BabelNet",
                "related_terms": []
            }
            
            # Extract basic information
            for item in babelnet_data:
                if "id" in item:
                    synset_id = item.get("id")
                    pos = item.get("pos", "")
                    source = item.get("source", "")
                    
                    babelnet_insights["synsets"].append({
                        "id": synset_id,
                        "pos": pos
                    })
                    
                    if pos:
                        babelnet_insights["pos"].add(pos)
    
    # Gather context
    context = ""
    if st.session_state.user_context:
        context += f"User provided context:\n{st.session_state.user_context}\n\n"
    
    # Add direct mentions to context
    if direct_mentions:
        context += "Relevant mentions from documents:\n"
        for mention in direct_mentions:
            context += f"From {mention['source']}:\n{mention['text']}\n\n"
    else:
        # If no direct mentions, fall back to semantic search
        relevant_info = st.session_state.rag_manager.retrieve_relevant_context(element_name)
        if relevant_info:
            context += "Relevant information from sources:\n"
            for info in relevant_info:
                context += f"From {info['source_name']}:\n{info['text']}\n\n"
    
    # Add BabelNet insights to context if available
    if babelnet_insights and len(babelnet_insights["synsets"]) > 0:
        context += "BabelNet semantic insights:\n"
        if babelnet_insights["pos"]:
            context += f"Part of speech: {', '.join(babelnet_insights['pos'])}\n"
        context += f"Synset IDs: {', '.join([s['id'] for s in babelnet_insights['synsets']])}\n\n"
    
    # Generate suggestion using LLM
    suggestion = st.session_state.llm_manager.generate_taxonomy_suggestion(
        element=element,
        potential_parents=potential_parents,
        context=context
    )
    
    return suggestion, direct_mentions, babelnet_insights

def setup_llm():
    """Setup the LLM for taxonomy construction"""
    st.subheader("LLM Model Selection")
    
    # Connect to LM Studio server
    if st.button("Connect to LM Studio"):
        with st.spinner("Connecting to LM Studio..."):
            if st.session_state.llm_manager.connect():
                st.success("Connected to LM Studio server")
                
                # Get available models
                models = st.session_state.llm_manager.get_available_models()
                if models:
                    st.session_state.available_models = models
                    st.success(f"Found {len(models)} models")
                else:
                    st.warning("No models found. Please ensure LM Studio is running with models loaded.")
            else:
                st.error("Failed to connect to LM Studio. Please ensure the server is running at http://127.0.0.1:1234")
    
    # Show available models if any
    if 'available_models' in st.session_state and st.session_state.available_models:
        selected_model = st.selectbox(
            "Select Model",
            options=st.session_state.available_models,
            index=0 if st.session_state.llm_manager.selected_model is None else 
                  st.session_state.available_models.index(st.session_state.llm_manager.selected_model)
        )
        
        if st.button("Set Model"):
            if st.session_state.llm_manager.set_model(selected_model):
                st.success(f"Model set to {selected_model}")
            else:
                st.error("Failed to set model")
    
    # Test LLM connection
    if st.session_state.llm_manager.selected_model:
        if st.button("Test LLM"):
            with st.spinner("Testing LLM..."):
                response = st.session_state.llm_manager.generate_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Say hello and confirm you're working properly."}
                    ],
                    max_tokens=100
                )
                st.success("LLM test successful!")
                st.info(f"LLM Response: {response}")


def setup_rag():
    """Setup RAG (Retrieval-Augmented Generation) for taxonomy construction"""
    st.subheader("RAG Configuration")
    st.write("Configure the knowledge sources that will help the LLM make better taxonomy decisions.")
    
    rag_tabs = st.tabs(["Taxonomies", "Literature", "BabelNet", "Custom Input", "RAG Details"])
    
    with rag_tabs[0]:
        st.write("### Original Taxonomies")
        st.write("Upload the original taxonomies to provide structural context.")
        
        # Upload taxonomy files
        uploaded_tax = st.file_uploader("Upload taxonomy JSON", type="json", key="taxonomy_uploader")
        
        if uploaded_tax:
            tax_name = st.text_input("Taxonomy name", value=uploaded_tax.name)
            
            if st.button("Add Taxonomy"):
                with st.spinner("Processing taxonomy..."):
                    source_id = st.session_state.rag_manager.add_taxonomy_source(tax_name, uploaded_tax)
                    if source_id:
                        st.success(f"Added taxonomy source: {tax_name}")
                    else:
                        st.error("Failed to add taxonomy source")
        
        # Show current taxonomy sources
        if st.session_state.rag_manager.taxonomy_sources:
            st.write("Current taxonomy sources:")
            source_df = pd.DataFrame([
                {"ID": id, "Name": source["name"], "Added": source["date_added"]}
                for id, source in st.session_state.rag_manager.taxonomy_sources.items()
            ])
            st.dataframe(source_df)
    
    with rag_tabs[1]:
        st.write("### Literature")
        st.write("Upload research papers to provide domain knowledge.")
        
        # Upload literature files
        uploaded_lit = st.file_uploader("Upload literature PDF", type="pdf", key="literature_uploader")
        
        if uploaded_lit:
            lit_name = st.text_input("Literature name", value=uploaded_lit.name)
            
            if st.button("Add Literature"):
                with st.spinner("Processing literature..."):
                    source_id = st.session_state.rag_manager.add_literature_source(lit_name, uploaded_lit.read())
                    if source_id:
                        st.success(f"Added literature source: {lit_name}")
                        
                        # Display chunking statistics
                        if source_id in st.session_state.rag_manager.chunk_stats:
                            stats = st.session_state.rag_manager.chunk_stats[source_id]
                            st.write("### Document Processing Stats")
                            st.info(f"""
                            - Total chunks: {stats['total_chunks']}
                            - Average chunk size: {stats['avg_chunk_size']:.0f} characters
                            - Sections identified: {stats['section_coverage']}
                            - Semantic boundaries preserved: {stats['semantic_boundaries']}
                            """)
                            
                            # Display document map
                            doc_map = st.session_state.rag_manager.get_document_map(source_id)
                            if doc_map:
                                with st.expander("Document Map"):
                                    st.write(f"**Title:** {doc_map.get('title', 'Unknown')}")
                                    st.write("**Structure:**")
                                    for item in doc_map.get('structure', []):
                                        indent = "  " * (item.get('level', 1) - 1)
                                        st.write(f"{indent}• {item.get('title', '')}")
                    else:
                        st.error("Failed to add literature source")
        
        # Show current literature sources
        if st.session_state.rag_manager.literature_sources:
            st.write("Current literature sources:")
            lit_df = pd.DataFrame([
                {"ID": id, "Name": source["name"], "Added": source["date_added"], 
                 "Chunks": len(source.get("chunks", [])), 
                 "Embedded": "Yes" if source.get("embeddings") is not None else "No"}
                for id, source in st.session_state.rag_manager.literature_sources.items()
            ])
            st.dataframe(lit_df)
            
            # Add a button to view document details
            selected_doc = st.selectbox("Select document to view details:", 
                                      options=[source["name"] for source_id, source in st.session_state.rag_manager.literature_sources.items()],
                                      key="doc_details_selector")
            
            if selected_doc and st.button("View Document Details"):
                # Find the source ID based on name
                source_id = next((sid for sid, source in st.session_state.rag_manager.literature_sources.items() 
                                 if source["name"] == selected_doc), None)
                
                if source_id:
                    source = st.session_state.rag_manager.literature_sources[source_id]
                    
                    with st.expander("Document Structure", expanded=True):
                        # Show document map if available
                        doc_map = st.session_state.rag_manager.get_document_map(source_id)
                        if doc_map:
                            st.write(f"**Title:** {doc_map.get('title', 'Unknown')}")
                            st.write("**Structure:**")
                            for item in doc_map.get('structure', []):
                                indent = "  " * (item.get('level', 1) - 1)
                                st.write(f"{indent}• {item.get('title', '')}")
                        
                        # Show some statistics
                        st.write("### Document Statistics")
                        st.info(f"""
                        - Total chunks: {len(source.get('chunks', []))}
                        - Document length: {len(source.get('content', ''))} characters
                        - Sections identified: {len(set(m.get('section', '') for m in source.get('chunk_metadata', [])))}
                        """)
                    
                    with st.expander("Sample Chunks"):
                        # Show a few sample chunks
                        chunks = source.get("chunks", [])
                        if chunks:
                            # Take a few samples from different parts of the document
                            indices = [0]  # First chunk
                            if len(chunks) > 1:
                                indices.append(len(chunks) // 2)  # Middle chunk
                            if len(chunks) > 2:
                                indices.append(len(chunks) - 1)  # Last chunk
                            
                            for idx in indices:
                                st.write(f"**Chunk {idx+1}:**")
                                st.text_area(f"Content (Chunk {idx+1})", value=chunks[idx][:500] + "...", height=150)
    
    with rag_tabs[2]:
        st.write("### BabelNet Integration")
        st.write("Connect to BabelNet API to enrich semantic relationships.")
        
        # BabelNet API key input
        babelnet_key = st.text_input(
            "BabelNet API Key", 
            type="password",
            key="babelnet_key_input"
        )
        
        # Store temporarily
        if babelnet_key:
            st.session_state.temp_babelnet_key = babelnet_key
        
        # Test term input
        test_term = st.text_input("Test term", value="effort estimation", key="test_term_input")
        st.session_state.test_term = test_term
        
        # Test button
        if st.button("Test BabelNet API"):
            test_babelnet_api()
    
    with rag_tabs[3]:
        st.write("### Custom Knowledge")
        st.write("Add your own knowledge or context for the LLM to consider.")
        
        # Custom context input
        custom_name = st.text_input("Context name", value="User knowledge")
        custom_text = st.text_area(
            "Custom knowledge", 
            value=st.session_state.get('user_context', ''),
            height=200,
            placeholder="Add any domain-specific knowledge, taxonomy design principles, or context that will help the LLM make better decisions..."
        )
        
        if st.button("Save Custom Knowledge"):
            st.session_state.user_context = custom_text
            source_id = st.session_state.rag_manager.add_custom_source(custom_name, custom_text)
            if source_id:
                st.success("Custom knowledge saved")
            else:
                st.error("Failed to save custom knowledge")
        
        # Show current custom sources
        if st.session_state.rag_manager.custom_sources:
            st.write("Current custom knowledge sources:")
            custom_df = pd.DataFrame([
                {"ID": id, "Name": source["name"], "Added": source["date_added"]}
                for id, source in st.session_state.rag_manager.custom_sources.items()
            ])
            st.dataframe(custom_df)
    
    with rag_tabs[4]:
        st.write("### RAG Implementation Details")
        
        st.write("""
        ## Advanced PDF Processing & Intelligent Retrieval
        
        This system uses a sophisticated approach to extract information from documents and retrieve relevant context:
        """)
        
        # Explain the PDF processing
        with st.expander("1. Enhanced PDF Processing", expanded=True):
            st.write("""
            ### Sophisticated PDF Text Extraction
            
            Our system uses multiple approaches to extract high-quality content from PDFs:
            
            1. **Dual-Engine Extraction**: Combines PyMuPDF and pdfplumber for robust extraction
            2. **Structure Preservation**: Identifies headings, sections, and special elements
            3. **Layout Analysis**: Preserves document organization when possible
            4. **Metadata Extraction**: Captures titles, authors, and document properties
            5. **Table of Contents**: Extracts TOC to understand document structure
            
            ### Document Map Generation
            
            A document map is created to understand the document's organization:
            
            1. **Hierarchical Outline**: Shows the document's structure
            2. **Section Identification**: Tracks different sections within the document
            3. **Heading Detection**: Identifies main and sub-headings
            
            This allows the LLM to "understand" the document organization rather than seeing it as a flat text.
            """)
        
        # Explain the semantic chunking
        with st.expander("2. Semantic Chunking", expanded=True):
            st.write("""
            ### Intelligent Semantic Chunking
            
            Traditional RAG often breaks documents into fixed-size chunks, losing context and logic. Our approach:
            
            1. **Semantic Boundary Preservation**: Respects logical divisions like sections and paragraphs
            2. **Hierarchy-Aware**: Maintains knowledge of where chunks belong in the document
            3. **Overlapping Content**: Creates contextual overlap between adjacent chunks
            4. **Metadata Enrichment**: Each chunk knows its section, page numbers, and significance
            5. **Size Optimization**: Balances chunk size for optimal LLM context windows
            
            ### Multi-Level Summarization
            
            Automatically generates summaries at three levels:
            
            1. **Document-Level**: Overall document summary
            2. **Section-Level**: Summary of each major section
            3. **Chunk-Level**: Concise summary of each chunk
            
            This creates a hierarchical understanding of the document content.
            """)
        
        # Explain the hierarchical indexing
        with st.expander("3. Hierarchical Indexing", expanded=True):
            st.write("""
            ### Multi-Level Vector Embeddings
            
            Our system creates three levels of embeddings:
            
            1. **Document Embeddings**: Capture the overall topic and content
            2. **Section Embeddings**: Represent major sections and their themes
            3. **Chunk Embeddings**: Detailed vectors for specific content pieces
            
            ### Indexing Strategy
            
            The hierarchical structure allows for more efficient information retrieval:
            
            1. **Top-Down Search**: First identify relevant documents and sections
            2. **Bottom-Up Verification**: Confirm relevance with specific chunk content
            3. **Cross-Reference Support**: Connect related information across the document
            """)
        
        # Explain the advanced retrieval
        with st.expander("4. Hybrid Retrieval Strategy", expanded=True):
            st.write("""
            ### Sophisticated Query Processing
            
            When processing a query, our system:
            
            1. **Query Decomposition**: Breaks complex queries into sub-components
            2. **Intent Detection**: Identifies the core information need
            3. **Linguistic Analysis**: Extracts key terms and relationships
            
            ### Multi-Strategy Search
            
            Combines multiple retrieval approaches:
            
            1. **Semantic Search**: Uses embedding similarity to find conceptually related content
            2. **Keyword Matching**: Ensures specific terms are present in results
            3. **Hybrid Scoring**: Balances semantic and keyword relevance
            
            ### Re-Ranking for Quality
            
            Applies sophisticated re-ranking to final results:
            
            1. **Diversity Promotion**: Ensures varied results from different sources
            2. **Section Coherence**: Prefers content from logically related sections
            3. **Quality Factors**: Considers factors like content length and specificity
            
            This multi-faceted approach ensures the LLM receives the most relevant context for your questions.
            """)
        
        # Explain the reliability and limitations
        with st.expander("5. Reliability and Limitations", expanded=True):
            st.write("""
            ### Reliability Assessment
            
            When considering the reliability of retrieved information:
            
            1. **High Reliability**: Content with both high semantic and keyword relevance, especially from multiple sources
            2. **Medium Reliability**: Content with good scores from a single approach or source
            3. **Lower Reliability**: Content with borderline relevance scores or minimal matching signals
            
            ### Known Limitations
            
            While advanced, our approach has some limitations:
            
            1. **Complex Layouts**: Very complex PDF layouts may lose some structural information
            2. **Language Limitations**: Works best with English content
            3. **Technical Content**: May struggle with highly specialized notation or symbols
            4. **Image-Based Information**: Cannot extract information from images, charts, or diagrams
            5. **Very Large Documents**: Extremely large documents may be processed with less detail
            
            ### Improving Results
            
            For best results:
            
            1. Ask clear, specific questions
            2. Reference document sections when known
            3. Break complex questions into simpler ones
            4. Review multiple retrieved passages when available
            """)
        
        # Add a test area
        st.write("### Test Retrieval")
        test_query = st.text_input("Try a test query:", placeholder="Enter a query to test retrieval...")
        test_mode = st.radio("Retrieval Mode:", ["hybrid", "semantic", "keyword"], horizontal=True)
        
        if test_query and st.button("Test Retrieval"):
            with st.spinner("Retrieving information..."):
                results = st.session_state.rag_manager.retrieve_relevant_context(
                    test_query, retrieval_mode=test_mode, max_results=3
                )
                
                # Show explanation
                explanation = st.session_state.rag_manager.explain_retrieval(test_query, results, detailed=True)
                
                st.write("### Retrieval Results")
                st.info(f"""
                Query: "{test_query}"
                Results found: {explanation['results_count']}
                Strategy: {explanation['retrieval_strategy']}
                Sources used: {', '.join(explanation['sources_used'])}
                Score range: {explanation['score_range']['min']:.2f} - {explanation['score_range']['max']:.2f}
                """)
                
                # Display retrieved passages
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1}: {result['source_name']} (Score: {result['score']:.2f})", expanded=i==0):
                        st.write(f"**Source**: {result['source_name']}")
                        if "section" in result and result["section"]:
                            st.write(f"**Section**: {result['section']}")
                        
                        score_details = f"Semantic: {result.get('semantic_score', 0):.2f}, Keyword: {result.get('keyword_score', 0):.2f}"
                        st.write(f"**Relevance**: {result['score']:.2f} ({score_details})")
                        
                        # Show metadata
                        if "metadata" in result:
                            meta = result["metadata"]
                            if "start_page" in meta and "end_page" in meta:
                                st.write(f"**Pages**: {meta['start_page']+1}-{meta['end_page']+1}")
                        
                        # Display the content
                        st.markdown("---")
                        st.text_area("Content", value=result["text"], height=200)
                        st.markdown("---")

def recalculate_node_embedding(element_id, new_name):
    """
    Recalculate embedding for a single element after name change
    
    Args:
        element_id: The ID of the element to update
        new_name: The new name for the element
        
    Returns:
        bool: Whether the recalculation was successful
    """
    try:
        # Get embedding calculator from session state
        embedding_calculator = st.session_state.embedding_calculator
        
        # Make sure model and tokenizer are loaded
        if not embedding_calculator.model or not embedding_calculator.tokenizer:
            if not embedding_calculator.load_model():
                return False
        
        # Update the name in the element map if it exists
        if element_id in embedding_calculator.element_map:
            embedding_calculator.element_map[element_id]['name'] = new_name
        else:
            # If element isn't in element map, add it
            embedding_calculator.element_map[element_id] = {'id': element_id, 'name': new_name}
        
        # Tokenize the new name
        inputs = embedding_calculator.tokenizer([new_name], padding=True, truncation=True, return_tensors="pt")
        
        # Compute new embedding
        with torch.no_grad():
            model_output = embedding_calculator.model(**inputs)
            embedding = embedding_calculator.mean_pooling(model_output, inputs['attention_mask'])
        
        # Update the embedding in the embeddings dictionary
        embedding_calculator.embeddings[element_id] = embedding[0].numpy()
        
        # Recalculate similarity matrix if it exists
        if embedding_calculator.similarity_matrix is not None:
            embedding_calculator.compute_similarity_matrix()
        
        return True
    except Exception as e:
        st.error(f"Error recalculating embedding: {str(e)}")
        return False

def display_unassigned_elements():
    """Display unassigned elements with truly non-collapsible cards"""
    st.markdown("<h3 style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>UNASSIGNED ELEMENTS</h3>", unsafe_allow_html=True)
    # Add this CSS along with your other style definitions
    st.markdown("""
    <style>
    /* This targets all Streamlit buttons */
    .stButton > button {
        font-size: 10px !important;  /* Change this value to your desired font size */
        padding: 1px 7px !important; /* Adjust padding to match the new font size */
    }
    </style>
    """, unsafe_allow_html=True)
    # Improved filtering and search options
    filter_col1, filter_col2, filter_col3 = st.columns([3, 1, 1])
    
    with filter_col1:
        # Enhanced search box
        search_term = st.text_input("🔍 Search elements...", key="search_elements", 
                                    placeholder="Type to search by name or description")
    
    with filter_col2:
        # Alphabetical sorting toggle
        sort_order = st.selectbox(
            "Sort by:",
            ["Name (A-Z)", "Name (Z-A)", "Type", "Source"],
            index=0
        )
    
    with filter_col3:
        # Filter by letter
        letter_filter = st.selectbox("Filter by:", ['All', 'A-E', 'F-J', 'K-O', 'P-T', 'U-Z', '#'], index=0)
    
    # Get and filter elements based on construction mode
    construction_mode = st.session_state.get("construction_mode", "Categories First")
    filtered_elements = st.session_state.unassigned_elements
    
    # Filter by type based on construction mode
    if construction_mode == "Categories First":
        filtered_elements = [e for e in filtered_elements if e["type"] == "category"]
    elif construction_mode == "Characteristics Second":
        filtered_elements = [e for e in filtered_elements if e["type"] == "characteristic"]
    
    # Apply search filter
    if search_term:
        filtered_elements = [e for e in filtered_elements if search_term.lower() in e["name"].lower() or 
                            (e.get("description") and search_term.lower() in e["description"].lower())]
    
    # Apply letter filter
    if letter_filter != 'All':
        if letter_filter == 'A-E':
            filtered_elements = [e for e in filtered_elements if e["name"][0].upper() in 'ABCDE']
        elif letter_filter == 'F-J':
            filtered_elements = [e for e in filtered_elements if e["name"][0].upper() in 'FGHIJ']
        elif letter_filter == 'K-O':
            filtered_elements = [e for e in filtered_elements if e["name"][0].upper() in 'KLMNO']
        elif letter_filter == 'P-T':
            filtered_elements = [e for e in filtered_elements if e["name"][0].upper() in 'PQRST']
        elif letter_filter == 'U-Z':
            filtered_elements = [e for e in filtered_elements if e["name"][0].upper() in 'UVWXYZ']
        elif letter_filter == '#':
            filtered_elements = [e for e in filtered_elements if not e["name"][0].upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    
    # Sort elements
    if sort_order == "Name (A-Z)":
        filtered_elements = sorted(filtered_elements, key=lambda e: e["name"])
    elif sort_order == "Name (Z-A)":
        filtered_elements = sorted(filtered_elements, key=lambda e: e["name"], reverse=True)
    elif sort_order == "Type":
        filtered_elements = sorted(filtered_elements, key=lambda e: (e["type"], e["name"]))
    elif sort_order == "Source":
        filtered_elements = sorted(filtered_elements, key=lambda e: (e.get("source", "Unknown"), e["name"]))
    
    # Pagination
    items_per_page = 18  # Six columns x 3 rows
    total_pages = (len(filtered_elements) + items_per_page - 1) // items_per_page if filtered_elements else 1
    
    # If we don't have a current page in session state, initialize it
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Ensure current page is valid
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = 1
    
    # Get the current page elements
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_elements))
    current_elements = filtered_elements[start_idx:end_idx] if filtered_elements else []
    
    # Display pagination info
    if filtered_elements:
        st.markdown(f"Showing {start_idx+1}-{end_idx} of {len(filtered_elements)} elements")
    
    # Create custom cards for elements
    if current_elements:
        # Calculate number of columns (use 6 columns)
        num_columns = 6
        
        # Create rows for elements
        for i in range(0, len(current_elements), num_columns):
            # Create columns for this row
            cols = st.columns(num_columns)
            
            # Add elements to this row
            for j in range(num_columns):
                idx = i + j
                if idx < len(current_elements):
                    element = current_elements[idx]
                    element_id = element["id"]
                    element_name = element["name"]
                    element_type = element["type"].capitalize()
                    element_source = element.get("source", "Unknown")
                    
                    with cols[j]:
                        # Each card is a separate container with border
                        with st.container(border=True):
                            # Type and source
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                if element_type == "Category":
                                    st.markdown(f"<span style='background-color: #2ecc71; color: white; padding: 2px 6px; border-radius: 12px; font-size: 12px;'>{element_type}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='background-color: #e74c3c; color: white; padding: 2px 6px; border-radius: 12px; font-size: 12px;'>{element_type}</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"<span style='background-color: #f5f5f5; color: #666; padding: 2px 6px; border-radius: 12px; font-size: 12px; float: right;'>{element_source}</span>", unsafe_allow_html=True)
                            
                            # Element name
                            st.markdown(f"<div style='font-weight: bold; font-size: 16px; margin: 5px 0; overflow-wrap: break-word; word-wrap: break-word; hyphens: auto;'>{element_name}</div>", unsafe_allow_html=True)
                            
                            # Buttons
                            button_cols = st.columns(2)
                            with button_cols[0]:
                                if st.button("Select", key=f"select_{element_id}", use_container_width=True):
                                    select_element(element_id)
                            with button_cols[1]:
                                if st.button("Add to Root", key=f"root_{element_id}", use_container_width=True):
                                    add_element_to_taxonomy(element_id, "root")
                                    st.rerun()
        
        # Pagination controls
        st.markdown("---")
        pagination_cols = st.columns([1, 3, 1])
        
        with pagination_cols[0]:
            if st.session_state.current_page > 1:
                if st.button("← Prev"):
                    st.session_state.current_page -= 1
                    st.rerun()
        
        with pagination_cols[1]:
            # Page numbers
            page_numbers = []
            if total_pages <= 7:
                page_numbers = list(range(1, total_pages + 1))
            else:
                if st.session_state.current_page <= 3:
                    page_numbers = list(range(1, 6)) + ["...", total_pages]
                elif st.session_state.current_page >= total_pages - 2:
                    page_numbers = [1, "..."] + list(range(total_pages - 4, total_pages + 1))
                else:
                    page_numbers = [1, "..."] + list(range(st.session_state.current_page - 1, 
                                                        st.session_state.current_page + 2)) + ["...", total_pages]
            
            # Create a compact page selector
            page_cols = st.columns(len(page_numbers))
            for i, page in enumerate(page_numbers):
                with page_cols[i]:
                    if page == "...":
                        st.markdown("...")
                    elif page == st.session_state.current_page:
                        st.markdown(f"**{page}**")
                    else:
                        if st.button(f"{page}", key=f"page_{page}"):
                            st.session_state.current_page = page
                            st.rerun()
        
        with pagination_cols[2]:
            if st.session_state.current_page < total_pages:
                if st.button("Next →"):
                    st.session_state.current_page += 1
                    st.rerun()
    else:
        if construction_mode == "Categories First":
            st.info("No unassigned categories found. You can now switch to 'Characteristics Second' mode.")
        else:
            st.info("No unassigned elements found. All elements have been placed in the taxonomy.")
def taxonomy_builder_interface():
    """Interactive taxonomy builder interface with improved layout based on user diagram"""
    st.title("Interactive Taxonomy Builder")
    
    # Top Row: Mode selection and header
    mode_col1, mode_col2 = st.columns([1, 3])
    
    with mode_col1:
        # Construction mode toggle for two-phase construction
        construction_mode = st.radio(
            "Construction Mode:",
            ["Categories First", "Characteristics Second"],
            horizontal=True,
            help="First build the category structure, then add characteristics"
        )
        st.session_state.construction_mode = construction_mode
    
    with mode_col2:
        st.write("""
        This interface follows a two-phase approach:
        1. **Categories First**: Build the structural hierarchy with categories
        2. **Characteristics Second**: Add detailed characteristics to the appropriate categories
        """)
    
    # Display unassigned elements with buttons inside cards
    display_unassigned_elements()
    
    # Middle Section: Taxonomy Visualization
    st.markdown("---")
    
    # Add a visually distinctive header for the visualization section
    st.markdown("<h3 style='text-align: center; background-color: #ffcccc; padding: 10px; border-radius: 5px;'>TAXONOMY VISUALIZATION</h3>", unsafe_allow_html=True)
    
    with st.container(border=True):
        # Generate taxonomy visualization with interactivity
        fig = st.session_state.taxonomy_builder.visualize_taxonomy(
            selected_node=st.session_state.selected_node,
            hover_node=st.session_state.hover_node
        )
        
        # Display the visualization
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom Section: Controls split into two columns
    st.markdown("---")
    
    # Left column: Manual Placement & Node Editing
    bottom_col1, bottom_col2 = st.columns(2)
    
    # Bottom Left: Manual Placement & Node Editing
    with bottom_col1:
        # Manual Placement section
        st.markdown("<h3 style='background-color: #ccccff; padding: 10px; border-radius: 5px;'>MANUAL PLACEMENT</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            if st.session_state.selected_element:
                selected = next((e for e in st.session_state.unassigned_elements if e["id"] == st.session_state.selected_element), None)
                if selected:
                    st.markdown(f"Place **{selected['name']}** under:")
                    
                    # Show all possible parent nodes
                    parent_options = [(node["id"], node["name"]) for node in st.session_state.taxonomy_builder.nodes]
                    
                    selected_parent = st.selectbox(
                        "Select parent", 
                        options=parent_options,
                        format_func=lambda x: x[1]
                    )
                    
                    if st.button("Place Element", key="place_element"):
                        add_element_to_taxonomy(selected["id"], selected_parent[0])
                        st.rerun()
            else:
                st.markdown("*Select an element from above to place it manually*")
        
        # Node Selection & Editing
        st.markdown("<h3 style='background-color: #ccffcc; padding: 10px; border-radius: 5px;'>SELECT AND EDIT NODE</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            # Add selection for nodes in the taxonomy
            node_options = [(node["id"], node["name"]) for node in st.session_state.taxonomy_builder.nodes]
            
            selected_node_option = st.selectbox(
                "Select a node from the taxonomy",
                options=node_options,
                format_func=lambda x: x[1],
                key="node_selector"
            )
            
            # Show node editing options when a node is selected
            if selected_node_option:
                node_id = selected_node_option[0]
                node = st.session_state.taxonomy_builder.get_node(node_id)
                
                if node:
                    st.session_state.selected_node = node_id
                    
                    # Edit name with warning
                    new_name = st.text_input("Name", value=node["name"], key=f"edit_name_{node_id}")
                    if new_name != node["name"]:
                        st.warning("⚠️ Changing the name will require recalculating embeddings for this element.")
                    
                    # Edit description
                    new_description = st.text_input("Description", value=node.get("description", ""), key=f"edit_desc_{node_id}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Save changes button
                        if st.button("Save Changes", key=f"save_{node_id}"):
                            updates = {
                                "name": new_name,
                                "description": new_description
                            }
                            st.session_state.taxonomy_builder.update_node(node_id, updates)
                            st.success("Node updated!")
                    
                    with col2:
                        # Delete node option (with confirmation)
                        if node_id != "root" and st.button("Delete Node", key=f"delete_{node_id}"):
                            if st.session_state.taxonomy_builder.delete_node(node_id):
                                st.success("Node deleted!")
                                st.session_state.selected_node = None
                                st.rerun()
                    
                    # Move node option
                    if node_id != "root":
                        # Get all potential new parents
                        potential_parents = [(n["id"], n["name"]) for n in st.session_state.taxonomy_builder.nodes 
                                            if n["id"] != node_id and not st.session_state.taxonomy_builder.is_descendant(node_id, n["id"])]
                        
                        new_parent = st.selectbox(
                            "Move to New Parent",
                            options=potential_parents,
                            format_func=lambda x: x[1],
                            key=f"new_parent_{node_id}"
                        )
                        
                        if st.button("Move Node", key=f"move_{node_id}"):
                            if st.session_state.taxonomy_builder.move_node(node_id, new_parent[0]):
                                st.success(f"Moved node to {new_parent[1]}")
                                st.rerun()
        
        # Add new category section
        st.markdown("<h3 style='background-color: #ccffcc; padding: 10px; border-radius: 5px;'>ADD NEW CATEGORY</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            new_cat_name = st.text_input("Category name", key="new_category_name")
            new_cat_desc = st.text_input("Description (optional)", key="new_category_desc")
            
            parent_options = [(node["id"], node["name"]) for node in st.session_state.taxonomy_builder.nodes]
            selected_parent = st.selectbox(
                "Parent node", 
                options=parent_options,
                format_func=lambda x: x[1],
                key="new_category_parent"
            )
            
            if st.button("Add Category", key="add_category") and new_cat_name:
                # Add to taxonomy
                st.session_state.taxonomy_builder.add_node(
                    name=new_cat_name,
                    node_type="category",
                    parent_id=selected_parent[0],
                    description=new_cat_desc,
                    source="user-created"
                )
                st.success(f"Added new category: {new_cat_name}")
                st.rerun()
    
    # Bottom Right: Similarity Analysis & AI Assistant
    with bottom_col2:
        # Similarity Analysis
        st.markdown("<h3 style='background-color: #ccccff; padding: 10px; border-radius: 5px;'>SIMILARITY ANALYSIS</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            if st.session_state.selected_element and hasattr(st.session_state, 'element_similar_elements'):
                selected = next((e for e in st.session_state.unassigned_elements if e["id"] == st.session_state.selected_element), None)
                similar_elements = getattr(st.session_state, 'element_similar_elements', [])
                
                if selected and similar_elements:
                    st.markdown(f"**{selected['name']}** is similar to:")
                    
                    for sim in similar_elements:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            similarity_percentage = int(sim["similarity"] * 100)
                            st.progress(sim["similarity"], text=f"{sim['name']}")
                        with col2:
                            st.markdown(f"{similarity_percentage}%")
                            
                            # Add quick-place button for similar elements
                            if st.button("Place", key=f"place_similar_{sim['id']}"):
                                # Get the parent of the selected node
                                if st.session_state.selected_node:
                                    parent_id = st.session_state.selected_node
                                    add_element_to_taxonomy(sim['id'], parent_id)
                                    st.rerun()
                else:
                    st.markdown("*No similar elements found*")
            else:
                st.markdown("*Select an element to see similarity analysis*")
        
        # AI Assistant
        st.markdown("<h3 style='background-color: #ccccff; padding: 10px; border-radius: 5px;'>AI ASSISTANT</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            # Get API key for BabelNet without saving it permanently
            babelnet_key_input = st.text_input(
                "BabelNet API Key (optional)", 
                type="password",
                key="babelnet_key_for_suggestions",
                help="Enter your BabelNet API key to include semantic information in suggestions"
            )
            
            # Temporarily store the key just for this operation
            if babelnet_key_input:
                st.session_state.temp_babelnet_key = babelnet_key_input
            
            # Get AI suggestions button
            if st.session_state.selected_element:
                if st.button("Get AI Suggestions", key="get_ai_suggestions"):
                    with st.spinner("Generating suggestions..."):
                        suggestion, document_fragments, babelnet_insights = generate_ai_suggestion(st.session_state.selected_element)
                        if suggestion and 'parent' in suggestion:
                            st.session_state.ai_suggestion = suggestion
                            st.session_state.document_fragments = document_fragments
                            st.session_state.babelnet_insights = babelnet_insights  # Store BabelNet insights
                        else:
                            st.error("Failed to generate AI suggestion")
            
            # AI suggestion
            if st.session_state.selected_element and hasattr(st.session_state, 'ai_suggestion') and st.session_state.ai_suggestion:
                selected = next((e for e in st.session_state.unassigned_elements if e["id"] == st.session_state.selected_element), None)
                suggestion = st.session_state.ai_suggestion
                
                if selected and 'parent' in suggestion and suggestion['parent']:
                    st.markdown(f"Based on analysis, **{selected['name']}** should be placed under:")
                    
                    st.info(f"**{suggestion['parent']['name']}**")
                    
                    # Show explanation
                    with st.expander("See explanation", expanded=False):
                        st.markdown(suggestion['explanation'])
                    
                    # Document evidence expander
                    if hasattr(st.session_state, 'document_fragments') and st.session_state.document_fragments:
                        fragments = st.session_state.document_fragments
                        with st.expander("Document evidence", expanded=False):
                            if fragments:
                                st.write(f"Found {len(fragments)} relevant mentions in documents:")
                                for i, fragment in enumerate(fragments):
                                    st.markdown(f"**From {fragment['source']}:**")
                                    
                                    # Replace highlight markers with HTML
                                    highlighted_html = fragment.get("highlighted_text", fragment["text"])
                                    highlighted_html = highlighted_html.replace(
                                        "{{HIGHLIGHT}}", "<span style='background-color: #FFF9C4; font-weight: bold; padding: 0 2px;'>")
                                    highlighted_html = highlighted_html.replace("{{/HIGHLIGHT}}", "</span>")
                                    
                                    # Display with highlighting
                                    st.markdown(highlighted_html, unsafe_allow_html=True)
                                    st.markdown("---")
                            else:
                                st.info(f"No direct mentions of '{selected['name']}' found in documents.")
                    
                    # BabelNet Insights expander
                    if hasattr(st.session_state, 'babelnet_insights') and st.session_state.babelnet_insights:
                        insights = st.session_state.babelnet_insights
                        with st.expander("BabelNet semantic insights", expanded=False):
                            if insights and insights.get("synsets") and len(insights["synsets"]) > 0:
                                st.markdown("### Semantic Information from BabelNet")
                                
                                # Show part of speech information
                                if insights["pos"]:
                                    st.markdown(f"**Part of Speech:** {', '.join(insights['pos'])}")
                                
                                # Show synsets
                                st.markdown("**BabelNet Synsets:**")
                                for synset in insights["synsets"]:
                                    synset_id = synset.get("id", "")
                                    pos = synset.get("pos", "")
                                    st.markdown(f"- ID: `{synset_id}` ({pos})")
                                
                                # Add link to BabelNet for more information
                                if len(insights["synsets"]) > 0:
                                    first_synset = insights["synsets"][0]["id"]
                                    st.markdown(f"""
                                    <a href="https://babelnet.org/synset?id={first_synset}" target="_blank">
                                        View in BabelNet
                                    </a>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("---")
                                st.info("These semantic insights help understand how this concept is represented in the BabelNet knowledge graph.")
                            else:
                                st.info(f"No BabelNet information found for '{selected['name']}'")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Accept", key="accept_suggestion"):
                            add_element_to_taxonomy(selected["id"], suggestion['parent']['id'])
                            # Clear the suggestion, fragments, and BabelNet insights
                            st.session_state.ai_suggestion = None
                            st.session_state.document_fragments = None
                            st.session_state.babelnet_insights = None
                            # Also clear the temporary BabelNet key
                            if 'temp_babelnet_key' in st.session_state:
                                del st.session_state.temp_babelnet_key
                            st.rerun()

                    with col2:
                        if st.button("Reject", key="reject_suggestion"):
                            # Clear the suggestion, fragments, and BabelNet insights
                            st.session_state.ai_suggestion = None
                            st.session_state.document_fragments = None
                            st.session_state.babelnet_insights = None
                            # Also clear the temporary BabelNet key
                            if 'temp_babelnet_key' in st.session_state:
                                del st.session_state.temp_babelnet_key
                            st.rerun()
    
    # Footer: Context & Advanced Options
    st.markdown("---")
    st.markdown("<h3 style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>CONTEXT FOR CURRENT TASK</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        user_context = st.text_area(
            "Add context for the AI assistant", 
            value=st.session_state.user_context,
            height=100
        )
        
        if st.button("Update Context", key="update_context"):
            st.session_state.user_context = user_context
            st.success("Context updated! AI will consider this in future suggestions.")
    
    # AI Advice & Auto-generate
    st.markdown("<h3 style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>ADVANCED AI FEATURES</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        advanced_tabs = st.tabs(["General AI Advice", "Auto-Generate Taxonomy", "Export"])
        
        with advanced_tabs[0]:
            st.markdown("""
            Get general advice about taxonomy construction rather than 
            specific element placement suggestions. This is useful when
            you're stuck or need high-level guidance.
            """)
            
            if st.button("Ask AI for General Advice", key="ask_ai"):
                if not st.session_state.user_context:
                    st.warning("Please add some context in the Context section first")
                else:
                    with st.spinner("Generating advice..."):
                        # Create a prompt for general taxonomy advice
                        prompt = f"""
                        I'm building a taxonomy for software engineering effort estimation methods.
                        
                        Context information:
                        {st.session_state.user_context}
                        
                        Please give me advice on how to structure this taxonomy in terms of:
                        1. High-level organization principles
                        2. What should be the main categories
                        3. How to handle overlapping concepts
                        """
                        
                        response = st.session_state.llm_manager.generate_chat_completion(
                            messages=[
                                {"role": "system", "content": "You are an AI assistant helping with taxonomy construction for software engineering effort estimation."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        st.info(f"AI Advice: {response}")
        
        with advanced_tabs[1]:
            st.markdown("""
            Automatically generate a taxonomy structure based on the clustering 
            results from Steps 2 and 3. This will create a hierarchical structure
            using embedding similarity.
            
            This is useful as a starting point that you can then refine manually.
            """)
            
            if st.button("Auto-Generate From Clusters", key="auto_arrange"):
                with st.spinner("Generating taxonomy from clusters..."):
                    # Check if clustering was performed
                    if not hasattr(st.session_state, 'category_clusters') or not st.session_state.category_clusters:
                        st.error("Category clustering not performed. Please complete Step 2 first.")
                    elif not hasattr(st.session_state, 'characteristic_clusters') or not st.session_state.characteristic_clusters:
                        st.error("Characteristic clustering not performed. Please complete Step 3 first.")
                    else:
                        # This function would implement the automatic taxonomy generation
                        # based on the clustering results
                        auto_generate_taxonomy_from_clusters()
                        st.success("Taxonomy auto-generated from clusters!")
                        st.rerun()
        
        with advanced_tabs[2]:
            st.markdown("""
            Export your taxonomy in JSON format for use in other applications.
            This will save all nodes, relationships, and metadata.
            """)
            
            # Create JSON export
            taxonomy_json = st.session_state.taxonomy_builder.export_taxonomy(format="json")
            
            # Create download button
            st.download_button(
                label="Download Taxonomy (JSON)",
                data=taxonomy_json,
                file_name="taxonomy_export.json",
                mime="application/json"
            )


# Helper function to display element grid
def display_element_grid(elements):
    """Display elements in a responsive grid layout"""
    # Determine number of columns based on number of elements
    max_cols = 4
    num_cols = min(max_cols, len(elements))
    
    # Create rows of elements
    for i in range(0, len(elements), num_cols):
        row_elements = elements[i:i+num_cols]
        cols = st.columns(num_cols)
        
        for j, element in enumerate(row_elements):
            with cols[j]:
                # Create a card-like container for each element
                with st.container(border=True):
                    # Element name as header
                    element_type_color = "#e6f3ff" if element["type"] == "category" else "#fff0e6"
                    st.markdown(f"""
                    <div style="background-color: {element_type_color}; padding: 5px; border-radius: 3px; margin-bottom: 5px">
                        <strong>{element["name"]}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Element type and source info
                    st.caption(f"Type: {element['type']}")
                    if "source" in element and element["source"]:
                        st.caption(f"Source: {element['source']}")
                    
                    # Description with truncation
                    if "description" in element and element["description"]:
                        desc = element["description"]
                        if len(desc) > 100:
                            desc = desc[:97] + "..."
                        st.caption(desc)
                    
                    # Action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Select", key=f"select_{element['id']}"):
                            select_element(element["id"])
                    with col2:
                        if st.button("Add to Root", key=f"root_{element['id']}"):
                            add_element_to_taxonomy(element["id"], "root")
                            st.rerun()


def auto_generate_taxonomy_from_clusters():
    """Generate a taxonomy automatically from HDBSCAN clustering results with hierarchy optimization"""
    # Clear existing taxonomy except root
    nodes_to_keep = ["root"]
    
    # Remove all nodes except root
    st.session_state.taxonomy_builder.nodes = [n for n in st.session_state.taxonomy_builder.nodes if n["id"] in nodes_to_keep]
    st.session_state.taxonomy_builder.edges = [e for e in st.session_state.taxonomy_builder.edges if e["source"] in nodes_to_keep and e["target"] in nodes_to_keep]
    
    # Get category cluster information
    category_ids = st.session_state.category_clusters['element_ids']
    category_cluster_labels = st.session_state.category_clusters['cluster_labels']
    category_map = {cat['id']: cat for cat in st.session_state.categories}
    
    # Get characteristic cluster information
    characteristic_ids = st.session_state.characteristic_clusters['element_ids']
    characteristic_cluster_labels = st.session_state.characteristic_clusters['cluster_labels']
    characteristic_map = {char['id']: char for char in st.session_state.characteristics}
    
    # Create category cluster mapping
    category_clusters = {}
    for i, cluster_label in enumerate(category_cluster_labels):
        if cluster_label not in category_clusters:
            category_clusters[cluster_label] = []
        
        category_id = category_ids[i]
        if category_id in category_map:
            category_clusters[cluster_label].append(category_map[category_id])
    
    # Create characteristic cluster mapping
    characteristic_clusters = {}
    for i, cluster_label in enumerate(characteristic_cluster_labels):
        if cluster_label not in characteristic_clusters:
            characteristic_clusters[cluster_label] = []
        
        characteristic_id = characteristic_ids[i]
        if characteristic_id in characteristic_map:
            characteristic_clusters[cluster_label].append(characteristic_map[characteristic_id])
    
    # First, add category cluster nodes
    cluster_node_ids = {}  # Map cluster labels to node IDs
    
    for cluster_label, categories in category_clusters.items():
        # Skip empty clusters
        if not categories:
            continue
        
        # Handle noise points (cluster_label = -1) specially
        if cluster_label == -1:
            # Add noise points directly to root
            for category in categories:
                node_id = st.session_state.taxonomy_builder.add_node(
                    name=category['name'],
                    node_type="category",
                    parent_id="root",
                    description=f"{category.get('description', '')} (Unclustered point)",
                    source=category.get('source', 'auto-generated'),
                    original_id=category['id'],
                    metadata={"cluster_label": -1}
                )
            continue
            
        # OPTIMIZATION: For single-element clusters, add directly to root instead of creating an intermediate cluster
        if len(categories) == 1:
            # Add the single category directly to root
            node_id = st.session_state.taxonomy_builder.add_node(
                name=categories[0]['name'],
                node_type="category",
                parent_id="root",
                description=categories[0].get('description', ''),
                source=categories[0].get('source', 'auto-generated'),
                original_id=categories[0]['id'],
                metadata={"cluster_label": cluster_label}
            )
            # Still store the node ID for characteristic attachment
            cluster_node_ids[cluster_label] = node_id
        else:
            # Create a name for the multi-element cluster
            category_names = [cat['name'] for cat in categories]
            words = [word.lower() for name in category_names for word in name.split()]
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Only consider words of length > 3
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_words:
                cluster_name = "Category Group: " + " ".join([word for word, _ in top_words])
            else:
                cluster_name = f"Category Group {cluster_label}"
            
            # Add cluster node
            node_id = st.session_state.taxonomy_builder.add_node(
                name=cluster_name,
                node_type="category",
                parent_id="root",
                description=f"Auto-generated cluster containing {len(categories)} categories",
                source="auto-generated",
                metadata={"cluster_label": cluster_label}
            )
            
            cluster_node_ids[cluster_label] = node_id
            
            # Add individual category nodes
            for category in categories:
                st.session_state.taxonomy_builder.add_node(
                    name=category['name'],
                    node_type="category",
                    parent_id=node_id,
                    description=category.get('description', ''),
                    source=category.get('source', 'auto-generated'),
                    original_id=category['id']
                )
    
    # Then, add characteristic cluster nodes
    for cluster_label, characteristics in characteristic_clusters.items():
        # Skip empty clusters
        if not characteristics:
            continue
        
        # Handle noise points specially
        if cluster_label == -1:
            # Add noise points directly to root
            for characteristic in characteristics:
                st.session_state.taxonomy_builder.add_node(
                    name=characteristic['name'],
                    node_type="characteristic",
                    parent_id="root",
                    description=f"{characteristic.get('description', '')} (Unclustered point)",
                    source=characteristic.get('source', 'auto-generated'),
                    original_id=characteristic['id'],
                    metadata={"cluster_label": -1}
                )
            continue
            
        # Find best matching category cluster using similarity
        best_match_score = -1
        best_match_cluster = None
        
        # For simplicity, we'll match based on the first characteristic in the cluster
        if characteristics:
            char_embedding = st.session_state.embedding_calculator.embeddings.get(characteristics[0]['id'])
            if char_embedding is not None:
                for cat_cluster_label, cat_cluster_node_id in cluster_node_ids.items():
                    # Get the categories in this cluster
                    cats_in_cluster = category_clusters.get(cat_cluster_label, [])
                    if cats_in_cluster:
                        # Use the first category for matching
                        cat_embedding = st.session_state.embedding_calculator.embeddings.get(cats_in_cluster[0]['id'])
                        if cat_embedding is not None:
                            # Calculate similarity
                            similarity = 1 - cosine(char_embedding, cat_embedding)
                            if similarity > best_match_score:
                                best_match_score = similarity
                                best_match_cluster = cat_cluster_node_id
        
        # If no match found, connect to root
        parent_id = best_match_cluster if best_match_cluster else "root"
        
        # OPTIMIZATION: For single-element clusters, attach directly to parent without intermediate node
        if len(characteristics) == 1:
            # Add the single characteristic directly to the parent
            st.session_state.taxonomy_builder.add_node(
                name=characteristics[0]['name'],
                node_type="characteristic",
                parent_id=parent_id,
                description=characteristics[0].get('description', ''),
                source=characteristics[0].get('source', 'auto-generated'),
                original_id=characteristics[0]['id']
            )
        else:
            # Create a name for the multi-element cluster
            characteristic_names = [char['name'] for char in characteristics]
            words = [word.lower() for name in characteristic_names for word in name.split()]
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Only consider words of length > 3
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_words:
                cluster_name = "Characteristic Group: " + " ".join([word for word, _ in top_words])
            else:
                cluster_name = f"Characteristic Group {cluster_label}"
            
            # Add characteristic group node
            char_group_node_id = st.session_state.taxonomy_builder.add_node(
                name=cluster_name,
                node_type="characteristic",
                parent_id=parent_id,
                description=f"Auto-generated cluster containing {len(characteristics)} characteristics",
                source="auto-generated",
                metadata={"cluster_label": cluster_label}
            )
            
            # Add individual characteristic nodes
            for characteristic in characteristics:
                st.session_state.taxonomy_builder.add_node(
                    name=characteristic['name'],
                    node_type="characteristic",
                    parent_id=char_group_node_id,
                    description=characteristic.get('description', ''),
                    source=characteristic.get('source', 'auto-generated'),
                    original_id=characteristic['id']
                )
    
    # Update unassigned elements to reflect the new taxonomy
    # All elements have been placed, so clear the unassigned list
    st.session_state.unassigned_elements = []

#############################################
# MAIN APPLICATION
#############################################

def main():
    """Main application function"""
    st.set_page_config(page_title="Agent 1: The Aggregator", page_icon="🌳", layout="wide")
    
    # Initialize session state
    initialize_state()
    
    # Create sidebar for navigation and status
    with st.sidebar:
        st.title("Agent 1: The Aggregator")
        
        # Step navigation
        st.header("Navigation")
        step = st.radio(
            "Select Step:",
            options=[
                "Step 1: Load Data",
                "Step 2: Cluster Categories",
                "Step 3: Cluster Characteristics",
                "Step 4: Setup LLM & RAG",
                "Step 5: Build Taxonomy"
            ],
            index=["Step 1: Load Data", 
                   "Step 2: Cluster Categories", 
                   "Step 3: Cluster Characteristics",
                   "Step 4: Setup LLM & RAG",
                   "Step 5: Build Taxonomy"].index(st.session_state.current_step),
            key="step_selection"
        )
        
        # Update current step
        st.session_state.current_step = step
        
        # Status section
        st.header("Status")
        st.text(st.session_state.status_message)
        
        # Data counts
        st.subheader("Data")
        st.text(f"Categories: {len(st.session_state.categories)}")
        st.text(f"Characteristics: {len(st.session_state.characteristics)}")
        
        # LLM status
        st.subheader("LLM Status")
        if st.session_state.llm_manager.selected_model:
            st.text(f"Model: {st.session_state.llm_manager.selected_model}")
        else:
            st.text("No model selected")
        
        # RAG sources
        st.subheader("RAG Sources")
        st.text(f"Taxonomies: {len(st.session_state.rag_manager.taxonomy_sources)}")
        st.text(f"Literature: {len(st.session_state.rag_manager.literature_sources)}")
        st.text(f"Custom: {len(st.session_state.rag_manager.custom_sources)}")
        
        # Taxonomy status
        st.subheader("Taxonomy")
        st.text(f"Nodes: {len(st.session_state.taxonomy_builder.nodes)}")
        st.text(f"Unassigned: {len(st.session_state.unassigned_elements)}")
        
        # Help text
        st.subheader("Help")
        st.info("Follow the steps in order from top to bottom. Each step builds on the previous one.")
    
    # Step 1: Load Data
    if st.session_state.current_step == "Step 1: Load Data":
        st.header("Step 1: Load Data")
        st.subheader("Upload taxonomy data from Agent 2 (The Preparator)")
        
        st.write("""
        In this step, you'll upload the JSON file exported from Agent 2 (The Preparator).
        The file should contain processed taxonomy elements including categories and characteristics.
        """)
        
        uploaded_file = st.file_uploader("Choose a JSON file", type="json", key="json_uploader")
        
        if uploaded_file is not None:
            if st.button("Load Data", key="load_data_btn"):
                load_data(uploaded_file)
        
        # Add explanation about embeddings
        if st.session_state.categories or st.session_state.characteristics:
            # Full width container for embeddings section
            st.markdown("## Word Embeddings")
            st.write("""
            **What are embeddings?** Word embeddings are numerical representations of text that capture their semantic meaning. 
            Similar concepts will have similar embedding vectors, even if they use different terminology.
            
            **How are they used?** The embeddings will enable us to:
            1. Identify semantically similar taxonomy elements
            2. Group related elements into clusters
            3. Create a hierarchical structure based on these relationships
            
            We use the Jina Embeddings v3 model, which captures semantic meaning well and works with 
            terminology from many domains.
            """)
            
            # Place the compute button outside of columns for better layout
            if st.button("Compute Embeddings", key="compute_embeddings_btn"):
                compute_embeddings()
            
            st.caption(f"Using model: {st.session_state.embedding_calculator.model_name}")
                
            # Add a "Next" button to guide users
            if st.session_state.embedding_calculator.embeddings:
                if st.button("Next: Cluster Categories →", key="next_to_categories_btn"):
                    st.session_state.current_step = "Step 2: Cluster Categories"
                    st.rerun()
    
    # Step 2: Cluster Categories
    elif st.session_state.current_step == "Step 2: Cluster Categories":
        st.header("Step 2: Cluster Categories")
        st.subheader("Perform HDBSCAN clustering on categories (inner nodes)")
        
        # Explanation of HDBSCAN clustering
        st.write("""
        ### What is HDBSCAN Clustering?
        
        HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a modern clustering 
        algorithm that excels at finding clusters of varying shapes and sizes. Unlike traditional algorithms, 
        HDBSCAN:
        
        1. Automatically determines the optimal number of clusters
        2. Identifies noise points that don't belong to any cluster
        3. Works well with high-dimensional data like our embeddings
        4. Handles clusters of varying densities and sizes
        
        This step specifically focuses on categories (inner nodes) to create the base structure of the taxonomy.
        """)
        
        # Check if embeddings are available
        if not st.session_state.embedding_calculator.embeddings:
            st.warning("⚠️ Embeddings not calculated. Please return to Step 1 and compute embeddings first.")
        else:
            # Configure clustering parameters with explanations
            st.subheader("Clustering Parameters")
            
            # Min cluster size explanation
            st.write("""
            **Minimum Cluster Size:** The minimum number of elements required to form a cluster.
            - **Lower values** (e.g., 2-3): Allow smaller, more specific clusters
            - **Higher values** (e.g., 5+): Require more elements to form clusters, resulting in broader groupings
            
            For taxonomy building, smaller values often work better to capture specific relationships.
            """)
            
            min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                help="Minimum samples needed to form a cluster"
            )
            
            # Min samples explanation
            st.write("""
            **Minimum Samples:** Determines how conservative the clustering is.
            - **Lower values** (close to min_cluster_size): More clusters and less noise
            - **Higher values**: More conservative clustering with more noise points
            
            Usually, set this equal to or just below the minimum cluster size.
            """)
            
            min_samples = st.slider(
                "Minimum Samples",
                min_value=1,
                max_value=min_cluster_size,
                value=min(2, min_cluster_size),
                step=1,
                help="Controls cluster conservativeness"
            )
            
            # Epsilon explanation
            st.write("""
            **Cluster Selection Epsilon:** Controls semantic distance threshold for clusters.
            - **0.0-0.5**: Very strict (only grouping highly similar concepts)
            - **0.5-1.00**: Balanced (it captures meaningful semantic relationships while avoiding unrelated concepts)
            - **1.00-2.00**: Inclusive (it allows concepts with low similarity or even opposite semantics to be grouped)

            We normalize vectors because word embeddings encode meaning in their direction, not magnitude. This normalization ensures clustering focuses purely on semantic relationships between concepts rather than arbitrary vector lengths. As a result, epsilon values directly reflect semantic similarity - lower values create clusters of closely related concepts, while higher values allow more broadly related concepts to group together.
            """)
            
            epsilon = st.slider(
                "Cluster Selection Epsilon",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.05,
                help="Distance threshold for expanding clusters"
            )
            
            # Perform clustering button
            if st.button("Cluster Categories", key="cluster_categories_btn"):
                cluster_categories(min_cluster_size, min_samples, epsilon)
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Load Data", key="back_to_load_btn"):
                    st.session_state.current_step = "Step 1: Load Data"
                    st.rerun()
            
            with col2:
                # Only enable Next if clustering was performed
                if 'category_clusters' in st.session_state and st.session_state.category_clusters:
                    if st.button("Next: Cluster Characteristics →", key="next_to_chars_btn"):
                        st.session_state.current_step = "Step 3: Cluster Characteristics"
                        st.rerun()
    
    # Step 3: Cluster Characteristics
    elif st.session_state.current_step == "Step 3: Cluster Characteristics":
        st.header("Step 3: Cluster Characteristics")
        st.subheader("Perform HDBSCAN clustering on characteristics (leaves)")
        
        # Explanation of HDBSCAN clustering for characteristics
        st.write("""
        ### Clustering Characteristics (Leaves)
        
        In this step, we apply HDBSCAN clustering to the characteristics (leaves) of the taxonomy.
        This process is similar to clustering categories but focuses on the more specific elements.
        
        **Why use HDBSCAN for characteristics?**
        
        Characteristics often have more varied and specific meanings than categories, which makes HDBSCAN ideal because:
        1. It can identify clusters of different shapes and sizes
        2. It can detect outliers (noise points) that don't fit well with other characteristics
        3. It automatically determines the optimal number of clusters based on the data
        
        The clusters identified here will be connected to the category clusters in the final step.
        """)
        
        # Check if embeddings are available
        if not st.session_state.embedding_calculator.embeddings:
            st.warning("⚠️ Embeddings not calculated. Please return to Step 1 and compute embeddings first.")
        else:
            # Configure clustering parameters with explanations
            st.subheader("Clustering Parameters")
            
            # Min cluster size explanation
            st.write("""
            **Minimum Cluster Size:** The minimum number of elements required to form a cluster.
            - **Lower values** (e.g., 2-3): Allow smaller, more specific clusters
            - **Higher values** (e.g., 5+): Require more elements to form clusters, resulting in broader groupings
            
            For characteristics, you might want different values than for categories.
            """)
            
            char_min_cluster_size = st.slider(
                "Minimum Cluster Size",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                key="char_min_cluster_size",
                help="Minimum samples needed to form a cluster"
            )
            
            # Min samples explanation
            st.write("""
            **Minimum Samples:** Determines how conservative the clustering is.
            - **Lower values** (close to min_cluster_size): More clusters and less noise
            - **Higher values**: More conservative clustering with more noise points
            
            Usually, set this equal to or just below the minimum cluster size.
            """)
            
            char_min_samples = st.slider(
                "Minimum Samples",
                min_value=1,
                max_value=char_min_cluster_size,
                value=min(2, char_min_cluster_size),
                step=1,
                key="char_min_samples",
                help="Controls cluster conservativeness"
            )
            
            # Epsilon explanation
            st.write("""
            **Cluster Selection Epsilon:** Controls semantic distance threshold for clusters.
            - **0.0-0.5**: Very strict (only grouping highly similar concepts)
            - **0.5-1.00**: Balanced (it captures meaningful semantic relationships while avoiding unrelated concepts)
            - **1.00-2.00**: Inclusive (it allows concepts with low similarity or even opposite semantics to be grouped)

            We normalize vectors because word embeddings encode meaning in their direction, not magnitude. This normalization ensures clustering focuses purely on semantic relationships between concepts rather than arbitrary vector lengths. As a result, epsilon values directly reflect semantic similarity - lower values create clusters of closely related concepts, while higher values allow more broadly related concepts to group together.
            """)
            
            char_epsilon = st.slider(
                "Cluster Selection Epsilon",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.05,
                key="char_epsilon",
                help="Distance threshold for expanding clusters"
            )
            
            # Perform clustering button
            if st.button("Cluster Characteristics", key="cluster_characteristics_btn"):
                cluster_characteristics(char_min_cluster_size, char_min_samples, char_epsilon)
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Cluster Categories", key="back_to_cats_btn"):
                    st.session_state.current_step = "Step 2: Cluster Categories"
                    st.rerun()
            
            with col2:
                # Only enable Next if clustering was performed
                if 'characteristic_clusters' in st.session_state and st.session_state.characteristic_clusters:
                    if st.button("Next: Setup LLM & RAG →", key="next_to_llm_rag_btn"):
                        st.session_state.current_step = "Step 4: Setup LLM & RAG"
                        st.rerun()
    
    # Step 4: Setup LLM & RAG
    elif st.session_state.current_step == "Step 4: Setup LLM & RAG":
        st.header("Step 4: Setup LLM & RAG")
        st.subheader("Configure LLM and RAG for intelligent taxonomy construction")
        
        st.write("""
        ### Human-Centered AI for Taxonomy Construction
        
        In this step, we'll configure the AI components that will assist in taxonomy construction:
        
        1. **Local LLM Integration**: Connect to a local LLM via LM Studio to provide intelligent assistance
        2. **Retrieval-Augmented Generation (RAG)**: Add knowledge sources to inform the LLM's suggestions
        
        These components will work together with your expert judgment to create a high-quality taxonomy.
        """)
        
        # LLM Setup section
        setup_llm()
        
        # Add some spacing
        st.markdown("---")
        
        # RAG Setup section 
        setup_rag()
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Cluster Characteristics", key="back_to_chars_from_llm_btn"):
                st.session_state.current_step = "Step 3: Cluster Characteristics"
                st.rerun()
        
        with col2:
            # Check if LLM is configured
            llm_ready = st.session_state.llm_manager.selected_model is not None
            
            if llm_ready:
                if st.button("Next: Build Taxonomy →", key="next_to_build_btn"):
                    st.session_state.current_step = "Step 5: Build Taxonomy"
                    st.rerun()
            else:
                st.warning("Please configure and select an LLM model before proceeding")
    
    # Step 5: Build Taxonomy
    elif st.session_state.current_step == "Step 5: Build Taxonomy":
        st.header("Step 5: Build Taxonomy")
        st.subheader("Interactive Taxonomy Construction")
        
        # Check prerequisites
        if not st.session_state.embedding_calculator.embeddings:
            st.warning("⚠️ Embeddings not calculated. Please return to Step 1 and compute embeddings first.")
            return
        
        if not st.session_state.llm_manager.selected_model:
            st.warning("⚠️ LLM not configured. Please return to Step 4 and set up an LLM model first.")
            return
        
        # Instructions expander
        with st.expander("📚 Instructions", expanded=False):
            st.markdown("""
            ### How to Build Your Taxonomy
            
            Follow these steps to create your merged taxonomy:
            
            1. **Browse Elements**: Review the unassigned elements in the left panel
            2. **Select Elements**: Click "Select" on an element you want to place
            3. **Review Information**: See element details and similarity analysis
            4. **Place in Taxonomy**: Either:
               - Accept AI suggestions (right panel)
               - Manually place elements (center panel)
               - Add to root (left panel)
            5. **Add Structure**: Create new categories as needed
            6. **Refine**: Review and adjust the taxonomy
            7. **Export**: Save your completed taxonomy
            
            **Tips:**
            - Let the AI help, but make the final decisions yourself
            - Use similarity scores as guidance
            - Add context in the right panel to improve AI suggestions
            - Work in small batches, starting with high-level categories
            """)
        
        # Display the builder interface
        taxonomy_builder_interface()
        
        # Navigation button to return to previous step
        if st.button("← Back to Setup LLM & RAG", key="back_to_llm_rag_btn"):
            st.session_state.current_step = "Step 4: Setup LLM & RAG"
            st.rerun()


if __name__ == "__main__":
    main()