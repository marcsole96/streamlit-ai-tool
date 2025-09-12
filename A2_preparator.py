import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import requests
import os
import io
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import networkx as nx
import hdbscan  # Added HDBSCAN import
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import normalize  # Add this import for normalization


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
    
    def compute_similarity_matrix(self, status_callback=None):
        """
        Compute similarity matrix between all elements
        
        Args:
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
        
        # Get all element IDs
        element_ids = list(self.embeddings.keys())
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
    
    def find_clusters(self, min_cluster_size=2, min_samples=None, cluster_selection_epsilon=0.5, status_callback=None):
        """
        Find clusters of similar elements using HDBSCAN
        
        Args:
            min_cluster_size: The minimum size of clusters (default: 2)
            min_samples: The number of samples in a neighborhood for a point to be considered a core point (default: same as min_cluster_size)
            cluster_selection_epsilon: Distance threshold for cluster extraction
            status_callback: Function to call with status updates
        
        Returns:
            List of clusters (each cluster is a list of element IDs)
        """
        if not self.embeddings:
            if status_callback:
                status_callback("No embeddings available. Please compute embeddings first.")
            return []
        
        if status_callback:
            status_callback(f"Finding clusters with HDBSCAN (min_cluster_size={min_cluster_size}, epsilon={cluster_selection_epsilon})...")
        
        # Extract element IDs and embeddings
        element_ids = list(self.embeddings.keys())
        embedding_matrix = np.vstack([self.embeddings[id] for id in element_ids])
        
        # Set min_samples equal to min_cluster_size if not specified
        if min_samples is None:
            min_samples = min_cluster_size
        
        # Normalize embeddings to unit length for cosine similarity effect
        if status_callback:
            status_callback("Normalizing embeddings for cosine similarity...")
        
        normalized_matrix = normalize(embedding_matrix, norm='l2', axis=1)
        
        # Run HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean',  # Still using euclidean, but on normalized vectors
            cluster_selection_method='eom'  # 'eom' is "excess of mass"
        )
        
        # Fit the clusterer to our normalized data
        cluster_labels = clusterer.fit_predict(normalized_matrix)
        
        # Reset clusters list
        self.clusters = []
        
        # Organize elements by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # HDBSCAN labels noise points as -1
                # Create a singleton cluster for noise points
                self.clusters.append([element_ids[i]])
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(element_ids[i])
        
        # Add non-noise clusters to the list
        for label in sorted(clusters.keys()):
            self.clusters.append(clusters[label])
        
        if status_callback:
            num_real_clusters = len([c for c in self.clusters if len(c) > 1])
            status_callback(f"Found {num_real_clusters} clusters (excluding singletons)")
        
        return self.clusters
    
    def get_element_by_id(self, element_id):
        """Get the full element data for an element ID"""
        return self.element_map.get(element_id)


def initialize_state():
    """Initialize the session state with default values if not already set"""
    # Basic app configuration
    if 'api_url' not in st.session_state:
        st.session_state.api_url = "http://127.0.0.1:1234/v1/chat/completions"
    
    if 'models_url' not in st.session_state:
        url_parts = st.session_state.api_url.split("/")
        if "v1" in url_parts:
            server_part = "/".join(url_parts[:3])  # http://server:port
            st.session_state.models_url = f"{server_part}/v1/models"
        else:
            base_url = "/".join(st.session_state.api_url.split("/")[:-1])
            st.session_state.models_url = f"{base_url}/models"
    
    # System role for LLM
    if 'default_system_role' not in st.session_state:
        st.session_state.default_system_role = """You are a taxonomy preparation assistant. Your task is to analyze taxonomy elements and identify potential synonyms or duplicate concepts.
Focus on identifying terms that refer to the same underlying concept, even if they use different wording.
Provide clear reasoning for your suggestions and be thorough in your analysis."""
    
    if 'system_role' not in st.session_state:
        st.session_state.system_role = st.session_state.default_system_role
    
    # Available models list
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = ""
    
    # Data structures for storing taxonomy elements
    if 'characteristics' not in st.session_state:
        st.session_state.characteristics = []  # Leaves
    
    if 'categories' not in st.session_state:
        st.session_state.categories = []  # Inner nodes
    
    if 'processed_characteristics' not in st.session_state:
        st.session_state.processed_characteristics = []
    
    if 'processed_categories' not in st.session_state:
        st.session_state.processed_categories = []
    
    if 'decisions' not in st.session_state:
        st.session_state.decisions = []
    
    # Embedding calculator
    if 'embedding_calculator' not in st.session_state:
        st.session_state.embedding_calculator = EmbeddingCalculator()
    
    # Current navigation section
    if 'current_section' not in st.session_state:
        st.session_state.current_section = "Welcome"
    
    # HDBSCAN parameters
    if 'min_cluster_size' not in st.session_state:
        st.session_state.min_cluster_size = 2
    
    if 'min_samples' not in st.session_state:
        st.session_state.min_samples = 2
    
    if 'cluster_selection_epsilon' not in st.session_state:
        st.session_state.cluster_selection_epsilon = 0.5
    
    if 'selected_cluster' not in st.session_state:
        st.session_state.selected_cluster = None
    
    if 'decision_type' not in st.session_state:
        st.session_state.decision_type = "merge"
    
    if 'merge_term' not in st.session_state:
        st.session_state.merge_term = ""
    
    if 'merge_source' not in st.session_state:
        st.session_state.merge_source = "Combined"
    
    if 'process_type' not in st.session_state:
        st.session_state.process_type = "both"
    
    # Interactive heatmap variables
    if 'custom_clusters' not in st.session_state:
        st.session_state.custom_clusters = []
    
    if 'selected_matrix_cells' not in st.session_state:
        st.session_state.selected_matrix_cells = set()
    
    if 'selected_element_ids' not in st.session_state:
        st.session_state.selected_element_ids = set()
    
    if 'heatmap_threshold' not in st.session_state:
        st.session_state.heatmap_threshold = 0.75
    
    if 'plotly_click_data' not in st.session_state:
        st.session_state.plotly_click_data = None
    
    # LLM suggestions
    if 'llm_suggestions' not in st.session_state:
        st.session_state.llm_suggestions = ""
    
    # Status message
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Ready"


def settings_page():
    """Render the settings page"""
    st.title("Settings")
    st.write("Configure how the application connects to the Language Model and processes your taxonomy data.")
    
    with st.expander("LLM Model Selection", expanded=True):
        st.write("Select which AI model to use for suggesting synonyms and providing analyses.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.available_models:
                model_index = 0
                if st.session_state.current_model in st.session_state.available_models:
                    model_index = st.session_state.available_models.index(st.session_state.current_model)
                
                st.session_state.current_model = st.selectbox(
                    "Select Model:", 
                    st.session_state.available_models,
                    index=model_index
                )
            else:
                st.selectbox("Select Model:", ["No models available"])
        
        with col2:
            if st.button("Refresh Models"):
                fetch_models()
        
        api_url = st.text_input("API Endpoint:", value=st.session_state.api_url, key="api_url_input")
        if st.button("Update Endpoint"):
            update_endpoint(api_url)
    
    with st.expander("Embedding Model Configuration", expanded=True):
        st.write("Configure the embedding model used for semantic similarity calculation.")
        st.write("Embedding Model: jinaai/jina-embeddings-v3")
        if st.button("Test Embedding Model"):
            test_embedding_model()
    
    with st.expander("LLM System Role Configuration", expanded=True):
        st.write("The system role defines how the AI approaches synonym detection. You can customize it to emphasize different aspects of analysis:")
        
        system_role = st.text_area(
            "System Role:", 
            value=st.session_state.system_role,
            height=200
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save System Role"):
                st.session_state.system_role = system_role
                st.success("System role saved successfully!")
        
        with col2:
            if st.button("Reset to Default"):
                st.session_state.system_role = st.session_state.default_system_role
                st.rerun()
    
    with st.expander("Test LLM Connection", expanded=True):
        st.write("Verify that the application can connect to the selected model.")
        if st.button("Test Connection"):
            test_llm_connection()


def data_input_page():
    """Render the data input page"""
    st.title("Data Input")
    st.write("Enter the taxonomy elements (characteristics and categories) that need to be processed.")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Manual Input", "CSV Import", "JSON Import"])
    
    with tab1:
        manual_input_tab()
    
    with tab2:
        csv_input_tab()
    
    with tab3:
        json_input_tab()
    
    # Data preview section
    st.subheader("Data Preview")
    preview_tab1, preview_tab2 = st.tabs(["Characteristics", "Categories"])
    
    with preview_tab1:
        if st.session_state.characteristics:
            st.dataframe(pd.DataFrame(st.session_state.characteristics))
        else:
            st.write("No characteristics added yet.")
    
    with preview_tab2:
        if st.session_state.categories:
            st.dataframe(pd.DataFrame(st.session_state.categories))
        else:
            st.write("No categories added yet.")
    
    # Control buttons
    element_count = len(st.session_state.characteristics) + len(st.session_state.categories)
    st.write(f"Total elements: {element_count} ({len(st.session_state.characteristics)} characteristics, {len(st.session_state.categories)} categories)")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Clear All Data"):
            clear_all_data()
    
    with col3:
        proceed_btn = st.button("Proceed to Analysis →", key="proceed_analysis_btn")
        if proceed_btn:
            if element_count > 0:
                st.session_state.current_section = "Similarity Analysis"
                st.rerun()
            else:
                st.error("Please add some data before proceeding to analysis.")


def manual_input_tab():
    """Render the manual input tab"""
    # Create tabs for characteristics and categories
    char_tab, cat_tab = st.tabs(["Characteristics", "Categories"])
    
    with char_tab:
        st.write("Enter one characteristic per line:")
        st.caption("Format: [name] | [source] | [description]")
        st.caption("Example: Lines of Code | Taxonomy 1 | Count of lines in source code")
        
        char_input = st.text_area(
            "Characteristics Input", 
            height=200,
            key="char_manual_input",
            help="Enter one characteristic per line using the format: name | source | description"
        )
    
    with cat_tab:
        st.write("Enter one category per line:")
        st.caption("Format: [name] | [source] | [parent]")
        st.caption("Example: Size Metrics | Taxonomy 2 | Effort Drivers")
        
        cat_input = st.text_area(
            "Categories Input", 
            height=200,
            key="cat_manual_input",
            help="Enter one category per line using the format: name | source | parent"
        )
    
    if st.button("Add to Preview", key="manual_add_btn"):
        add_manual_input(char_input, cat_input)



def csv_input_tab():
    """Render the CSV import tab with multiple file upload support"""
    st.write("Import taxonomy elements from CSV files. Each file should have the required columns.")
    
    with st.expander("CSV Format Help"):
        st.markdown("""
        ### CSV FILE FORMAT:

        The CSV file should include the following columns:
        - name (required): The name of the element
        - type (required): Either "characteristic" or "category"
        - source (required): The source taxonomy
        - description (optional): For characteristics, a description of what they measure
        - parent (optional): For categories, the parent category name

        Example CSV content:
        ```
        name,type,source,description,parent
        Lines of Code,characteristic,Taxonomy 1,"Count of code lines",
        Function Points,characteristic,Taxonomy 2,"Measure of functionality",
        Size Metrics,category,Taxonomy 1,,"Effort Drivers"
        Technical Factors,category,Taxonomy 2,,"Project Characteristics"
        ```
        """)
    
    # Use accept_multiple_files=True for multiple file uploading
    uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True, key="csv_uploader")
    
    if uploaded_files:
        # Display info about uploaded files
        st.write(f"📂 {len(uploaded_files)} CSV files uploaded")
        
        # Create tabs for file previews
        file_tabs = st.tabs([f"File {i+1}: {file.name}" for i, file in enumerate(uploaded_files)])
        
        # Show preview for each file in its respective tab
        for i, (tab, file) in enumerate(zip(file_tabs, uploaded_files)):
            with tab:
                try:
                    df = pd.read_csv(file)
                    st.write(f"Preview of {file.name}:")
                    st.dataframe(df.head(5))
                    
                    # Check for required columns
                    required_columns = ["name", "type", "source"]
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"⚠️ File is missing required columns: {', '.join(missing_columns)}")
                    else:
                        st.success("✅ File format looks good!")
                        
                except Exception as e:
                    st.error(f"Error previewing file: {str(e)}")
                    
        # Add a button to import all files
        if st.button("Import All CSV Files"):
            import_multiple_csv_files(uploaded_files)

def json_input_tab():
    """Render the JSON import tab with multiple file upload support"""
    st.write("Import taxonomy elements from JSON files. Each file should use one of the supported formats.")
    
    with st.expander("JSON Format Help"):
        st.markdown("""
        ### JSON FILE FORMAT:

        Option 1 - Array of elements:
        ```json
        [
          {
            "name": "Lines of Code",
            "type": "characteristic",
            "source": "Taxonomy 1",
            "description": "Count of code lines"
          },
          {
            "name": "Size Metrics",
            "type": "category",
            "source": "Taxonomy 1",
            "parent": "Effort Drivers"
          }
        ]
        ```

        Option 2 - Structured format:
        ```json
        {
          "characteristics": [
            {
              "name": "Lines of Code",
              "source": "Taxonomy 1",
              "description": "Count of code lines"
            }
          ],
          "categories": [
            {
              "name": "Size Metrics",
              "source": "Taxonomy 1",
              "parent": "Effort Drivers"
            }
          ]
        }
        ```
        """)
    
    # Use accept_multiple_files=True for multiple file uploading
    uploaded_files = st.file_uploader("Choose JSON files", type="json", accept_multiple_files=True, key="json_uploader")
    
    if uploaded_files:
        # Display info about uploaded files
        st.write(f"📂 {len(uploaded_files)} JSON files uploaded")
        
        # Create tabs for file previews
        file_tabs = st.tabs([f"File {i+1}: {file.name}" for i, file in enumerate(uploaded_files)])
        
        # Track if all files are valid
        all_valid = True
        
        # Show preview for each file in its respective tab
        for i, (tab, file) in enumerate(zip(file_tabs, uploaded_files)):
            with tab:
                try:
                    content = file.read().decode('utf-8')
                    json_data = json.loads(content)
                    
                    st.write(f"Preview of {file.name}:")
                    st.json(json_data)
                    
                    # Reset file pointer for later use
                    file.seek(0)
                    
                    # Basic validation
                    if isinstance(json_data, list):
                        if len(json_data) > 0:
                            st.success("✅ File format looks good (array of elements)")
                        else:
                            st.warning("⚠️ File contains an empty array")
                            all_valid = False
                    elif isinstance(json_data, dict) and ("characteristics" in json_data or "categories" in json_data):
                        st.success("✅ File format looks good (structured format)")
                    else:
                        st.error("⚠️ Unknown JSON format")
                        all_valid = False
                    
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON file: {str(e)}")
                    all_valid = False
                except Exception as e:
                    st.error(f"Error previewing file: {str(e)}")
                    all_valid = False
        
        # Add a button to import all files
        import_button = st.button("Import All JSON Files")
        
        if import_button:
            if all_valid:
                import_multiple_json_files(uploaded_files)
            else:
                st.error("Cannot import: One or more files have invalid format. Please fix the issues first.")


def import_multiple_csv_files(uploaded_files):
    """Import data from multiple uploaded CSV files"""
    total_char_count = 0
    total_cat_count = 0
    error_count = 0
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                progress = (i / len(uploaded_files))
                progress_bar.progress(progress)
                status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Reset file pointer to start
                uploaded_file.seek(0)
                
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Check for required columns
                required_columns = ["name", "type", "source"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                # Show error if required columns are missing
                if missing_columns:
                    st.error(f"CSV file {uploaded_file.name} is missing required columns: {', '.join(missing_columns)}")
                    error_count += 1
                    continue
                
                # Process each row
                file_char_count = 0
                file_cat_count = 0
                
                for _, row in df.iterrows():
                    element_type = row.get("type", "").strip().lower()
                    
                    # Process characteristics
                    if element_type == "characteristic":
                        char_id = f"char_{len(st.session_state.characteristics) + 1}"
                        
                        # Add to data structure
                        st.session_state.characteristics.append({
                            "id": char_id,
                            "name": row.get("name", "Unknown"),
                            "source": row.get("source", "Unknown"),
                            "description": row.get("description", ""),
                            "type": "characteristic"
                        })
                        
                        file_char_count += 1
                        total_char_count += 1
                    
                    # Process categories
                    elif element_type == "category":
                        cat_id = f"cat_{len(st.session_state.categories) + 1}"
                        
                        # Add to data structure
                        st.session_state.categories.append({
                            "id": cat_id,
                            "name": row.get("name", "Unknown"),
                            "source": row.get("source", "Unknown"),
                            "parent": row.get("parent", ""),
                            "type": "category"
                        })
                        
                        file_cat_count += 1
                        total_cat_count += 1
                
                # Show success for individual file
                st.info(f"File {uploaded_file.name}: Imported {file_char_count} characteristics and {file_cat_count} categories")
                
            except Exception as e:
                st.error(f"Error importing CSV file {uploaded_file.name}: {str(e)}")
                error_count += 1
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Show final success message
        if error_count == 0:
            st.success(f"Successfully imported {total_char_count} characteristics and {total_cat_count} categories from {len(uploaded_files)} files")
        else:
            st.warning(f"Imported {total_char_count} characteristics and {total_cat_count} categories from {len(uploaded_files) - error_count} files. {error_count} files had errors.")
    
    except Exception as e:
        st.error(f"Error during import: {str(e)}")


def import_multiple_json_files(uploaded_files):
    """Import data from multiple uploaded JSON files"""
    total_char_count = 0
    total_cat_count = 0
    error_count = 0
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # Update progress
                progress = (i / len(uploaded_files))
                progress_bar.progress(progress)
                status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Reset file pointer to start
                uploaded_file.seek(0)
                
                # Read the file content
                content = uploaded_file.read().decode('utf-8')
                
                # Parse JSON
                data = json.loads(content)
                
                # Counters for imported elements from this file
                file_char_count = 0
                file_cat_count = 0
                
                # Check the format of the JSON data
                if isinstance(data, list):
                    # Array of elements format
                    for element in data:
                        element_type = element.get("type", "").lower()
                        
                        # Process characteristics
                        if element_type == "characteristic":
                            char_id = f"char_{len(st.session_state.characteristics) + 1}"
                            
                            # Add to data structure
                            st.session_state.characteristics.append({
                                "id": char_id,
                                "name": element.get("name", "Unknown"),
                                "source": element.get("source", "Unknown"),
                                "description": element.get("description", ""),
                                "type": "characteristic"
                            })
                            
                            file_char_count += 1
                            total_char_count += 1
                        
                        # Process categories
                        elif element_type == "category":
                            cat_id = f"cat_{len(st.session_state.categories) + 1}"
                            
                            # Add to data structure
                            st.session_state.categories.append({
                                "id": cat_id,
                                "name": element.get("name", "Unknown"),
                                "source": element.get("source", "Unknown"),
                                "parent": element.get("parent", ""),
                                "type": "category"
                            })
                            
                            file_cat_count += 1
                            total_cat_count += 1
                
                elif "characteristics" in data and "categories" in data:
                    # Structured format with separate arrays
                    
                    # Import characteristics
                    for char in data.get("characteristics", []):
                        char_id = f"char_{len(st.session_state.characteristics) + 1}"
                        
                        # Add to data structure
                        st.session_state.characteristics.append({
                            "id": char_id,
                            "name": char.get("name", "Unknown"),
                            "source": char.get("source", "Unknown"),
                            "description": char.get("description", ""),
                            "type": "characteristic"
                        })
                        
                        file_char_count += 1
                        total_char_count += 1
                    
                    # Import categories
                    for cat in data.get("categories", []):
                        cat_id = f"cat_{len(st.session_state.categories) + 1}"
                        
                        # Add to data structure
                        st.session_state.categories.append({
                            "id": cat_id,
                            "name": cat.get("name", "Unknown"),
                            "source": cat.get("source", "Unknown"),
                            "parent": cat.get("parent", ""),
                            "type": "category"
                        })
                        
                        file_cat_count += 1
                        total_cat_count += 1
                
                else:
                    # Unknown format
                    st.error(f"File {uploaded_file.name}: Unknown JSON format. Expected array of elements or structured format.")
                    error_count += 1
                    continue
                
                # Show success for individual file
                st.info(f"File {uploaded_file.name}: Imported {file_char_count} characteristics and {file_cat_count} categories")
                
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON file {uploaded_file.name}: {str(e)}")
                error_count += 1
            except Exception as e:
                st.error(f"Error importing JSON file {uploaded_file.name}: {str(e)}")
                error_count += 1
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Show final success message
        if error_count == 0:
            st.success(f"Successfully imported {total_char_count} characteristics and {total_cat_count} categories from {len(uploaded_files)} files")
        else:
            st.warning(f"Imported {total_char_count} characteristics and {total_cat_count} categories from {len(uploaded_files) - error_count} files. {error_count} files had errors.")
    
    except Exception as e:
        st.error(f"Error during import: {str(e)}")



def similarity_analysis_page():
    """Render the similarity analysis page"""
    st.title("Similarity Analysis")
    st.write("Analyze semantic similarity between taxonomy elements to find potential synonyms.")
    
    # Control section
    with st.expander("Analysis Controls", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Select which element types to analyze:")
            
            process_type = st.radio(
                "Process:",
                ["Both Types", "Characteristics Only", "Categories Only"],
                index=["both", "characteristics", "categories"].index(st.session_state.process_type),
                horizontal=True,
                key="process_type_radio"
            )
            
            # Update process_type in session state
            if process_type == "Both Types":
                st.session_state.process_type = "both"
            elif process_type == "Characteristics Only":
                st.session_state.process_type = "characteristics"
            else:
                st.session_state.process_type = "categories"
        
        with col2:
            if st.button("Calculate Embeddings"):
                calculate_embeddings()
    
    # Only show the rest if embeddings have been calculated
    if hasattr(st.session_state.embedding_calculator, 'embeddings') and st.session_state.embedding_calculator.embeddings:
        # HDBSCAN parameter controls
        st.subheader("Clustering Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_cluster_size = st.slider(
                "Minimum Cluster Size:",
                min_value=2,
                max_value=10,
                value=st.session_state.min_cluster_size,
                step=1,
                help="The minimum number of elements required to form a cluster"
            )
        
        with col2:
            min_samples = st.slider(
                "Minimum Samples:",
                min_value=1,
                max_value=10,
                value=st.session_state.min_samples,
                step=1,
                help="The number of samples in a neighborhood for a point to be considered a core point"
            )
        
        # Add epsilon slider
        epsilon = st.slider(
            "Cluster Selection Epsilon:",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.cluster_selection_epsilon,
            step=0.05,
            help="Distance threshold for cluster extraction. Higher values create fewer, larger clusters."
        )
        
        # Update parameters and find clusters when changed
        params_changed = False
        if min_cluster_size != st.session_state.min_cluster_size:
            st.session_state.min_cluster_size = min_cluster_size
            params_changed = True
        
        if min_samples != st.session_state.min_samples:
            st.session_state.min_samples = min_samples
            params_changed = True
            
        if epsilon != st.session_state.cluster_selection_epsilon:
            st.session_state.cluster_selection_epsilon = epsilon
            params_changed = True
        
        if params_changed or not hasattr(st.session_state.embedding_calculator, 'clusters'):
            st.session_state.embedding_calculator.find_clusters(
                min_cluster_size=st.session_state.min_cluster_size,
                min_samples=st.session_state.min_samples,
                cluster_selection_epsilon=st.session_state.cluster_selection_epsilon,
                status_callback=lambda msg: st.session_state.update(status_message=msg)
            )
        
        # Visualization tabs - Updated tab name
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Automatically Detected Clusters", "Interactive Heatmap", "Similarity Matrix", "3D Visualization"])
        
        with viz_tab1:
            display_clusters()
        
        with viz_tab2:
            display_interactive_similarity_matrix()
        
        with viz_tab3:
            display_similarity_matrix()
        
        with viz_tab4:
            display_3d_visualization()
        
        # Decision section for selected cluster
        if st.session_state.selected_cluster:
            display_cluster_decision()
        
        # Navigation buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Input"):
                st.session_state.current_section = "Data Input"
                st.rerun()
        
        with col2:
            if st.button("Continue to Review →"):
                finalize_processing()
                st.session_state.current_section = "Review"
                st.rerun()
    else:
        st.info("Calculate embeddings first to view similarity analysis.")


def initialize_state_additions():
    """Additional session state initializations for new features"""
    # Initialize processed_element_ids to track which elements have been processed
    if 'processed_element_ids' not in st.session_state:
        st.session_state.processed_element_ids = set()
    
    # Initialize partial_selection to track subset of elements selected for merge
    if 'partial_selection' not in st.session_state:
        st.session_state.partial_selection = {}


def review_page():
    """Render the review and export page"""
    st.title("Review & Export")
    st.write("Review the results of synonym processing and export the final taxonomy.")
    
    # Processing summary
    with st.expander("Processing Summary", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Original Characteristics", len(st.session_state.characteristics))
            st.metric("Processed Characteristics", len(st.session_state.processed_characteristics))
            st.metric("Clusters Analyzed", 
                      len(st.session_state.embedding_calculator.clusters) if hasattr(st.session_state.embedding_calculator, 'clusters') else 0)
        
        with col2:
            st.metric("Original Categories", len(st.session_state.categories))
            st.metric("Processed Categories", len(st.session_state.processed_categories))
            st.metric("Decisions Made", len(st.session_state.decisions))
        
        # Calculate reduction
        total_orig = len(st.session_state.characteristics) + len(st.session_state.categories)
        total_proc = len(st.session_state.processed_characteristics) + len(st.session_state.processed_categories)
        
        if total_orig > 0:
            reduction = ((total_orig - total_proc) / total_orig) * 100
            st.metric("Element Reduction", f"{reduction:.1f}%")
    
    # Display processed elements
    st.subheader("Processed Elements")
    proc_tab1, proc_tab2 = st.tabs(["Processed Characteristics", "Processed Categories"])
    
    with proc_tab1:
        if st.session_state.processed_characteristics:
            st.dataframe(pd.DataFrame(st.session_state.processed_characteristics))
        else:
            st.write("No processed characteristics yet.")
    
    with proc_tab2:
        if st.session_state.processed_categories:
            st.dataframe(pd.DataFrame(st.session_state.processed_categories))
        else:
            st.write("No processed categories yet.")
    
    # Export options
    st.subheader("Export Options")
    export_format = st.radio(
        "Export Format:",
        ["JSON", "CSV"],
        horizontal=True
    )
    
    with st.expander("Export Format Help"):
        st.markdown("""
        ### EXPORT FORMATS:

        **JSON EXPORT:**
        Creates a single JSON file with a structured representation of the taxonomy, including:
        - All processed characteristics with their properties
        - All processed categories with their properties
        - Metadata about the processing (counts, original elements, etc.)
        Ideal for preserving the full structure and relationships.

        **CSV EXPORT:**
        Creates three separate CSV files:
        1. characteristics.csv: All processed characteristics
        2. categories.csv: All processed categories
        3. metadata.csv: Statistics and processing information
        Ideal for importing into spreadsheets or databases.
        """)
    
    if st.button("Export Processed Data"):
        if export_format == "JSON":
            export_json()
        else:
            export_csv()
    
    # Navigation
    if st.button("← Back to Analysis"):
        st.session_state.current_section = "Similarity Analysis"
        st.rerun()


def welcome_page():
    """Display the welcome page with app information"""
    st.title("Taxonomy Element Processor")
    st.subheader("Agent 2: The Preparator")
    
    st.markdown("""
    This tool helps you prepare taxonomy elements by identifying and processing synonyms.

    The workflow consists of four main steps:

    1️⃣ **Settings**: Configure the language model and embedding model
    2️⃣ **Data Input**: Enter or import your taxonomy elements
    3️⃣ **Similarity Analysis**: Analyze semantic similarity and identify potential synonyms using HDBSCAN clustering
    4️⃣ **Review & Export**: Review the results and export the final taxonomy

    The tool uses embedding-based similarity analysis with HDBSCAN clustering to identify potential synonyms,
    giving you visual feedback and statistical support for your decisions.

    To get started, make sure you have a language model running locally,
    then proceed to the Settings tab to configure the application.
    """)
    
    st.subheader("Quick Tips")
    
    st.markdown("""
    • Check the Settings tab first to ensure proper model connection
    • HDBSCAN clustering parameters control how elements are grouped:
      - Minimum Cluster Size: The minimum number of elements required to form a cluster
      - Minimum Samples: Controls density requirements (higher = more conservative clustering)
      - Cluster Selection Epsilon: Controls the distance threshold for considering points as part of the same cluster
    • The Interactive Heatmap lets you manually select element pairs to create custom clusters
    • Use the 3D visualization to see how elements cluster together semantically
    • Request LLM suggestions for specific clusters as needed
    • Consider processing characteristics and categories separately for best results
    """)
    
    if st.button("Get Started"):
        st.session_state.current_section = "Settings"
        st.rerun()


# Data Input Functions
def add_manual_input(char_text, cat_text):
    """Process manual input and add to data"""
    # Process characteristics
    if char_text:
        # Split into lines
        lines = char_text.split("\n")
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split by pipe delimiter
            parts = [part.strip() for part in line.split("|")]
            
            # Extract parts with defaults for missing values
            name = parts[0] if len(parts) > 0 else f"Unknown {i+1}"
            source = parts[1] if len(parts) > 1 else "Unknown"
            description = parts[2] if len(parts) > 2 else ""
            
            # Generate ID for the characteristic
            char_id = f"char_{len(st.session_state.characteristics) + 1}"
            
            # Add to data structure
            st.session_state.characteristics.append({
                "id": char_id,
                "name": name,
                "source": source,
                "description": description,
                "type": "characteristic"
            })
    
    # Process categories
    if cat_text:
        # Split into lines
        lines = cat_text.split("\n")
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split by pipe delimiter
            parts = [part.strip() for part in line.split("|")]
            
            # Extract parts with defaults for missing values
            name = parts[0] if len(parts) > 0 else f"Unknown {i+1}"
            source = parts[1] if len(parts) > 1 else "Unknown"
            parent = parts[2] if len(parts) > 2 else ""
            
            # Generate ID for the category
            cat_id = f"cat_{len(st.session_state.categories) + 1}"
            
            # Add to data structure
            st.session_state.categories.append({
                "id": cat_id,
                "name": name,
                "source": source,
                "parent": parent,
                "type": "category"
            })
    
    # Show success message
    char_count = len(char_text.split("\n")) if char_text else 0
    cat_count = len(cat_text.split("\n")) if cat_text else 0
    if char_count > 0 or cat_count > 0:
        st.success(f"Added {char_count} characteristics and {cat_count} categories")


def import_csv_data(uploaded_file):
    """Import data from uploaded CSV file"""
    try:
        # Reset file pointer to start
        uploaded_file.seek(0)
        
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns
        required_columns = ["name", "type", "source"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Show error if required columns are missing
        if missing_columns:
            st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
            return
        
        # Process each row
        char_count = 0
        cat_count = 0
        
        for _, row in df.iterrows():
            element_type = row.get("type", "").strip().lower()
            
            # Process characteristics
            if element_type == "characteristic":
                char_id = f"char_{len(st.session_state.characteristics) + 1}"
                
                # Add to data structure
                st.session_state.characteristics.append({
                    "id": char_id,
                    "name": row.get("name", "Unknown"),
                    "source": row.get("source", "Unknown"),
                    "description": row.get("description", ""),
                    "type": "characteristic"
                })
                
                char_count += 1
            
            # Process categories
            elif element_type == "category":
                cat_id = f"cat_{len(st.session_state.categories) + 1}"
                
                # Add to data structure
                st.session_state.categories.append({
                    "id": cat_id,
                    "name": row.get("name", "Unknown"),
                    "source": row.get("source", "Unknown"),
                    "parent": row.get("parent", ""),
                    "type": "category"
                })
                
                cat_count += 1
        
        # Show success message
        st.success(f"Successfully imported {char_count} characteristics and {cat_count} categories")
    
    except Exception as e:
        st.error(f"Error importing CSV: {str(e)}")


def import_json_data(content):
    """Import data from JSON content"""
    try:
        # Parse JSON
        data = json.loads(content)
        
        # Counters for imported elements
        char_count = 0
        cat_count = 0
        
        # Check the format of the JSON data
        if isinstance(data, list):
            # Array of elements format
            for element in data:
                element_type = element.get("type", "").lower()
                
                # Process characteristics
                if element_type == "characteristic":
                    char_id = f"char_{len(st.session_state.characteristics) + 1}"
                    
                    # Add to data structure
                    st.session_state.characteristics.append({
                        "id": char_id,
                        "name": element.get("name", "Unknown"),
                        "source": element.get("source", "Unknown"),
                        "description": element.get("description", ""),
                        "type": "characteristic"
                    })
                    
                    char_count += 1
                
                # Process categories
                elif element_type == "category":
                    cat_id = f"cat_{len(st.session_state.categories) + 1}"
                    
                    # Add to data structure
                    st.session_state.categories.append({
                        "id": cat_id,
                        "name": element.get("name", "Unknown"),
                        "source": element.get("source", "Unknown"),
                        "parent": element.get("parent", ""),
                        "type": "category"
                    })
                    
                    cat_count += 1
        
        elif "characteristics" in data and "categories" in data:
            # Structured format with separate arrays
            
            # Import characteristics
            for char in data.get("characteristics", []):
                char_id = f"char_{len(st.session_state.characteristics) + 1}"
                
                # Add to data structure
                st.session_state.characteristics.append({
                    "id": char_id,
                    "name": char.get("name", "Unknown"),
                    "source": char.get("source", "Unknown"),
                    "description": char.get("description", ""),
                    "type": "characteristic"
                })
                
                char_count += 1
            
            # Import categories
            for cat in data.get("categories", []):
                cat_id = f"cat_{len(st.session_state.categories) + 1}"
                
                # Add to data structure
                st.session_state.categories.append({
                    "id": cat_id,
                    "name": cat.get("name", "Unknown"),
                    "source": cat.get("source", "Unknown"),
                    "parent": cat.get("parent", ""),
                    "type": "category"
                })
                
                cat_count += 1
        
        else:
            # Unknown format
            st.error("Unknown JSON format. Expected array of elements or structured format.")
            return
        
        # Show success message
        st.success(f"Successfully imported {char_count} characteristics and {cat_count} categories")
    
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {str(e)}")
    except Exception as e:
        st.error(f"Error importing JSON: {str(e)}")


def clear_all_data():
    """Clear all imported data"""
    confirm_clear = st.button("Confirm Clear All Data", key="confirm_clear_btn")
    if confirm_clear:
        st.session_state.characteristics = []
        st.session_state.categories = []
        st.success("All data cleared")
        st.rerun()


# Settings Functions
def fetch_models():
    """Fetch available models from the API"""
    with st.spinner("Fetching available models..."):
        try:
            # Make API request
            response = requests.get(st.session_state.models_url)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Handle different JSON response formats
            # Standard OpenAI format
            if "data" in data and isinstance(data["data"], list):
                st.session_state.available_models = [model["id"] for model in data["data"]]
            # LM Studio sometimes uses a different format
            elif "object" in data and data.get("object") == "list":
                st.session_state.available_models = [model["id"] for model in data.get("data", [])]
            # Yet another format used by some LM Studio versions
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                st.session_state.available_models = [model.get("id", model.get("name", "unknown")) for model in data]
            # Direct model list (simpler format)
            elif isinstance(data, dict) and "models" in data and isinstance(data["models"], list):
                st.session_state.available_models = data["models"]
            # If no recognizable format, try to extract anything that might be a model ID
            else:
                # Last resort: try to find any field that could be model IDs
                potential_models = []
                
                def extract_model_names(obj, path=""):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k in ["id", "name", "model"]:
                                if isinstance(v, str):
                                    potential_models.append(v)
                            else:
                                extract_model_names(v, f"{path}.{k}" if path else k)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            extract_model_names(item, f"{path}[{i}]")
                
                extract_model_names(data)
                
                if potential_models:
                    st.session_state.available_models = potential_models
                else:
                    # If still no models found, show a more descriptive error
                    response_excerpt = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                    raise ValueError(f"Could not find models in response: {response_excerpt}")
            
            # Fallback for empty model list
            if not st.session_state.available_models:
                st.session_state.available_models = ["llama2", "mistral", "mixtral", "phi2"]
                st.warning("No models were found in the API response. Using default placeholder models.")
            
            # Set current model if none is set
            if not st.session_state.current_model and st.session_state.available_models:
                st.session_state.current_model = st.session_state.available_models[0]
            
            st.success(f"Found {len(st.session_state.available_models)} available models")
            st.rerun()
        
        except Exception as e:
            st.error(f"Error connecting to models API: {str(e)}")
            
            # Suggest possible fixes based on the error
            if "Connection refused" in str(e):
                st.warning("Make sure LM Studio is running and the server is started.")
            elif "404" in str(e):
                st.warning("The models endpoint URL may be incorrect. Try adjusting the API endpoint in Settings.")
            elif "timeout" in str(e):
                st.warning("The server took too long to respond. Check if LM Studio is running properly.")
            else:
                st.warning("Check your connection settings and make sure LM Studio is running correctly.")


def update_endpoint(new_endpoint):
    """Update the API endpoint"""
    # Validate that it's a proper URL
    if not new_endpoint.startswith("http"):
        st.error("Endpoint must be a valid URL starting with http:// or https://")
        return
    
    # Update the API endpoint
    st.session_state.api_url = new_endpoint
    
    # Parse the URL to construct the models URL correctly
    url_parts = new_endpoint.split("/")
    
    # Check if the endpoint URL already has "v1" in it
    if "v1" in url_parts:
        # Base URL is everything up to and including the server address (before any paths)
        server_part = "/".join(url_parts[:3])  # http://server:port
        st.session_state.models_url = f"{server_part}/v1/models"
    else:
        # Standard construction - remove the last path component
        base_url = "/".join(new_endpoint.split("/")[:-1])
        st.session_state.models_url = f"{base_url}/models"
    
    # Fetch models with new endpoint
    fetch_models()


def test_embedding_model():
    """Test if the embedding model can be loaded"""
    with st.spinner("Testing embedding model..."):
        try:
            # Try to load the embedding model
            success = st.session_state.embedding_calculator.load_model(
                lambda msg: st.session_state.update(status_message=msg)
            )
            
            if success:
                # Generate a test embedding
                texts = ["Test text for embedding"]
                inputs = st.session_state.embedding_calculator.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                
                with torch.no_grad():
                    model_output = st.session_state.embedding_calculator.model(**inputs)
                    embeddings = st.session_state.embedding_calculator.mean_pooling(model_output, inputs['attention_mask'])
                
                # Test passed if we got embeddings
                if embeddings.shape[1] > 0:  # Check if embedding has dimensions
                    st.success(f"""
                    Embedding model loaded successfully! ✓
                    Model: {st.session_state.embedding_calculator.model_name}
                    Embedding dimensions: {embeddings.shape[1]}
                    """)
                else:
                    st.error("Error: Model loaded but produced invalid embeddings")
            else:
                st.error("Error: Failed to load embedding model")
        
        except Exception as e:
            st.error(f"Test failed: ✗\nError: {str(e)}")


def test_llm_connection():
    """Test the connection to the LLM API"""
    model = st.session_state.current_model
    
    if not model:
        st.error("Please select a model before testing the connection.")
        return
    
    with st.spinner("Testing LLM connection..."):
        try:
            # Create a simple test message
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello briefly."}
                ],
                "temperature": 0.7,
                "max_tokens": 50
            }
            
            # Send API request with a timeout
            st.session_state.status_message = "Sending request to model..."
            response = requests.post(st.session_state.api_url, json=data, timeout=60)
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            
            # Handle different response formats
            content = ""
            if "choices" in result and len(result["choices"]) > 0:
                # Standard OpenAI format
                choice = result["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                elif "text" in choice:
                    content = choice["text"]
            elif "response" in result:
                # Some LM Studio versions use this format
                content = result["response"]
            else:
                # Try to find any text in the response
                result_str = str(result)
                content = f"Received response but format was unexpected: {result_str[:100]}..."
            
            # Show success
            st.success(f"""
            Connection successful! ✓
            Model: {model}
            Response: {content}
            """)
        
        except requests.exceptions.Timeout:
            # Handle timeout specifically
            st.error("Connection timed out. The model may be loading or too slow to respond.")
        
        except Exception as e:
            # Show detailed error information
            error_details = str(e)
            if hasattr(e, 'response') and e.response:
                try:
                    error_json = e.response.json()
                    if 'error' in error_json:
                        error_details = f"{error_details}\nAPI Error: {error_json['error']}"
                except:
                    error_details = f"{error_details}\nStatus code: {e.response.status_code}"
            
            st.error(f"Connection failed: ✗\nError: {error_details}")
            
            # Provide helpful suggestions based on error type
            if "Connection refused" in str(e):
                st.warning("Make sure the LM Studio server is running and the API endpoint is correct.")
            elif "404" in str(e):
                st.warning("The API endpoint URL may be incorrect. Check your settings.")
            elif "invalid model" in str(e).lower() or "not found" in str(e).lower():
                st.warning(f"The model '{model}' might not be loaded in LM Studio or has a different name.")
            else:
                st.warning("Check your connection settings and make sure LM Studio is properly configured.")


# Similarity Analysis Functions
def calculate_embeddings():
    """Calculate embeddings for the selected elements"""
    # Get selected process type
    process_type = st.session_state.process_type
    
    # Filter elements based on process type
    if process_type == "characteristics":
        if not st.session_state.characteristics:
            st.error("No characteristics available to process")
            return
        elements_to_process = st.session_state.characteristics
    elif process_type == "categories":
        if not st.session_state.categories:
            st.error("No categories available to process")
            return
        elements_to_process = st.session_state.categories
    else:  # "both"
        if not st.session_state.characteristics and not st.session_state.categories:
            st.error("No elements available to process")
            return
        elements_to_process = st.session_state.characteristics + st.session_state.categories
    
    # Calculate embeddings
    with st.spinner(f"Calculating embeddings for {len(elements_to_process)} elements..."):
        success = st.session_state.embedding_calculator.compute_embeddings(
            elements_to_process,
            lambda msg: st.session_state.update(status_message=msg)
        )
        
        if success:
            # Compute similarity matrix
            st.session_state.embedding_calculator.compute_similarity_matrix(
                lambda msg: st.session_state.update(status_message=msg)
            )
            
            # Find clusters using HDBSCAN
            st.session_state.embedding_calculator.find_clusters(
                min_cluster_size=st.session_state.min_cluster_size,
                min_samples=st.session_state.min_samples,
                cluster_selection_epsilon=st.session_state.cluster_selection_epsilon,
                status_callback=lambda msg: st.session_state.update(status_message=msg)
            )
            
            st.success("Embeddings calculated successfully!")
            st.rerun()
        else:
            st.error("Failed to compute embeddings")


def display_clusters():
    """Display clusters with partial processing support - fixed nested expanders"""
    # Display HDBSCAN clusters
    if not hasattr(st.session_state.embedding_calculator, 'clusters') or not st.session_state.embedding_calculator.clusters:
        st.info("No automatic clusters available. Try adjusting the clustering parameters.")
    else:
        clusters = st.session_state.embedding_calculator.clusters
        
        # Initialize processed_element_ids if not already in session state
        if 'processed_element_ids' not in st.session_state:
            st.session_state.processed_element_ids = set()
        
        # Categorize clusters based on their processing status
        fully_processed_clusters = []
        partially_processed_clusters = []
        unprocessed_clusters = []
        single_element_clusters = []
        
        total_elements = 0
        processed_elements = 0
        
        # Process all automatic clusters
        for cluster in clusters:
            # Get the elements in this cluster
            elements = []
            processed_in_cluster = 0
            
            for element_id in cluster:
                element = st.session_state.embedding_calculator.get_element_by_id(element_id)
                if element:
                    elements.append(element)
                    total_elements += 1
                    if element_id in st.session_state.processed_element_ids:
                        processed_in_cluster += 1
                        processed_elements += 1
            
            if len(elements) == 0:
                continue  # Skip empty clusters
            
            # Determine if cluster is single-element, fully processed, partially processed, or unprocessed
            if len(elements) == 1:
                single_element_clusters.append((cluster, elements, processed_in_cluster))
            elif processed_in_cluster == len(elements):
                fully_processed_clusters.append((cluster, elements, processed_in_cluster))
            elif processed_in_cluster > 0:
                partially_processed_clusters.append((cluster, elements, processed_in_cluster))
            else:
                unprocessed_clusters.append((cluster, elements, 0))
        
        # Process custom clusters
        for i, custom_cluster in enumerate(st.session_state.custom_clusters):
            # Get the elements in this custom cluster
            elements = []
            processed_in_cluster = 0
            
            for element_id in custom_cluster:
                element = st.session_state.embedding_calculator.get_element_by_id(element_id)
                if element:
                    elements.append(element)
                    # Don't double-count elements that were already counted in automatic clusters
                    if element_id in st.session_state.processed_element_ids:
                        processed_in_cluster += 1
            
            if len(elements) == 0:
                continue  # Skip empty clusters
            
            # Determine if cluster is fully processed, partially processed, or unprocessed
            if processed_in_cluster == len(elements):
                fully_processed_clusters.append((custom_cluster, elements, processed_in_cluster))
            elif processed_in_cluster > 0:
                partially_processed_clusters.append((custom_cluster, elements, processed_in_cluster))
            else:
                unprocessed_clusters.append((custom_cluster, elements, 0))
        
        # Summary statistics
        st.info(f"""
        ### Clustering Summary
        - Found {len(single_element_clusters)} elements that didn't cluster with others
        - Total clusters with multiple elements: {len(unprocessed_clusters) + len(partially_processed_clusters) + len(fully_processed_clusters)}
        - Unprocessed clusters: {len(unprocessed_clusters)}
        - Partially processed clusters: {len(partially_processed_clusters)}
        - Fully processed clusters: {len(fully_processed_clusters)}
        - Elements processed: {processed_elements} of {total_elements} ({int(processed_elements/max(1, total_elements)*100)}%)
        """)
        
        # Display unprocessed multi-element clusters first
        if unprocessed_clusters:
            st.subheader("Unprocessed Clusters")
            st.write("These clusters contain elements that need decisions:")
            
            for i, (cluster, elements, _) in enumerate(unprocessed_clusters):
                # Generate a stable cluster ID
                cluster_id = f"cluster_{'_'.join(sorted([e['id'] for e in elements]))}"
                
                # Create an expander for each unprocessed cluster
                expander_label = f"Cluster {i+1}: {', '.join([e['name'] for e in elements])} ({len(elements)} elements)"
                
                with st.expander(expander_label, expanded=True):
                    # Show element details
                    cluster_df = pd.DataFrame([
                        {
                            "Name": element["name"],
                            "Type": element["type"],
                            "Source": element["source"],
                            "Description/Parent": element.get("description", element.get("parent", "")),
                            "ID": element["id"]
                        }
                        for element in elements
                    ])
                    
                    st.dataframe(cluster_df)
                    
                    # Buttons for selecting and getting LLM suggestions
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button(f"Select Cluster {i+1}", key=f"select_cluster_{i}"):
                            select_cluster([e['id'] for e in elements], cluster_id)
                    
                    with col2:
                        if st.button(f"Get LLM Suggestions {i+1}", key=f"llm_suggest_{i}"):
                            get_llm_suggestions(elements, cluster_id)
        else:
            st.success("✅ All clusters have been processed!")
        
        # Display partially processed clusters
        if partially_processed_clusters:
            st.subheader("Partially Processed Clusters")
            st.write("These clusters have some elements that still need decisions:")
            
            for i, (cluster, elements, processed_count) in enumerate(partially_processed_clusters):
                # Generate a stable cluster ID
                cluster_id = f"partial_{'_'.join(sorted([e['id'] for e in elements]))}"
                
                # Create an expander for each partially processed cluster
                expander_label = f"Cluster {i+1}: {', '.join([e['name'] for e in elements])} ({processed_count}/{len(elements)} processed)"
                
                with st.expander(expander_label, expanded=True):
                    # Prepare element data with status indicators
                    element_data = []
                    for element in elements:
                        is_processed = element["id"] in st.session_state.processed_element_ids
                        status = "✅ Processed" if is_processed else "⏳ Pending"
                        
                        element_data.append({
                            "Name": element["name"],
                            "Type": element["type"],
                            "Source": element["source"],
                            "Description/Parent": element.get("description", element.get("parent", "")),
                            "Status": status,
                            "ID": element["id"]
                        })
                    
                    # Show element details with status
                    st.dataframe(pd.DataFrame(element_data))
                    
                    # Buttons for selecting and getting LLM suggestions
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if st.button(f"Process Remaining Elements {i+1}", key=f"process_partial_{i}"):
                            # Filter to get only unprocessed elements
                            unprocessed_elements = [e for e in elements if e["id"] not in st.session_state.processed_element_ids]
                            select_cluster([e['id'] for e in unprocessed_elements], cluster_id)
                    
                    with col2:
                        if st.button(f"Get LLM Suggestions {i+1}", key=f"llm_suggest_partial_{i}"):
                            # Only get suggestions for unprocessed elements
                            unprocessed_elements = [e for e in elements if e["id"] not in st.session_state.processed_element_ids]
                            get_llm_suggestions(unprocessed_elements, cluster_id)
        
        # Display fully processed clusters WITHOUT nesting expanders
        if fully_processed_clusters:
            st.subheader("Fully Processed Clusters")
            show_processed = st.checkbox("Show processed clusters", value=False)
            
            if show_processed:
                st.write("These clusters have been completely processed:")
                
                for i, (cluster, elements, _) in enumerate(fully_processed_clusters):
                    # Find the relevant decision(s) for this cluster
                    cluster_decisions = []
                    decision_indices = []
                    
                    for j, decision in enumerate(st.session_state.decisions):
                        # Check if any elements from this cluster were involved in this decision
                        if any(e["id"] in decision["cluster"] for e in elements):
                            decision_indices.append(j)
                            cluster_decisions.append(decision)
                    
                    # Create a bordered section for each processed cluster
                    st.markdown("---")
                    st.markdown(f"### ✅ Cluster {i+1}: {', '.join([e['name'] for e in elements])} ({len(elements)} elements)")
                    
                    # Show the decisions made for this cluster
                    st.markdown(f"**Decisions applied to this cluster:**")
                    
                    for j, decision in enumerate(cluster_decisions):
                        decision_elements = decision["elements"]
                        st.markdown(f"**Decision {j+1}:**")
                        
                        if decision["decision_type"] == "merge":
                            st.markdown(f"- Merged elements as: **{decision.get('merge_term', 'Unknown')}**")
                            st.markdown(f"- Source: {decision.get('merge_source', 'Unknown')}")
                            st.markdown("- Elements merged:")
                        else:
                            st.markdown("- Kept elements separate:")
                        
                        # List the elements involved in this decision
                        element_df = pd.DataFrame([
                            {
                                "Name": element["name"],
                                "Type": element["type"],
                                "Source": element["source"]
                            }
                            for element in decision_elements
                        ])
                        
                        st.dataframe(element_df)
                        
                        # Add a button to reverse this decision
                        if st.button(f"Reverse Decision {j+1}", key=f"reverse_decision_{i}_{j}"):
                            reverse_decision(decision_indices[j])
        
        # Display single-element clusters at the bottom
        if single_element_clusters:
            st.subheader("Single-Element Clusters")
            show_singles = st.checkbox("Show single element clusters", value=False)
            
            if show_singles:
                st.write("These elements didn't cluster with others:")
                
                # Create a dataframe of all single-element clusters
                single_elements = []
                for cluster, elements, _ in single_element_clusters:
                    element = elements[0]  # Single element in the cluster
                    is_processed = element["id"] in st.session_state.processed_element_ids
                    status = "✅ Processed" if is_processed else "⏳ Not Processed"
                    
                    single_elements.append({
                        "Name": element["name"],
                        "Type": element["type"],
                        "Source": element["source"],
                        "Description/Parent": element.get("description", element.get("parent", "")),
                        "Status": status
                    })
                
                # Show all single elements in one table
                st.dataframe(pd.DataFrame(single_elements))


def reverse_decision(decision_index):
    """Reverse a previous decision"""
    try:
        # Get the decision to reverse
        decision = st.session_state.decisions[decision_index]
        
        # Remove the processed status from all elements in this decision
        for element_id in decision["cluster"]:
            if element_id in st.session_state.processed_element_ids:
                st.session_state.processed_element_ids.remove(element_id)
        
        # Remove the decision from the list
        st.session_state.decisions.pop(decision_index)
        
        st.success("Decision successfully reversed. The elements are now available for processing again.")
        st.rerun()
    except (IndexError, KeyError) as e:
        st.error(f"Error reversing decision: {str(e)}")


def display_similarity_matrix():
    """Display the similarity matrix visualization"""
    if not hasattr(st.session_state.embedding_calculator, 'similarity_matrix') or st.session_state.embedding_calculator.similarity_matrix is None:
        st.info("No similarity matrix available. Calculate embeddings first.")
        return
    
    matrix = st.session_state.embedding_calculator.similarity_matrix
    
    st.write("Similarity Matrix Heatmap:")
    
    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get element names
    element_ids = matrix.index.tolist()
    element_names = [st.session_state.embedding_calculator.get_element_by_id(id)['name'] for id in element_ids]
    
    # If there are too many elements, show fewer ticks
    if len(element_names) > 20:
        tick_indices = np.linspace(0, len(element_names) - 1, 20, dtype=int)
        tick_labels = [element_names[i] for i in tick_indices]
    else:
        tick_indices = np.arange(len(element_names))
        tick_labels = element_names
    
    # Create heatmap
    im = ax.imshow(matrix.values, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Similarity')
    
    # Set title
    ax.set_title('Similarity Matrix')
    plt.tight_layout()
    
    # Display the figure
    st.pyplot(fig)


def display_interactive_similarity_matrix():
    """Display an interactive similarity matrix visualization with batch selection and source information"""
    if not hasattr(st.session_state.embedding_calculator, 'similarity_matrix') or st.session_state.embedding_calculator.similarity_matrix is None:
        st.info("No similarity matrix available. Calculate embeddings first.")
        return
    
    matrix = st.session_state.embedding_calculator.similarity_matrix
    
    st.write("Interactive Similarity Matrix Heatmap:")
    st.write("Adjust the semantic similarity threshold to identify and select highly similar pairs.")
    
    # Add a threshold slider
    threshold = st.slider(
        "Highlight Similarity Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.heatmap_threshold,
        step=0.05,
        format="%.2f",
        key="heatmap_threshold_slider"
    )
    
    # Update threshold in session state
    st.session_state.heatmap_threshold = threshold
    
    # Get element IDs and names
    element_ids = matrix.index.tolist()
    elements = [st.session_state.embedding_calculator.get_element_by_id(id) for id in element_ids]
    
    # Create element names with source information
    element_names_with_source = [f"{element['name']} [{element['source']}]" for element in elements]
    
    # Create the heatmap with Plotly
    fig = go.Figure()
    
    # Add main heatmap
    heatmap = go.Heatmap(
        z=matrix.values,
        x=element_names_with_source,  # Now includes source info
        y=element_names_with_source,  # Now includes source info
        colorscale='Viridis',
        zmin=0, 
        zmax=1,
        hovertemplate='<b>Similarity:</b> %{z:.4f}<br><b>Elements:</b> %{x} & %{y}<extra></extra>'
    )
    
    fig.add_trace(heatmap)
    
    # Create highlighted cells data structure for cells above threshold
    highlighted_cells = []
    for i in range(len(element_ids)):
        for j in range(len(element_ids)):
            if i != j and matrix.values[i][j] >= threshold:
                # Add to highlighted cells
                highlighted_cells.append({
                    "id1": element_ids[i], 
                    "id2": element_ids[j],
                    "name1": elements[i]['name'],
                    "name2": elements[j]['name'],
                    "source1": elements[i]['source'],
                    "source2": elements[j]['source'],
                    "similarity": float(matrix.values[i][j]),
                    "cell_key": f"{element_ids[i]}|{element_ids[j]}"  # Use pipe as separator
                })
                
                # Add a symbol to highlight high similarity
                cell_key = f"{element_ids[i]}|{element_ids[j]}"  # Use pipe as separator
                is_selected = cell_key in st.session_state.selected_matrix_cells
                
                symbol_color = 'red' if is_selected else 'white'
                symbol = '✓' if is_selected else '●'
                
                fig.add_annotation(
                    x=element_names_with_source[j],
                    y=element_names_with_source[i],
                    text=symbol,
                    showarrow=False,
                    font=dict(color=symbol_color, size=12)
                )
    
    # Update layout
    fig.update_layout(
        title=f"Similarity Matrix (threshold: {threshold})",
        xaxis_title="Elements [Source]",
        yaxis_title="Elements [Source]",
        height=700,
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    # Make y-axis labels properly visible
    fig.update_yaxes(automargin=True)
    
    # Display the interactive plotly chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Alternative method: Show a table of highlighted cells that can be selected
    if highlighted_cells:
        st.subheader("High Similarity Pairs")
        st.write(f"Select the pairs you want to include in a custom cluster, then click 'Confirm Selections'.")
        
        # Sort by similarity (highest first)
        highlighted_cells.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Only show each pair once
        unique_pairs = {}
        for cell in highlighted_cells:
            pair_key = frozenset([cell['id1'], cell['id2']])
            if pair_key not in unique_pairs:
                unique_pairs[pair_key] = cell
        
        # Initialize batch selection form
        with st.form(key="batch_selection_form"):
            # Create the pairs dataframe
            pairs_data = []
            
            # Initialize keys to store selection info for post-processing
            element_id_pairs = []  # Store pairs of element IDs
            
            for cell in unique_pairs.values():
                # Check if this pair is already selected
                is_selected = cell['cell_key'] in st.session_state.selected_matrix_cells
                
                # Add to the data for display with source information
                pairs_data.append({
                    "Select": is_selected,  # Pre-check already selected pairs
                    "Element 1": f"{cell['name1']} [{cell['source1']}]",
                    "Element 2": f"{cell['name2']} [{cell['source2']}]",
                    "Similarity": f"{cell['similarity']:.4f}",
                    # Store the element IDs as a hidden field
                    "element_ids": (cell['id1'], cell['id2']),
                })
                
                # Add to element_id_pairs for lookup during processing
                element_id_pairs.append((cell['id1'], cell['id2']))
            
            # Create the dataframe
            pairs_df = pd.DataFrame(pairs_data)
            
            # Display the form elements with column order that shows both elements separately
            edited_df = st.data_editor(
                pairs_df,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select", 
                        help="Select pairs to include in custom cluster"
                    ),
                    "element_ids": st.column_config.Column(
                        "Element IDs",
                        disabled=True,
                        required=True
                    )
                },
                hide_index=True,
                disabled=False,
                column_order=["Select", "Element 1", "Element 2", "Similarity"],
                use_container_width=True
            )
            
            # Add a submit button
            submitted = st.form_submit_button("Confirm Selections")
            
            if submitted:
                # Process selections as before...
                # [rest of the function remains the same]
                # Clear existing selections if needed
                new_selected_cells = set()
                new_selected_elements = set()
                
                # Process all selections at once
                for i, row in edited_df.iterrows():
                    if row["Select"]:
                        # Get the element IDs for this pair
                        id1, id2 = element_id_pairs[i]
                        
                        # Add cell keys to new selections
                        cell_key = f"{id1}|{id2}"
                        reverse_key = f"{id2}|{id1}"
                        new_selected_cells.add(cell_key)
                        new_selected_cells.add(reverse_key)
                        
                        # Add element IDs to new selections
                        new_selected_elements.add(id1)
                        new_selected_elements.add(id2)
                
                # Update session state with all selections at once
                st.session_state.selected_matrix_cells = new_selected_cells
                st.session_state.selected_element_ids = new_selected_elements
                
                st.success(f"Updated selections: {len(new_selected_elements)} elements in {len(new_selected_cells)//2} pairs")
    
    # Display selected element pairs
    if st.session_state.selected_matrix_cells:
        st.subheader("Selected Element Pairs")
        
        # Calculate how many unique elements are involved
        unique_elements = st.session_state.selected_element_ids
        
        # Check if there are any elements
        if not unique_elements:
            st.write("No elements selected.")
        else:
            elements = [st.session_state.embedding_calculator.get_element_by_id(id) for id in unique_elements]
            elements = [e for e in elements if e is not None]  # Filter out any None elements
            
            st.write(f"Selected {len(elements)} elements for custom cluster:")
            
            # Display the selected elements
            if elements:
                elements_df = pd.DataFrame([
                    {
                        "Name": elem['name'],
                        "Type": elem['type'],
                        "Source": elem['source']
                    }
                    for elem in elements
                ])
                
                st.dataframe(elements_df, use_container_width=True)
                
                # Display selected pairs
                st.write("Selected pairs:")
                
                # Build a list of selected pairs to display
                selected_pairs_data = []
                pairs_processed = set()
                
                # Get the matrix for similarity values
                for cell_key in st.session_state.selected_matrix_cells:
                    # Skip if invalid format
                    if "|" not in cell_key:
                        continue
                    
                    # Split with pipe separator
                    parts = cell_key.split('|')
                    if len(parts) != 2:
                        continue  # Skip invalid keys
                        
                    id1, id2 = parts
                    
                    # Skip if we've already processed this pair (to avoid duplicates)
                    pair_ids = frozenset([id1, id2])
                    if pair_ids in pairs_processed:
                        continue
                    
                    pairs_processed.add(pair_ids)
                    
                    # Get elements
                    elem1 = st.session_state.embedding_calculator.get_element_by_id(id1)
                    elem2 = st.session_state.embedding_calculator.get_element_by_id(id2)
                    
                    if elem1 is None or elem2 is None:
                        continue
                    
                    # Get similarity value from matrix
                    similarity = matrix.loc[id1, id2] if id1 in matrix.index and id2 in matrix.columns else 0
                    
                    # Add to pair data
                    selected_pairs_data.append({
                        "Element 1": elem1['name'],
                        "Element 2": elem2['name'],
                        "Similarity": f"{similarity:.4f}", 
                        "Above Threshold": similarity >= threshold
                    })
                
                # Sort by similarity
                selected_pairs_data.sort(key=lambda x: float(x["Similarity"]), reverse=True)
                
                # Display pairs
                st.dataframe(pd.DataFrame(selected_pairs_data), use_container_width=True)
                
                # Create two columns for the buttons
                col1, col2, col3 = st.columns(3)
                
                # Button to create a custom cluster
                with col1:
                    if st.button("Create Custom Cluster"):
                        create_custom_cluster(st.session_state.selected_element_ids)
                
                # Button to get LLM suggestions for selected elements
                with col2:
                    if st.button("Get LLM Suggestions"):
                        # Generate a stable cluster ID for the temporary cluster
                        temp_cluster_id = f"temp_custom_{'_'.join(sorted(st.session_state.selected_element_ids))}"
                        # Call get_llm_suggestions with the selected elements
                        get_llm_suggestions(elements, temp_cluster_id)
                
                # Button to clear selection
                with col3:
                    if st.button("Clear All Selections"):
                        st.session_state.selected_matrix_cells = set()
                        st.session_state.selected_element_ids = set()
                        st.session_state.llm_suggestions = ""  # Clear any LLM suggestions
                        st.rerun()


def plotly_chart_with_clicks(fig):
    """Render a Plotly chart and capture click events"""
    # Create a container for the chart
    chart_container = st.empty()
    
    # Display the chart
    chart_container.plotly_chart(fig, use_container_width=True)
    
    # Create a custom key for the chart's container
    chart_key = f"plotly_chart_{id(fig)}"
    
    # Capture click events using Streamlit's experimental get_user_input feature (if available)
    # This is a hack and might not work in all Streamlit versions
    try:
        if "streamlit" in st.__dict__ and hasattr(st, "get_user_inputs"):
            user_inputs = st.get_user_inputs()
            if chart_key in user_inputs:
                click_data = user_inputs[chart_key]
                return click_data
    except:
        pass
    
    # Fallback to manual click handling
    # This approach uses JavaScript to handle clicks but requires custom HTML/JS
    try:
        fig_json = fig.to_json()
        
        # JavaScript event handling
        st.markdown("""
        <script>
            const graphDiv = document.getElementById('plotly-chart');
            graphDiv.on('plotly_click', function(data) {
                const points = data.points;
                Streamlit.setComponentValue({
                    points: points.map(pt => ({
                        x: pt.x,
                        y: pt.y,
                        z: pt.z
                    }))
                });
            });
        </script>
        """, unsafe_allow_html=True)
        
        return st.session_state.get('plotly_click_data')
    except:
        return None


def create_custom_cluster(element_ids):
    """Create a custom cluster from selected element IDs"""
    if not element_ids or len(element_ids) < 2:
        st.error("Please select at least two elements to create a cluster.")
        return
    
    # Create a list from the set of element IDs
    cluster = list(element_ids)
    
    # Add to custom clusters
    if 'custom_clusters' not in st.session_state:
        st.session_state.custom_clusters = []
    
    st.session_state.custom_clusters.append(cluster)
    
    # Generate a cluster ID for the new custom cluster
    cluster_id = f"custom_{'_'.join(sorted(element_ids))}"
    
    # Select the newly created cluster
    st.session_state.selected_cluster = cluster
    st.session_state.current_cluster_id = cluster_id
    
    # Clear the interactive matrix selection
    st.session_state.selected_matrix_cells = set()
    st.session_state.selected_element_ids = set()
    
    # Show success message
    st.success(f"Created custom cluster with {len(cluster)} elements. Scroll down to the decision section to process it.")
    
    # Force a rerun to show the cluster in the decision section
    st.rerun()

def display_3d_visualization():
    """Display 3D visualization of embeddings with ellipsoid cluster boundaries"""
    if not hasattr(st.session_state.embedding_calculator, 'embeddings') or not st.session_state.embedding_calculator.embeddings:
        st.info("No embeddings available. Calculate embeddings first.")
        return
    
    # Get embeddings
    embeddings = st.session_state.embedding_calculator.embeddings
    element_ids = list(embeddings.keys())
    
    # Stack embeddings into a matrix
    embedding_matrix = np.stack([embeddings[id] for id in element_ids])
    
    # Reduce dimensions to 3D using t-SNE
    with st.spinner("Reducing dimensions to 3D..."):
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, max(5, len(element_ids)-1)))
        embeddings_3d = tsne.fit_transform(embedding_matrix)
    
    # Create a figure for the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mapping from element_id to index in the element_ids list
    element_id_to_index = {element_id: i for i, element_id in enumerate(element_ids)}
    
    # Create a mapping from element_id to cluster
    element_to_cluster = {}
    processed_elements = set()
    
    # Initialize multi-element clusters list
    multi_element_clusters = []
    
    # First, map elements to their HDBSCAN clusters
    if hasattr(st.session_state.embedding_calculator, 'clusters'):
        # Get the HDBSCAN clusters
        hdbscan_clusters = st.session_state.embedding_calculator.clusters
        
        # Extract multi-element clusters
        multi_element_clusters = [c for c in hdbscan_clusters if len(c) > 1]
        
        # Map elements to their cluster indices
        for cluster_idx, cluster in enumerate(multi_element_clusters):
            for element_id in cluster:
                element_to_cluster[element_id] = cluster_idx
    
    # Also mark processed elements
    if hasattr(st.session_state, 'processed_element_ids'):
        processed_elements = st.session_state.processed_element_ids
    
    # Add custom clusters to the mapping - FIXED HERE
    custom_cluster_offset = len(multi_element_clusters)
    for custom_idx, cluster in enumerate(st.session_state.custom_clusters):
        if len(cluster) > 1:  # Only consider multi-element custom clusters
            multi_element_clusters.append(cluster)
            for element_id in cluster:
                # Custom clusters take precedence over automatic clusters
                element_to_cluster[element_id] = custom_cluster_offset + custom_idx
    
    # Get all unique cluster indices
    cluster_indices = set(element_to_cluster.values())
    
    # Initialize cluster_points with proper size - FIXED HERE
    max_cluster_idx = max(cluster_indices) if cluster_indices else -1
    cluster_points = [[] for _ in range(max_cluster_idx + 1)]
    
    # Generate distinct colors for clusters - use a colorful palette
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(cluster_indices))))
    
    # Control options for visualization
    st.write("Visualization Controls:")
    col1, col2, col3 = st.columns(3)
    
    # Controls for point size and opacity
    with col1:
        point_size = st.slider("Point Size", 20, 150, 50, key="point_size_slider")
        hull_opacity = st.slider("Cluster Boundary Opacity", 0.0, 0.5, 0.15, key="hull_opacity_slider")
    
    # Option for marker style and noise points
    with col2:
        show_noise = st.checkbox("Show Noise Points", True, key="show_noise_checkbox")
        noise_color = st.selectbox("Noise Point Color", 
                                ["Gray", "Black", "White", "Yellow", "Purple", "Orange", "Brown"],
                                index=0, key="noise_color_selectbox")
        
        # Map color name to RGB
        noise_color_map = {
            "Gray": (0.7, 0.7, 0.7),
            "Black": (0.1, 0.1, 0.1),
            "White": (0.95, 0.95, 0.95),
            "Yellow": (0.9, 0.9, 0.1),
            "Purple": (0.7, 0.1, 0.7),
            "Orange": (0.9, 0.6, 0.1),
            "Brown": (0.6, 0.3, 0.1)
        }
        noise_rgb = noise_color_map[noise_color]
    
    # Option to show labels and adjust view
    with col3:
        show_labels = st.checkbox("Show Labels", False, key="show_labels_checkbox")
        show_legend = st.checkbox("Show Legend", True, key="show_legend_checkbox")
    
    # Plot unclustered (noise) points first
    noise_points = []
    for i, element_id in enumerate(element_ids):
        if element_id not in element_to_cluster:
            noise_points.append(i)
    
    if show_noise and noise_points:
        noise_x = [embeddings_3d[i, 0] for i in noise_points]
        noise_y = [embeddings_3d[i, 1] for i in noise_points]
        noise_z = [embeddings_3d[i, 2] for i in noise_points]
        
        ax.scatter(
            noise_x, noise_y, noise_z,
            c=[noise_rgb],
            s=point_size * 0.8,  # Slightly smaller
            alpha=0.7,
            marker='o',
            label="Unclustered Elements"
        )
    
    # Plot the clustered points and collect cluster points for ellipsoids
    for cluster_idx in sorted(cluster_indices):
        # Get all elements in this cluster
        cluster_element_ids = [eid for eid, cid in element_to_cluster.items() if cid == cluster_idx]
        
        # Get the indices in embeddings_3d
        point_indices = [element_id_to_index[eid] for eid in cluster_element_ids if eid in element_id_to_index]
        
        if not point_indices:
            continue
        
        # Get the 3D coordinates
        x = embeddings_3d[point_indices, 0]
        y = embeddings_3d[point_indices, 1]
        z = embeddings_3d[point_indices, 2]
        
        # Store points for ellipsoid fitting
        for idx in point_indices:
            cluster_points[cluster_idx].append(embeddings_3d[idx])
        
        # Determine cluster color
        color = cluster_colors[cluster_idx % len(cluster_colors)]
        
        # Get a representative name for the cluster
        if cluster_idx < len(multi_element_clusters):
            sample_element_id = multi_element_clusters[cluster_idx][0]
            sample_element = st.session_state.embedding_calculator.get_element_by_id(sample_element_id)
            if sample_element:
                cluster_name = f"Cluster {cluster_idx+1}: {sample_element['name'][:15]}..."
            else:
                cluster_name = f"Cluster {cluster_idx+1}"
        else:
            cluster_name = f"Custom Cluster {cluster_idx - custom_cluster_offset + 1}"
        
        # Plot the points
        ax.scatter(
            x, y, z,
            c=[color],
            s=point_size,
            alpha=0.8,
            marker='o',
            label=cluster_name
        )
        
        # Add labels if enabled
        if show_labels:
            for i, idx in enumerate(point_indices):
                element_id = element_ids[idx]
                element = st.session_state.embedding_calculator.get_element_by_id(element_id)
                if element:
                    ax.text(
                        x[i], y[i], z[i],
                        element['name'],
                        fontsize=8,
                        alpha=0.7
                    )
    
    # Function to create ellipsoid data
    def get_ellipsoid_data(points, scale_factor=2.0):
        if len(points) < 3:  # Need at least 3 points for a meaningful ellipsoid
            return None, None, None
        
        points = np.array(points)
        
        # Calculate center (mean) of points
        center = points.mean(axis=0)
        
        # Center the points
        centered_points = points - center
        
        # Calculate covariance matrix
        cov = np.cov(centered_points, rowvar=False)
        
        # Ensure covariance matrix is properly conditioned
        cov = cov + np.eye(3) * 1e-5
        
        # Calculate eigenvectors and eigenvalues
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Ensure positive eigenvalues (for numerical stability)
            eigvals = np.maximum(eigvals, 1e-5)
            
            # Scale the eigenvalues to make the ellipsoid more visible
            eigvals = eigvals * scale_factor
            
            # Generate points on a unit sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            
            # Stack the coordinates
            sphere_points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
            
            # Transform the sphere to ellipsoid
            transform = eigvecs.dot(np.diag(np.sqrt(eigvals)))
            ellipsoid_points = sphere_points.dot(transform.T) + center
            
            # Reshape for plotting
            x = ellipsoid_points[:, 0].reshape(x.shape)
            y = ellipsoid_points[:, 1].reshape(y.shape)
            z = ellipsoid_points[:, 2].reshape(z.shape)
            
            return x, y, z
        except np.linalg.LinAlgError:
            # If there's a numerical error, return None
            return None, None, None
    
    # Plot ellipsoids for each cluster if there are enough points
    for cluster_idx in sorted(cluster_indices):
        points = cluster_points[cluster_idx]
        
        if len(points) < 3:
            continue
        
        # Calculate ellipsoid
        x, y, z = get_ellipsoid_data(points)
        
        if x is not None and y is not None and z is not None:
            # Determine cluster color (matching the points)
            color = cluster_colors[cluster_idx % len(cluster_colors)]
            
            # Plot the ellipsoid surface
            ax.plot_surface(
                x, y, z,
                color=color,
                alpha=hull_opacity,
                shade=True,
                rstride=1, cstride=1
            )
    
    # Add legend only if requested
    if show_legend:
        # Create a legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # Limit legend entries if there are too many
        if len(by_label) > 10:
            # Extract just a few clusters and add "..." entry
            first_labels = list(by_label.keys())[:9]
            shortened_by_label = {label: by_label[label] for label in first_labels}
            shortened_by_label["+ more clusters..."] = plt.Line2D([0], [0], marker='o', color='w', 
                                                                 markerfacecolor=(0.5, 0.5, 0.5), markersize=10)
            ax.legend(shortened_by_label.values(), shortened_by_label.keys(), 
                     loc='upper right', fontsize=8, framealpha=0.7)
        else:
            ax.legend(by_label.values(), by_label.keys(), 
                     loc='upper right', fontsize=8, framealpha=0.7)
    
    # Set title and labels
    ax.set_title('3D Visualization with Cluster Boundaries')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    # Improve the appearance with a grid
    ax.xaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.5)
    ax.yaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.5)
    ax.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 0.5)
    
    # Adjust the viewing angle for better visibility
    ax.view_init(elev=30, azim=45)
    
    # Display the plot
    st.pyplot(fig)
    
    # Add explanation
    st.caption("""
    **Visualization Explanation:**
    - Each color represents a different cluster
    - Semi-transparent ellipsoids show the approximate boundaries of each cluster
    - Unclustered elements (noise points) are shown in gray 
    - Adjust the controls to customize the visualization
    """)

def display_cluster_decision():
    """Display the decision UI for the selected cluster with partial merging support"""
    st.subheader("Cluster Decision")
    st.write("Decide what to do with these elements:")
    
    # Get element data for the cluster
    elements = [st.session_state.embedding_calculator.get_element_by_id(element_id) for element_id in st.session_state.selected_cluster]
    elements = [e for e in elements if e is not None]  # Remove any None elements
    
    # Show selected elements with checkboxes for partial selection
    st.write("Selected elements:")
    
    # Create dataframe with element details and selection checkboxes
    element_data = []
    for i, element in enumerate(elements):
        element_id = element["id"]
        
        # Check if this element has already been processed
        is_processed = element_id in st.session_state.processed_element_ids
        
        # Only allow selection if not already processed
        is_selected = False
        if not is_processed:
            is_selected = st.session_state.partial_selection.get(element_id, True)
        
        status = "✅ Already Processed" if is_processed else ""
        
        element_data.append({
            "Select": is_selected,  # Can't select already processed elements
            "Name": element["name"],
            "Type": element["type"],
            "Source": element["source"],
            "Status": status,
            "ID": element_id  # Hidden column to store element ID
        })
    
    # Create a dataframe for display
    elements_df = pd.DataFrame(element_data)
    
    # Display the dataframe with selection checkboxes
    # Don't use lambda in disabled parameter - not supported in older Streamlit versions
    edited_df = st.data_editor(
        elements_df,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Include",
                help="Select elements to include in the merge or action"
            ),
            "ID": st.column_config.Column(
                "ID",
                disabled=True,
                required=True
            ),
            "Status": st.column_config.Column(
                "Status",
                help="Indicates if an element has already been processed"
            )
        },
        hide_index=True,
        column_order=["Select", "Name", "Type", "Source", "Status"],
        key="element_selection_editor",
        disabled=False
    )
    
    # Update partial selection based on edited dataframe
    if edited_df is not None:
        for i, row in edited_df.iterrows():
            element_id = row["ID"]
            is_selected = row["Select"]
            
            # Update selection state only for unprocessed elements
            if element_id not in st.session_state.processed_element_ids:
                st.session_state.partial_selection[element_id] = is_selected
    
    # Count how many elements are selected and not already processed
    selected_elements = []
    for i, row in edited_df.iterrows():
        element_id = row["ID"]
        is_selected = row["Select"]
        if is_selected and element_id not in st.session_state.processed_element_ids:
            selected_elements.append(elements[i])
    
    # Show LLM suggestions if available
    if st.session_state.llm_suggestions:
        with st.expander("LLM Suggestions", expanded=True):
            st.write(st.session_state.llm_suggestions)
    
    # Decision type radio buttons
    decision_type = st.radio(
        "Decision:",
        ["Merge selected elements", "Keep elements separate"],
        index=0 if st.session_state.decision_type == "merge" else 1,
        key="decision_type_radio",
        horizontal=True
    )
    
    # Update decision type in session state
    st.session_state.decision_type = "merge" if decision_type == "Merge selected elements" else "keep"
    
    # Merge options (only show if merge is selected)
    if st.session_state.decision_type == "merge":
        if len(selected_elements) < 2:
            st.warning("You need to select at least 2 elements to merge.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                merge_term = st.text_input(
                    "Merge as:",
                    value=st.session_state.merge_term if st.session_state.merge_term else (selected_elements[0]["name"] if selected_elements else "")
                )
                st.session_state.merge_term = merge_term
            
            with col2:
                # Get default source by concatenating all unique sources of selected elements
                if selected_elements:
                    sources = set(element['source'] for element in selected_elements)
                    if len(sources) == 1:
                        default_source = next(iter(sources))
                    else:
                        # Join all sources with commas
                        default_source = ", ".join(sorted(sources))
                else:
                    default_source = ""
                
                merge_source = st.text_input(
                    "Source:",
                    value=st.session_state.merge_source if hasattr(st.session_state, 'merge_source') else default_source
                )
                st.session_state.merge_source = merge_source
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Disable the button if we're trying to merge but fewer than 2 elements are selected
        apply_disabled = (st.session_state.decision_type == "merge" and len(selected_elements) < 2)
        
        if st.button("Apply Decision", disabled=apply_disabled, key="apply_decision_btn"):
            # Get the IDs of the selected elements
            selected_element_ids = [elem["id"] for elem in selected_elements]
            
            if selected_element_ids:
                # Apply the decision to only the selected elements
                apply_cluster_decision(selected_element_ids)
            else:
                st.error("No elements selected for processing")
    
    with col2:
        if st.button("Skip"):
            skip_cluster()
    
    with col3:
        if st.button("Select All Unprocessed"):
            # Manually handle selection of all unprocessed elements
            for element_id in st.session_state.selected_cluster:
                if element_id not in st.session_state.processed_element_ids:
                    st.session_state.partial_selection[element_id] = True
            st.rerun()


def skip_cluster():
    """Skip the current cluster without making a decision"""
    st.session_state.selected_cluster = None
    st.session_state.llm_suggestions = ""
    st.session_state.partial_selection = {}
    st.rerun()


def select_cluster(cluster, cluster_id=None):
    """Select a cluster to work with"""
    st.session_state.selected_cluster = cluster
    
    # Store the cluster ID if provided
    if cluster_id:
        st.session_state.current_cluster_id = cluster_id
    else:
        # Generate a stable cluster ID for tracking
        st.session_state.current_cluster_id = f"cluster_{'_'.join(sorted(cluster))}"
    
    # Initialize the partial selection with all elements selected
    # BUT only if they haven't been processed yet
    st.session_state.partial_selection = {}
    for element_id in cluster:
        st.session_state.partial_selection[element_id] = element_id not in st.session_state.processed_element_ids
    
    # Get element data
    elements = [st.session_state.embedding_calculator.get_element_by_id(element_id) for element_id in cluster]
    elements = [e for e in elements if e is not None and e['id'] not in st.session_state.processed_element_ids]
    
    # Set default merge term to first element name if there are unprocessed elements
    if elements:
        st.session_state.merge_term = elements[0]['name']
        
        # Set default source by concatenating all unique sources
        sources = set(element['source'] for element in elements)
        if len(sources) == 1:
            st.session_state.merge_source = next(iter(sources))
        else:
            # Join all sources with commas instead of just saying "Combined"
            st.session_state.merge_source = ", ".join(sorted(sources))
    
    st.rerun()

def get_llm_suggestions(elements, cluster_id=None):
    """Get LLM suggestions for a cluster"""
    with st.spinner("Requesting LLM suggestions..."):
        try:
            # Create a simplified list for the LLM
            element_list = []
            for i, element in enumerate(elements):
                element_list.append({
                    "id": str(i+1),  # Simple numeric ID
                    "name": element["name"],
                    "type": element["type"],
                    "source": element["source"]
                })
            
            # Get the current model
            model = st.session_state.current_model
            if not model:
                raise ValueError("No model selected")
            
            # Create the prompt for the LLM
            system_message = st.session_state.system_role
            user_message = f"""
            I need suggestions for merging these similar taxonomy elements:
            
            {json.dumps(element_list, indent=2)}
            
            These elements were identified as semantically similar based on embedding-based similarity analysis.
            
            Please provide:
            1. A recommendation on whether these should be merged as synonyms or kept separate
            2. If they should be merged, suggest the best canonical term to use
            3. Explain your reasoning
            
            Format your response in a conversational style.
            """
            
            # Call the LLM API
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Send API request
            response = requests.post(st.session_state.api_url, json=data)
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Store the suggestions
            st.session_state.llm_suggestions = content
            
            # Select the cluster
            element_ids = [element['id'] for element in elements]
            select_cluster(element_ids, cluster_id)
            
            # Try to extract a suggested term from the content
            suggested_term = None
            term_matches = re.findall(r'canonical term.*?["\']([^"\']+)["\']', content, re.IGNORECASE)
            if term_matches:
                suggested_term = term_matches[0]
            else:
                # Try other patterns
                term_matches = re.findall(r'suggest using ["\']([^"\']+)["\']', content, re.IGNORECASE)
                if term_matches:
                    suggested_term = term_matches[0]
                else:
                    term_matches = re.findall(r'recommend ["\']([^"\']+)["\']', content, re.IGNORECASE)
                    if term_matches:
                        suggested_term = term_matches[0]
            
            # If a term was found, set it in the merge term field
            if suggested_term:
                st.session_state.merge_term = suggested_term
            else:
                # Default to first element name
                st.session_state.merge_term = elements[0]['name']
            
            # Set default source based on common source or "Combined"
            sources = set(element['source'] for element in elements)
            if len(sources) == 1:
                st.session_state.merge_source = next(iter(sources))
            else:
                st.session_state.merge_source = "Combined"
            
            st.success("LLM suggestions received")
            st.rerun()
        
        except Exception as e:
            st.error(f"Error getting suggestions: {str(e)}")


def apply_cluster_decision(element_ids=None):
    """Apply the current decision to the selected cluster or specific elements"""
    # If no element IDs provided, use the full selected cluster
    if element_ids is None:
        element_ids = st.session_state.selected_cluster
    
    # Filter out any elements that have already been processed
    element_ids = [eid for eid in element_ids if eid not in st.session_state.processed_element_ids]
    
    # Check if any elements are left to process
    if not element_ids:
        st.error("No unprocessed elements selected for decision")
        return
    
    # Get the elements from IDs
    elements = [st.session_state.embedding_calculator.get_element_by_id(element_id) for element_id in element_ids]
    elements = [e for e in elements if e is not None]  # Remove any None elements
    
    if not elements:
        st.error("No valid elements found for the selected IDs")
        return
    
    # Validate merge decision
    if st.session_state.decision_type == "merge" and not st.session_state.merge_term:
        st.error("Please enter a term to merge as")
        return
    
    # Record the decision
    decision = {
        "cluster": element_ids,
        "elements": elements,
        "decision_type": st.session_state.decision_type,
        "timestamp": datetime.now().isoformat()
    }
    
    # For merge decisions, add merge details
    if st.session_state.decision_type == "merge":
        decision["merge_term"] = st.session_state.merge_term
        decision["merge_source"] = st.session_state.merge_source
    
    # Add the decision to the list
    st.session_state.decisions.append(decision)
    
    # Mark only the processed elements as processed
    for element_id in element_ids:
        st.session_state.processed_element_ids.add(element_id)
    
    # Reset current selection
    st.session_state.selected_cluster = None
    st.session_state.llm_suggestions = ""
    st.session_state.partial_selection = {}
    
    st.success(f"Decision applied. You've made {len(st.session_state.decisions)} decisions so far.")
    st.rerun()


def finalize_processing():
    """Process all decisions and prepare for review"""
    # Apply all decisions
    apply_all_decisions()
    
    # Update status
    st.session_state.status_message = "Processing complete. Review the results and export if satisfied."


def apply_all_decisions():
    """Apply all decisions to create processed data"""
    # Initialize processed data with copies of original data
    st.session_state.processed_characteristics = st.session_state.characteristics.copy()
    st.session_state.processed_categories = st.session_state.categories.copy()
    
    # Process each decision
    for decision in st.session_state.decisions:
        elements = decision.get("elements", [])
        if not elements:
            continue
        
        # Check element type from first element (all should be same type)
        first_type = elements[0].get("type", "")
        
        # Handle merge decisions
        if decision.get("decision_type") == "merge":
            merge_term = decision.get("merge_term", "")
            merge_source = decision.get("merge_source", "")
            
            if not merge_term:
                continue
            
            # Get element IDs to merge
            element_ids = [elem.get("id", "") for elem in elements]
            
            # Handle merging based on type
            if first_type == "characteristic":
                # Create merged characteristic
                original_terms = [elem.get("name", "") for elem in elements]
                merged_char = {
                    "id": f"merged_char_{len(st.session_state.decisions)}",
                    "name": merge_term,
                    "source": merge_source,
                    "description": f"Merged from: {', '.join(original_terms)}",
                    "type": "characteristic",
                    "original_terms": original_terms
                }
                
                # Remove original characteristics
                st.session_state.processed_characteristics = [
                    char for char in st.session_state.processed_characteristics 
                    if char["id"] not in element_ids
                ]
                
                # Add merged characteristic
                st.session_state.processed_characteristics.append(merged_char)
            
            elif first_type == "category":
                # Create merged category
                original_terms = [elem.get("name", "") for elem in elements]
                merged_cat = {
                    "id": f"merged_cat_{len(st.session_state.decisions)}",
                    "name": merge_term,
                    "source": merge_source,
                    "parent": elements[0].get("parent", ""),  # Use parent from first element
                    "type": "category",
                    "original_terms": original_terms
                }
                
                # Remove original categories
                st.session_state.processed_categories = [
                    cat for cat in st.session_state.processed_categories 
                    if cat["id"] not in element_ids
                ]
                
                # Add merged category
                st.session_state.processed_categories.append(merged_cat)


# Export Functions
def export_json():
    """Export data as JSON"""
    # Ensure we have processed data
    if not st.session_state.processed_characteristics and not st.session_state.processed_categories:
        st.error("Please complete the similarity analysis before exporting.")
        return
    
    # Prepare export data
    export_data = {
        "characteristics": st.session_state.processed_characteristics,
        "categories": st.session_state.processed_categories,
        "metadata": {
            "original_characteristics": len(st.session_state.characteristics),
            "original_categories": len(st.session_state.categories),
            "clusters_analyzed": len(st.session_state.embedding_calculator.clusters) if hasattr(st.session_state.embedding_calculator, 'clusters') else 0,
            "custom_clusters_created": len(st.session_state.custom_clusters),
            "decisions_made": len(st.session_state.decisions),
            "export_date": datetime.now().isoformat(),
            "clustering_method": "HDBSCAN",
            "min_cluster_size": st.session_state.min_cluster_size,
            "min_samples": st.session_state.min_samples,
            "cluster_selection_epsilon": st.session_state.cluster_selection_epsilon
        }
    }
    
    # Convert to JSON string
    json_str = json.dumps(export_data, indent=2)
    
    # Create a download button
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="taxonomy_processed.json",
        mime="application/json"
    )


def export_csv():
    """Export data as CSV"""
    # Ensure we have processed data
    if not st.session_state.processed_characteristics and not st.session_state.processed_categories:
        st.error("Please complete the similarity analysis before exporting.")
        return
    
    # Create CSV data
    csv_data = {}
    
    # Characteristics CSV
    char_df = pd.DataFrame(st.session_state.processed_characteristics)
    if not char_df.empty:
        # Ensure all expected columns exist
        for col in ["id", "name", "source", "description", "type"]:
            if col not in char_df.columns:
                char_df[col] = ""
        
        # Convert original_terms list to string if it exists
        if "original_terms" in char_df.columns:
            char_df["original_terms"] = char_df["original_terms"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        
        csv_data["characteristics.csv"] = char_df.to_csv(index=False)
    
    # Categories CSV
    cat_df = pd.DataFrame(st.session_state.processed_categories)
    if not cat_df.empty:
        # Ensure all expected columns exist
        for col in ["id", "name", "source", "parent", "type"]:
            if col not in cat_df.columns:
                cat_df[col] = ""
        
        # Convert original_terms list to string if it exists
        if "original_terms" in cat_df.columns:
            cat_df["original_terms"] = cat_df["original_terms"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        
        csv_data["categories.csv"] = cat_df.to_csv(index=False)
    
    # Metadata CSV
    metadata = [
        {"key": "original_characteristics", "value": len(st.session_state.characteristics)},
        {"key": "original_categories", "value": len(st.session_state.categories)},
        {"key": "processed_characteristics", "value": len(st.session_state.processed_characteristics)},
        {"key": "processed_categories", "value": len(st.session_state.processed_categories)},
        {"key": "clusters_analyzed", "value": len(st.session_state.embedding_calculator.clusters) if hasattr(st.session_state.embedding_calculator, 'clusters') else 0},
        {"key": "custom_clusters_created", "value": len(st.session_state.custom_clusters)},
        {"key": "decisions_made", "value": len(st.session_state.decisions)},
        {"key": "export_date", "value": datetime.now().isoformat()},
        {"key": "embedding_model", "value": st.session_state.embedding_calculator.model_name},
        {"key": "clustering_method", "value": "HDBSCAN"},
        {"key": "min_cluster_size", "value": st.session_state.min_cluster_size},
        {"key": "min_samples", "value": st.session_state.min_samples},
        {"key": "cluster_selection_epsilon", "value": st.session_state.cluster_selection_epsilon}
    ]
    
    meta_df = pd.DataFrame(metadata)
    csv_data["metadata.csv"] = meta_df.to_csv(index=False)
    
    # Create a zip file containing all CSVs
    import zipfile
    
    # Create a BytesIO object
    zip_buffer = io.BytesIO()
    
    # Create a ZipFile object
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename, data in csv_data.items():
            zipf.writestr(filename, data)
    
    # Reset buffer position
    zip_buffer.seek(0)
    
    # Create a download button
    st.download_button(
        label="Download CSV Files (ZIP)",
        data=zip_buffer,
        file_name="taxonomy_processed_csv.zip",
        mime="application/zip"
    )


def main():
    """Main app function"""
    st.set_page_config(
        page_title="Taxonomy Element Processor",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    initialize_state()
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    
    # Navigation buttons
    if st.sidebar.button("Welcome"):
        st.session_state.current_section = "Welcome"
        st.rerun()
    
    if st.sidebar.button("1. Settings"):
        st.session_state.current_section = "Settings"
        st.rerun()
    
    if st.sidebar.button("2. Data Input"):
        st.session_state.current_section = "Data Input"
        st.rerun()
    
    if st.sidebar.button("3. Similarity Analysis"):
        st.session_state.current_section = "Similarity Analysis"
        st.rerun()
    
    if st.sidebar.button("4. Review & Export"):
        st.session_state.current_section = "Review"
        st.rerun()
    
    # Show current status
    st.sidebar.subheader("Status")
    st.sidebar.text(st.session_state.status_message)
    
    # Element counts
    st.sidebar.subheader("Elements")
    st.sidebar.text(f"Characteristics: {len(st.session_state.characteristics)}")
    st.sidebar.text(f"Categories: {len(st.session_state.categories)}")
    
    # Display the current page
    if st.session_state.current_section == "Welcome":
        welcome_page()
    elif st.session_state.current_section == "Settings":
        settings_page()
    elif st.session_state.current_section == "Data Input":
        data_input_page()
    elif st.session_state.current_section == "Similarity Analysis":
        similarity_analysis_page()
    elif st.session_state.current_section == "Review":
        review_page()


if __name__ == "__main__":
    main()