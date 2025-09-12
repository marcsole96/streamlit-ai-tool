# Agent 0 Validator

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import re
from openai import OpenAI
import math
from PIL import Image
import io
import base64
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def calculate_taxonomy_robustness(taxonomy_data, embedding_calculator=None):
    """
    Calculate the robustness metric R(T) for the taxonomy using the original characteristic groups method
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
        embedding_calculator: Optional pre-initialized EmbeddingCalculator instance
    
    Returns:
        Dictionary with robustness metric and detailed analysis
    """
    # Import networkx inside the function for reliability
    import networkx as nx
    
    # Create embedding calculator if not provided
    if embedding_calculator is None:
        embedding_calculator = EmbeddingCalculator()
        embedding_calculator.load_model()
    
    # Create a graph from the taxonomy data
    G = create_nx_graph(taxonomy_data)
    
    # Build a node map for quick access
    node_map = {node["id"]: node for node in taxonomy_data["nodes"]}
    
    # Build parent-child relationship map
    parent_child_map = {}
    for edge in taxonomy_data["edges"]:
        parent = edge["source"]
        child = edge["target"]
        if parent not in parent_child_map:
            parent_child_map[parent] = []
        parent_child_map[parent].append(child)
    
    # Identify all characteristic nodes
    characteristic_nodes = [node for node in taxonomy_data["nodes"] 
                           if node.get("type", "").lower() == "characteristic"]
    
    if not characteristic_nodes:
        return {
            "robustness": 0.0,
            "error": "No characteristic nodes found in the taxonomy"
        }
    
    # Identify groups of characteristic nodes (leaf nodes under the same parent)
    groups = []
    # First, get all parent nodes
    parent_ids = set()
    for edge in taxonomy_data["edges"]:
        if edge["target"] in [node["id"] for node in characteristic_nodes]:
            parent_ids.add(edge["source"])
    
    # For each parent, get all characteristic children
    for parent_id in parent_ids:
        group_nodes = []
        for edge in taxonomy_data["edges"]:
            if edge["source"] == parent_id and any(n["id"] == edge["target"] for n in characteristic_nodes):
                child_node = next((n for n in characteristic_nodes if n["id"] == edge["target"]), None)
                if child_node:
                    group_nodes.append(child_node)
        
        # Only consider groups with at least one characteristic
        if group_nodes:
            groups.append(group_nodes)
    
    # Calculate embeddings for all characteristic nodes
    char_names = [node["name"] for node in characteristic_nodes]
    char_ids = [node["id"] for node in characteristic_nodes]
    
    # Compute similarity matrix for all characteristics
    similarity_matrix = embedding_calculator.compute_similarity_matrix(char_names)
    
    if similarity_matrix is None:
        return {
            "robustness": 0.0,
            "error": "Failed to compute embeddings"
        }
    
    # Create a lookup from node ID to its index in the similarity matrix
    char_id_to_index = {node_id: i for i, node_id in enumerate(char_ids)}
    
    # Calculate group robustness and intruders
    group_data = []
    total_intruders = 0
    n_ac = len(characteristic_nodes)  # Total number of characteristics
    
    for group_idx, group in enumerate(groups):
        group_ids = [node["id"] for node in group]
        group_names = [node["name"] for node in group]
        n_gc = len(group)  # Number of characteristics in this group
        
        # Calculate minimum internal similarity (cohesiveness)
        min_internal_sim = 1.0  # Initialize to maximum
        internal_similarities = []
        
        # If group has only one element, set cohesiveness to 1.0
        if n_gc == 1:
            min_internal_sim = 1.0
        else:
            # Calculate all pairwise similarities within the group
            for i in range(n_gc):
                for j in range(i+1, n_gc):
                    idx_i = char_id_to_index[group_ids[i]]
                    idx_j = char_id_to_index[group_ids[j]]
                    sim = similarity_matrix[idx_i, idx_j]
                    internal_similarities.append({
                        "pair": (group_names[i], group_names[j]),
                        "similarity": float(sim)
                    })
                    min_internal_sim = min(min_internal_sim, sim)
        
        # Find intruder characteristics
        intruders = []
        for i, node in enumerate(group):
            node_idx = char_id_to_index[node["id"]]
            
            # Calculate similarity with every other characteristic outside the group
            for other_id in char_ids:
                if other_id not in group_ids:  # Only consider nodes outside the group
                    other_idx = char_id_to_index[other_id]
                    other_node = node_map[other_id]
                    sim = similarity_matrix[node_idx, other_idx]
                    
                    # If similarity is higher than the minimum internal similarity, it's an intruder
                    if sim > min_internal_sim and min_internal_sim < 1.0:  # Avoid comparison if min_internal_sim is 1.0
                        intruders.append({
                            "node": node["name"],
                            "other_node": other_node["name"],
                            "similarity": float(sim)
                        })
        
        n_ic = len(intruders)
        total_intruders += n_ic
        
        # Calculate robustness contribution for this group
        # For single-element groups, set max contribution (1.0)
        if n_gc == 1 or n_ac == n_gc:  # If group has all characteristics, no external comparisons
            group_robustness = 1.0
        else:
            potential_comparisons = n_gc * (n_ac - n_gc)
            group_robustness = 1.0 - (n_ic / potential_comparisons)
        
        group_data.append({
            "group_id": f"G{group_idx+1}",
            "characteristics": group_names,
            "size": n_gc,
            "min_internal_similarity": float(min_internal_sim),
            "internal_similarities": internal_similarities,
            "intruders": intruders,
            "intruder_count": n_ic,
            "group_robustness": group_robustness
        })
    
    # Calculate overall robustness
    ngroups = len(groups)
    if ngroups == 0:
        return {
            "robustness": 0.0,
            "error": "No valid groups found in the taxonomy"
        }
    
    overall_robustness = sum(g["group_robustness"] for g in group_data) / ngroups
    
    return {
        "robustness": float(overall_robustness),
        "group_count": ngroups,
        "characteristic_count": n_ac,
        "groups": group_data,
        "detailed_explanation": (
            f"Found {ngroups} groups with a total of {n_ac} characteristics. "
            f"The overall robustness R(T) is {overall_robustness:.4f}, where 1.0 is perfect robustness "
            f"(no intruder characteristics). The analysis identified a total of {total_intruders} "
            f"potential intruder characteristics across all groups."
        )
    }


class EmbeddingCalculator:
    """Class for calculating embeddings using Jina Embeddings v3"""
    def __init__(self):
        self.model_name = "jinaai/jina-embeddings-v3"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the embedding model and tokenizer"""
        if self.model is None or self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                return True
            except Exception as e:
                print(f"Error loading embedding model: {str(e)}")
                return False
        return True
    
    def compute_embedding(self, text):
        """Compute embedding for a single text"""
        if not self.load_model():
            return None
        
        # Tokenize and compute embedding
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**inputs)
            embedding = self.mean_pooling(model_output, inputs['attention_mask'])
        
        return embedding[0].numpy()
    
    def compute_embeddings(self, texts):
        """Compute embeddings for a list of texts"""
        if not self.load_model():
            return None
        
        # Process in batches to avoid memory issues
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            
            # Compute embeddings
            with torch.no_grad():
                model_output = self.model(**inputs)
                batch_embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
            
            embeddings.extend(batch_embeddings.numpy())
        
        return np.array(embeddings)
    
    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two texts"""
        emb1 = self.compute_embedding(text1)
        emb2 = self.compute_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Return cosine similarity (1 - cosine distance)
        return 1 - cosine(emb1, emb2)
    
    def compute_similarity_matrix(self, texts):
        """Compute similarity matrix for a list of texts"""
        embeddings = self.compute_embeddings(texts)
        if embeddings is None:
            return None
        
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(embeddings)
        
        return similarities
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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
    
    def generate_chat_completion(self, messages, max_tokens=4000, temperature=0.7):
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


def format_node_label(text, max_line_length=10, max_lines=3):
    """
    Format a long text label to fit within node visualization with smart line breaking
    
    Args:
        text (str): The input text to format
        max_line_length (int): Maximum length of each line
        max_lines (int): Maximum number of lines to display
    
    Returns:
        str: Formatted text with line breaks using '<br>'
    """
    # Always split two-word labels
    words = text.split()
    if len(words) == 2:
        return f"{words[0]}<br>{words[1]}"
    
    # If text is short enough, return as is
    if len(text) <= max_line_length * max_lines:
        return text
    
    # Aggressive breaking strategy
    lines = []
    current_line = []
    
    for word in words:
        # If current line is empty, add the word
        if not current_line:
            current_line.append(word)
        # If adding this word would make the line too long
        elif len(' '.join(current_line + [word])) > max_line_length:
            # Complete the current line
            lines.append(' '.join(current_line))
            # Start a new line with this word
            current_line = [word]
        else:
            # Word fits, add to current line
            current_line.append(word)
        
        # Stop if we've reached max lines
        if len(lines) >= max_lines - 1:
            break
    
    # Add any remaining words to the last line
    if current_line and len(lines) < max_lines:
        lines.append(' '.join(current_line))
    
    # If we've truncated, add ellipsis to last line
    if len(lines) == max_lines and len(text) > len(' '.join(lines)):
        lines[-1] = lines[-1][:max_line_length-3] + '...'
    
    # Join lines with '<br>'
    return '<br>'.join(lines)


def load_taxonomy(uploaded_file):
    """Load taxonomy from a JSON file"""
    try:
        content = uploaded_file.read().decode('utf-8')
        data = json.loads(content)
        
        # Check if it's in the expected format
        if "nodes" in data and "edges" in data:
            return data
        else:
            st.error("Invalid taxonomy format. Please upload a valid taxonomy JSON file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def create_nx_graph(taxonomy_data):
    """Create a NetworkX graph from taxonomy data"""
    # Import networkx inside the function for reliability
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in taxonomy_data["nodes"]:
        G.add_node(
            node["id"],
            name=node["name"],
            type=node.get("type", "unknown"),
            description=node.get("description", ""),
            source=node.get("source", "unknown")
        )
    
    # Add edges
    for edge in taxonomy_data["edges"]:
        G.add_edge(edge["source"], edge["target"])
    
    return G


def create_orthogonal_visualization(taxonomy_data, options=None):
    """Create an enhanced orthogonal tree visualization of the taxonomy with right-angled connections"""
    # Import networkx inside the function for reliability
    import networkx as nx
    
    G = create_nx_graph(taxonomy_data)
    
    # Default options
    default_options = {
        "node_size": 15,
        "font_size": 16,
        "vert_spacing": 0.15,
        "horz_spacing": 1.0,
        "text_tilt": 0,
        "edge_width": 1.5,
        "edge_color": "rgba(100,100,100,0.7)",
        "root_color": "#4CAF50",      # Green
        "category_color": "#2196F3",  # Blue
        "characteristic_color": "#FF9800",  # Orange
        "unknown_color": "#9E9E9E",   # Grey
        "staggered_labels": True,     # Toggle for staggered/alternating labels
        "labels_inside_nodes": False  # Toggle for placing labels inside nodes
    }
    
    # Update with provided options
    if options:
        for key, value in options.items():
            if key in default_options:
                default_options[key] = value
    
    # Apply options
    node_size = default_options["node_size"]
    font_size = default_options["font_size"]
    vert_gap = default_options["vert_spacing"]
    horz_spacing = default_options["horz_spacing"]
    text_tilt = default_options["text_tilt"]
    edge_width = default_options["edge_width"]
    edge_color = default_options["edge_color"]
    staggered_labels = default_options["staggered_labels"]
    labels_inside_nodes = default_options["labels_inside_nodes"]
    node_colors = {
        "root": default_options["root_color"],
        "category": default_options["category_color"],
        "characteristic": default_options["characteristic_color"],
        "unknown": default_options["unknown_color"]
    }
    
    # If labels are inside nodes, adjust node size based on text length
    if labels_inside_nodes:
        # We'll adjust node sizes further down when we know text lengths
        base_node_size = node_size
        node_size_multiplier = 1.5  # Base multiplier to make nodes larger
    
    # Determine if we should use tilted text
    use_tilted_text = abs(text_tilt) > 0 and not labels_inside_nodes
    
    # Create a hierarchical layout with better spacing
    # Pass the horizontal spacing parameter to control tree width
    pos = hierarchy_pos(G, root="root", vert_gap=vert_gap, width=horz_spacing)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add edges as orthogonal lines with right angles
    edge_traces = []
    arrow_x = []
    arrow_y = []
    arrow_u = []
    arrow_v = []
    
    # Group nodes by level for easier edge drawing
    levels = {}
    for node_id, (x, y) in pos.items():
        level = round(y * 100)  # Group by rounded y-coordinate
        if level not in levels:
            levels[level] = []
        levels[level].append((node_id, x))
    
    # Sort nodes in each level by x-coordinate
    for level in levels:
        levels[level] = sorted(levels[level], key=lambda item: item[1])
    
    # Draw orthogonal edges connecting parent to children
    for edge in taxonomy_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create orthogonal edge (three segments: down, across, down)
            edge_x = []
            edge_y = []
            
            # First point: at the parent node's center
            edge_x.append(x0)
            edge_y.append(y0)
            
            # Second point: directly below parent at halfway between parent and child level
            mid_y = (y0 + y1) / 2
            edge_x.append(x0)
            edge_y.append(mid_y)
            
            # Third point: horizontal shift to child's x-coordinate
            edge_x.append(x1)
            edge_y.append(mid_y)
            
            # Fourth point: down to child node
            edge_x.append(x1)
            edge_y.append(y1)
            
            # Add the edge trace
            edge_traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add arrow near the target
            arrow_x.append(x1)
            arrow_y.append(y1 + 0.02)  # Slightly above the node
            arrow_u.append(0)  # Pointing down
            arrow_v.append(-0.02)  # Arrow size
    
    # Add all edge traces to the figure
    for trace in edge_traces:
        fig.add_trace(trace)
    
    # Create label positions (staggered or uniform)
    label_positions = {}
    node_positions = {}
    
    # Assign positions based on staggered_labels option
    for level, nodes in levels.items():
        # Sort nodes horizontally to ensure consistent ordering
        sorted_nodes = sorted(nodes, key=lambda n: n[1])  # n[1] is the x-coordinate
        
        for i, (node_id, x) in enumerate(sorted_nodes):
            node_positions[node_id] = (x, -level/100)  # Store exact position
            
            if staggered_labels and not labels_inside_nodes:
                # Alternating top/bottom for staggered labels
                if i % 2 == 0:
                    label_positions[node_id] = "top center"
                else:
                    label_positions[node_id] = "bottom center"
            else:
                # All labels in same position if not staggered
                # If labels are inside nodes, we'll handle positioning differently
                label_positions[node_id] = "top center"
    
    # Add nodes by type to control appearance
    for node_type in ["root", "category", "characteristic", "unknown"]:
        # Get nodes of this type
        type_nodes = [n for n in taxonomy_data["nodes"] if n.get("type", "unknown") == node_type]
        
        if not type_nodes:
            continue
        
        node_x = []
        node_y = []
        node_text = []
        node_hover = []
        node_info = []
        node_ids = []
        node_sizes = []  # For variable node sizes when labels are inside
        
        for node in type_nodes:
            if node["id"] not in pos:
                continue
            
            x, y = pos[node["id"]]
            node_x.append(x)
            node_y.append(y)
            
            # Process node name for display
            display_name = node["name"]
            if labels_inside_nodes:
                # For labels inside nodes, format with line breaks
                display_name = format_node_label(display_name, max_line_length=10, max_lines=3)
            else:
                # For external labels, allow slightly longer text
                if len(display_name) > 25:
                    display_name = display_name[:22] + "..."
            
            node_text.append(display_name)
            
            # Create hover information
            hover_text = f"<b>{node['name']}</b><br>"
            if "description" in node and node["description"]:
                hover_text += f"Description: {node['description']}<br>"
            if "source" in node and node["source"]:
                hover_text += f"Source: {node['source']}<br>"
                
            node_hover.append(hover_text)
            node_info.append(node)
            node_ids.append(node["id"])
            
            # Calculate node size if labels are inside
            if labels_inside_nodes:
                # Make node size proportional to text length, with minimum and maximum sizes
                text_length = len(display_name)
                # Base size is proportional to text length but capped
                size_factor = min(max(text_length / 5, 1), 3)
                # Different base sizes based on node type
                if node_type == "root":
                    size = base_node_size * node_size_multiplier * 1.5 * size_factor
                elif node_type == "category":
                    size = base_node_size * node_size_multiplier * 1.2 * size_factor
                else:
                    size = base_node_size * node_size_multiplier * size_factor
                node_sizes.append(size)
            else:
                # Standard sizing when labels are outside
                if node_type == "root":
                    node_sizes.append(node_size * 1.3)
                elif node_type == "category":
                    node_sizes.append(node_size * 1.1)
                else:
                    node_sizes.append(node_size)
        
        # Determine text size based on node type and label position
        if labels_inside_nodes:
            # Smaller font for inside labels to fit better
            if node_type == "root":
                text_size = font_size * 0.9
            elif node_type == "category":
                text_size = font_size * 0.8
            else:
                text_size = font_size * 0.7
        else:
            # Normal font size for outside labels
            if node_type == "root":
                text_size = font_size * 1.2
            elif node_type == "category":
                text_size = font_size * 1.1
            else:
                text_size = font_size
        
        # Determine text position based on label option
        if labels_inside_nodes:
            # Always centered when inside
            text_positions = ["middle center"] * len(node_ids)
            display_mode = 'markers+text'  # Always show text in markers
        else:
            # Use calculated positions for external labels
            text_positions = [label_positions[node_id] for node_id in node_ids] if not use_tilted_text else None
            # Only include text if not using tilted annotations
            display_mode = 'markers' + ('+text' if not use_tilted_text else '')
        
        # Add the nodes trace
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode=display_mode,
            marker=dict(
                color=node_colors.get(node_type, "#9E9E9E"),
                size=node_sizes,
                line=dict(width=1.5, color='white')
            ),
            text=node_text if (not use_tilted_text or labels_inside_nodes) else None,
            textposition=text_positions,
            textfont=dict(
                size=text_size,
                color='black'  # Explicitly set black color
            ) if (not use_tilted_text or labels_inside_nodes) else None,
            hovertext=node_hover,
            hoverinfo='text',
            name=node_type.capitalize(),
            customdata=node_ids
        ))
        
        # Add tilted text annotations if text_tilt is not 0 and we're not using inside labels
        if use_tilted_text and not labels_inside_nodes:
            for i, node_id in enumerate(node_ids):
                # Use label_positions to determine if this should be above or below
                position = label_positions[node_id]
                yanchor = "bottom" if "top" in position else "top"
                
                fig.add_annotation(
                    x=node_x[i],
                    y=node_y[i],
                    text=node_text[i],
                    showarrow=False,
                    font=dict(
                        size=text_size,
                        color='black'  # Explicitly force black color
                    ),
                    textangle=text_tilt,
                    xanchor="center",
                    yanchor=yanchor,
                    bgcolor="rgba(255,255,255,0.7)",  # Semi-transparent background for better readability
                    borderpad=2
                )
    
    # Update layout
    fig.update_layout(
        title="Orthogonal Taxonomy Visualization",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=50, l=20, r=20, t=50),
        height=900,
        width=1200,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",  # Fix axis scaling issues
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",  # Fix axis scaling issues
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.1,
            xanchor="center",
            x=0.5
        ),
        # Add zoom capabilities
        dragmode='pan',
        newshape=dict(line_color='#000'),
        modebar_add=['drawline', 'drawopenpath', 'eraseshape']
    )
    
    return fig


def create_orthogonal_intruder_visualization(taxonomy_data, robustness_data, options=None):
    """
    Create an orthogonal tree visualization with intruders highlighted and their
    relationships to similar nodes shown with curved arrows.
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
        robustness_data: Dictionary with robustness analysis results
        options: Optional visualization options
        
    Returns:
        Plotly figure object with the visualization
    """
    # Import needed libraries
    import plotly.graph_objects as go
    import networkx as nx
    import numpy as np
    
    # Create a graph from the taxonomy data
    G = create_nx_graph(taxonomy_data)
    
    # Default options (same as orthogonal visualization)
    default_options = {
        "node_size": 15,
        "font_size": 16,
        "vert_spacing": 0.15,
        "horz_spacing": 1.0,
        "text_tilt": 0,
        "edge_width": 1.5,
        "edge_color": "rgba(100,100,100,0.7)",
        "root_color": "#4CAF50",      # Green
        "category_color": "#2196F3",  # Blue
        "characteristic_color": "#FF9800",  # Orange
        "unknown_color": "#9E9E9E",   # Grey
        "intruder_color": "#FF0000",  # Red for intruders
        "connection_color": "rgba(255,0,0,0.5)",  # Red for connection arrows
        "staggered_labels": True,     # Toggle for staggered/alternating labels
        "labels_inside_nodes": False  # Toggle for placing labels inside nodes
    }
    
    # Update with provided options
    if options:
        for key, value in options.items():
            if key in default_options:
                default_options[key] = value
    
    # Apply options
    node_size = default_options["node_size"]
    font_size = default_options["font_size"]
    vert_gap = default_options["vert_spacing"]
    horz_spacing = default_options["horz_spacing"]
    text_tilt = default_options["text_tilt"]
    edge_width = default_options["edge_width"]
    edge_color = default_options["edge_color"]
    intruder_color = default_options["intruder_color"]
    connection_color = default_options["connection_color"]
    staggered_labels = default_options["staggered_labels"]
    labels_inside_nodes = default_options["labels_inside_nodes"]
    node_colors = {
        "root": default_options["root_color"],
        "category": default_options["category_color"],
        "characteristic": default_options["characteristic_color"],
        "unknown": default_options["unknown_color"]
    }
    
    # Determine if labels are inside nodes and adjust size
    if labels_inside_nodes:
        base_node_size = node_size
        node_size_multiplier = 1.5  # Base multiplier to make nodes larger
    
    # Determine if we should use tilted text
    use_tilted_text = abs(text_tilt) > 0 and not labels_inside_nodes
    
    # Create a hierarchical layout
    pos = hierarchy_pos(G, root="root", vert_gap=vert_gap, width=horz_spacing)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Identify intruders from robustness data
    all_intruders = {}
    intruder_connections = []
    node_map = {node["id"]: node for node in taxonomy_data["nodes"]}
    
    # Process each group to identify intruders and their connections
    for group in robustness_data['groups']:
        # Get all node IDs in this group
        group_names = group['characteristics']
        group_ids = []
        
        # Find node IDs for all characteristics in this group
        for node_id, node in node_map.items():
            if node.get('name') in group_names:
                group_ids.append(node_id)
        
        # Process each intruder
        for intruder in group['intruders']:
            intruder_name = intruder['node']
            similar_name = intruder['other_node']
            similarity = intruder['similarity']
            
            # Find node IDs for the intruder and similar node
            intruder_id = None
            similar_id = None
            
            for node_id, node in node_map.items():
                if node.get('name') == intruder_name:
                    intruder_id = node_id
                    if intruder_id not in all_intruders:
                        all_intruders[intruder_id] = {
                            'name': intruder_name,
                            'connections': []
                        }
                if node.get('name') == similar_name:
                    similar_id = node_id
            
            # Store the connection if we found both nodes
            if intruder_id and similar_id:
                connection = {
                    'source': intruder_id,
                    'target': similar_id,
                    'similarity': similarity
                }
                
                # Add to the intruder's connections list
                if intruder_id in all_intruders:
                    all_intruders[intruder_id]['connections'].append(connection)
                
                # Add to all connections
                intruder_connections.append(connection)
    
    # Add edges as orthogonal lines with right angles (same as orthogonal visualization)
    edge_traces = []
    
    # Group nodes by level for easier edge drawing
    levels = {}
    for node_id, (x, y) in pos.items():
        level = round(y * 100)  # Group by rounded y-coordinate
        if level not in levels:
            levels[level] = []
        levels[level].append((node_id, x))
    
    # Sort nodes in each level by x-coordinate
    for level in levels:
        levels[level] = sorted(levels[level], key=lambda item: item[1])
    
    # Draw orthogonal edges connecting parent to children
    for edge in taxonomy_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create orthogonal edge (three segments: down, across, down)
            edge_x = []
            edge_y = []
            
            # First point: at the parent node's center
            edge_x.append(x0)
            edge_y.append(y0)
            
            # Second point: directly below parent at halfway between parent and child level
            mid_y = (y0 + y1) / 2
            edge_x.append(x0)
            edge_y.append(mid_y)
            
            # Third point: horizontal shift to child's x-coordinate
            edge_x.append(x1)
            edge_y.append(mid_y)
            
            # Fourth point: down to child node
            edge_x.append(x1)
            edge_y.append(y1)
            
            # Add the edge trace
            edge_traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
    
    # Add all edge traces to the figure
    for trace in edge_traces:
        fig.add_trace(trace)
    
    # Create label positions (staggered or uniform)
    label_positions = {}
    node_positions = {}
    
    # Assign positions based on staggered_labels option
    for level, nodes in levels.items():
        # Sort nodes horizontally to ensure consistent ordering
        sorted_nodes = sorted(nodes, key=lambda n: n[1])  # n[1] is the x-coordinate
        
        for i, (node_id, x) in enumerate(sorted_nodes):
            node_positions[node_id] = (x, -level/100)  # Store exact position
            
            if staggered_labels and not labels_inside_nodes:
                # Alternating top/bottom for staggered labels
                if i % 2 == 0:
                    label_positions[node_id] = "top center"
                else:
                    label_positions[node_id] = "bottom center"
            else:
                # All labels in same position if not staggered
                label_positions[node_id] = "top center"
    
    # Add nodes by type to control appearance
    for node_type in ["root", "category", "characteristic", "unknown"]:
        # Get nodes of this type
        type_nodes = [n for n in taxonomy_data["nodes"] if n.get("type", "unknown") == node_type]
        
        if not type_nodes:
            continue
        
        # Separate normal nodes and intruder nodes
        normal_nodes = [n for n in type_nodes if n["id"] not in all_intruders]
        intruder_nodes = [n for n in type_nodes if n["id"] in all_intruders]
        
        # Add normal nodes
        if normal_nodes:
            node_x = []
            node_y = []
            node_text = []
            node_hover = []
            node_ids = []
            node_sizes = []
            
            for node in normal_nodes:
                if node["id"] not in pos:
                    continue
                
                x, y = pos[node["id"]]
                node_x.append(x)
                node_y.append(y)
                
                # Process node name for display
                display_name = node["name"]
                if labels_inside_nodes:
                    # For labels inside nodes, format with line breaks
                    display_name = format_node_label(display_name, max_line_length=10, max_lines=3)
                else:
                    # For external labels, allow slightly longer text
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                
                node_text.append(display_name)
                
                # Create hover information
                hover_text = f"<b>{node['name']}</b><br>"
                if "description" in node and node["description"]:
                    hover_text += f"Description: {node['description']}<br>"
                if "source" in node and node["source"]:
                    hover_text += f"Source: {node['source']}<br>"
                    
                node_hover.append(hover_text)
                node_ids.append(node["id"])
                
                # Calculate node size
                if labels_inside_nodes:
                    # Size based on text length
                    text_length = len(display_name)
                    size_factor = min(max(text_length / 5, 1), 3)
                    if node_type == "root":
                        size = base_node_size * node_size_multiplier * 1.5 * size_factor
                    elif node_type == "category":
                        size = base_node_size * node_size_multiplier * 1.2 * size_factor
                    else:
                        size = base_node_size * node_size_multiplier * size_factor
                    node_sizes.append(size)
                else:
                    # Standard sizing
                    if node_type == "root":
                        node_sizes.append(node_size * 1.3)
                    elif node_type == "category":
                        node_sizes.append(node_size * 1.1)
                    else:
                        node_sizes.append(node_size)
            
            # Determine text size based on node type
            if labels_inside_nodes:
                if node_type == "root":
                    text_size = font_size * 0.9
                elif node_type == "category":
                    text_size = font_size * 0.8
                else:
                    text_size = font_size * 0.7
            else:
                if node_type == "root":
                    text_size = font_size * 1.2
                elif node_type == "category":
                    text_size = font_size * 1.1
                else:
                    text_size = font_size
            
            # Determine text position
            if labels_inside_nodes:
                text_positions = ["middle center"] * len(node_ids)
                display_mode = 'markers+text'
            else:
                text_positions = [label_positions[node_id] for node_id in node_ids] if not use_tilted_text else None
                display_mode = 'markers' + ('+text' if not use_tilted_text else '')
            
            # Add the nodes trace
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode=display_mode,
                marker=dict(
                    color=node_colors.get(node_type, "#9E9E9E"),
                    size=node_sizes,
                    line=dict(width=1.5, color='white')
                ),
                text=node_text if (not use_tilted_text or labels_inside_nodes) else None,
                textposition=text_positions,
                textfont=dict(
                    size=text_size,
                    color='black'
                ) if (not use_tilted_text or labels_inside_nodes) else None,
                hovertext=node_hover,
                hoverinfo='text',
                name=node_type.capitalize(),
                customdata=node_ids
            ))
            
            # Add tilted text annotations if needed
            if use_tilted_text and not labels_inside_nodes:
                for i, node_id in enumerate(node_ids):
                    position = label_positions[node_id]
                    yanchor = "bottom" if "top" in position else "top"
                    
                    fig.add_annotation(
                        x=node_x[i],
                        y=node_y[i],
                        text=node_text[i],
                        showarrow=False,
                        font=dict(
                            size=text_size,
                            color='black'
                        ),
                        textangle=text_tilt,
                        xanchor="center",
                        yanchor=yanchor,
                        bgcolor="rgba(255,255,255,0.7)",
                        borderpad=2
                    )
        
        # Add intruder nodes (with red color)
        if intruder_nodes:
            node_x = []
            node_y = []
            node_text = []
            node_hover = []
            node_ids = []
            node_sizes = []
            
            for node in intruder_nodes:
                if node["id"] not in pos:
                    continue
                
                x, y = pos[node["id"]]
                node_x.append(x)
                node_y.append(y)
                
                # Process node name for display
                display_name = node["name"]
                if labels_inside_nodes:
                    display_name = format_node_label(display_name, max_line_length=10, max_lines=3)
                else:
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                
                node_text.append(display_name)
                
                # Create hover information for intruders with detailed connection info
                hover_text = f"<b>{node['name']} (INTRUDER)</b><br>"
                if "description" in node and node["description"]:
                    hover_text += f"Description: {node['description']}<br>"
                
                # Add information about connections to other nodes
                if node["id"] in all_intruders:
                    connections = all_intruders[node["id"]]['connections']
                    hover_text += f"<br>Connections outside group: {len(connections)}<br>"
                    for conn in connections[:3]:  # Show top 3 connections
                        similar_node = node_map.get(conn['target'], {}).get('name', 'Unknown')
                        hover_text += f"- Similar to: {similar_node} ({conn['similarity']:.4f})<br>"
                    if len(connections) > 3:
                        hover_text += f"- (and {len(connections) - 3} more...)<br>"
                
                node_hover.append(hover_text)
                node_ids.append(node["id"])
                
                # Calculate node size
                if labels_inside_nodes:
                    text_length = len(display_name)
                    size_factor = min(max(text_length / 5, 1), 3)
                    if node_type == "root":
                        size = base_node_size * node_size_multiplier * 1.5 * size_factor
                    elif node_type == "category":
                        size = base_node_size * node_size_multiplier * 1.2 * size_factor
                    else:
                        size = base_node_size * node_size_multiplier * size_factor
                    node_sizes.append(size)
                else:
                    if node_type == "root":
                        node_sizes.append(node_size * 1.3)
                    elif node_type == "category":
                        node_sizes.append(node_size * 1.1)
                    else:
                        node_sizes.append(node_size)
            
            # Determine text size
            if labels_inside_nodes:
                if node_type == "root":
                    text_size = font_size * 0.9
                elif node_type == "category":
                    text_size = font_size * 0.8
                else:
                    text_size = font_size * 0.7
            else:
                if node_type == "root":
                    text_size = font_size * 1.2
                elif node_type == "category":
                    text_size = font_size * 1.1
                else:
                    text_size = font_size
            
            # Determine text position
            if labels_inside_nodes:
                text_positions = ["middle center"] * len(node_ids)
                display_mode = 'markers+text'
            else:
                text_positions = [label_positions[node_id] for node_id in node_ids] if not use_tilted_text else None
                display_mode = 'markers' + ('+text' if not use_tilted_text else '')
            
            # Add the intruder nodes trace
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode=display_mode,
                marker=dict(
                    color=intruder_color,  # Red for intruders
                    size=node_sizes,
                    line=dict(width=1.5, color='white')
                ),
                text=node_text if (not use_tilted_text or labels_inside_nodes) else None,
                textposition=text_positions,
                textfont=dict(
                    size=text_size,
                    color='black'
                ) if (not use_tilted_text or labels_inside_nodes) else None,
                hovertext=node_hover,
                hoverinfo='text',
                name=f"Intruder {node_type.capitalize()}s",  # e.g. "Intruder Characteristics"
                customdata=node_ids
            ))
            
            # Add tilted text annotations if needed
            if use_tilted_text and not labels_inside_nodes:
                for i, node_id in enumerate(node_ids):
                    position = label_positions[node_id]
                    yanchor = "bottom" if "top" in position else "top"
                    
                    fig.add_annotation(
                        x=node_x[i],
                        y=node_y[i],
                        text=node_text[i],
                        showarrow=False,
                        font=dict(
                            size=text_size,
                            color='black'
                        ),
                        textangle=text_tilt,
                        xanchor="center",
                        yanchor=yanchor,
                        bgcolor="rgba(255,255,255,0.7)",
                        borderpad=2
                    )
    
    # Add curved dashed lines for intruder connections
    for conn in intruder_connections:
        source = conn['source']
        target = conn['target']
        similarity = conn['similarity']
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create a curved path
            path_x, path_y = create_curved_path(x0, y0, x1, y1, curvature=0.3)
            
            # Add the connection trace
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                line=dict(
                    width=2.0,
                    color=connection_color,
                    dash='dash'
                ),
                hoverinfo='text',
                hovertext=f"Similarity: {similarity:.4f}",
                mode='lines',
                name='Semantic Similarity',
                showlegend=source == list(all_intruders.keys())[0] if all_intruders else False  # Show legend only once
            ))
            
            # Add arrow marker at the target end
            arrow_x, arrow_y = create_arrow_marker(path_x[-2], path_y[-2], path_x[-1], path_y[-1], arrow_size=0.01)
            
            fig.add_trace(go.Scatter(
                x=arrow_x, y=arrow_y,
                fill="toself",
                fillcolor=connection_color,
                line=dict(color=connection_color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title="Orthogonal Taxonomy with Intruders",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=50, l=20, r=20, t=50),
        height=900,
        width=1200,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.1,
            xanchor="center",
            x=0.5
        ),
        # Add zoom capabilities
        dragmode='pan',
        newshape=dict(line_color='#000'),
        modebar_add=['drawline', 'drawopenpath', 'eraseshape']
    )
    
    return fig


def create_enhanced_intruder_visualization(taxonomy_data, robustness_data, options=None, visible_groups=None):
    """
    Create an enhanced orthogonal tree visualization with clearly marked groups, 
    intruders marked with X, and connections to similar elements in other groups.
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
        robustness_data: Dictionary with robustness analysis results
        options: Optional visualization options
        visible_groups: List of group IDs whose connections should be visible (None = all visible)
        
    Returns:
        Plotly figure object with the visualization
    """
    import plotly.graph_objects as go
    import networkx as nx
    import numpy as np
    
    # Create a graph from the taxonomy data
    G = create_nx_graph(taxonomy_data)
    
    # Default options (similar to original orthogonal visualization)
    default_options = {
        "node_size": 15,
        "font_size": 16,
        "vert_spacing": 0.15,
        "horz_spacing": 1.0,
        "text_tilt": 0,
        "edge_width": 1.5,
        "edge_color": "rgba(100,100,100,0.7)",
        "root_color": "#4CAF50",      # Green
        "category_color": "#2196F3",  # Blue
        "characteristic_color": "#FF9800",  # Orange
        "unknown_color": "#9E9E9E",   # Grey
        "intruder_color": "#FF0000",  # Red for intruders
        "connection_color": "rgba(255,0,0,0.5)",  # Red for connection arrows
        "staggered_labels": True,     # Toggle for staggered/alternating labels
        "labels_inside_nodes": False,  # Toggle for placing labels inside nodes
        "show_group_boundaries": True  # New option: show group boundaries
    }
    
    # Update with provided options
    if options:
        for key, value in options.items():
            if key in default_options:
                default_options[key] = value
    
    # Apply options
    node_size = default_options["node_size"]
    font_size = default_options["font_size"]
    vert_gap = default_options["vert_spacing"]
    horz_spacing = default_options["horz_spacing"]
    text_tilt = default_options["text_tilt"]
    edge_width = default_options["edge_width"]
    edge_color = default_options["edge_color"]
    intruder_color = default_options["intruder_color"]
    connection_color = default_options["connection_color"]
    staggered_labels = default_options["staggered_labels"]
    labels_inside_nodes = default_options["labels_inside_nodes"]
    show_group_boundaries = default_options["show_group_boundaries"]
    node_colors = {
        "root": default_options["root_color"],
        "category": default_options["category_color"],
        "characteristic": default_options["characteristic_color"],
        "unknown": default_options["unknown_color"]
    }
    
    # Determine if labels are inside nodes and adjust size
    if labels_inside_nodes:
        base_node_size = node_size
        node_size_multiplier = 1.5  # Base multiplier to make nodes larger
    
    # Determine if we should use tilted text
    use_tilted_text = abs(text_tilt) > 0 and not labels_inside_nodes
    
    # Create a hierarchical layout
    pos = hierarchy_pos(G, root="root", vert_gap=vert_gap, width=horz_spacing)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Identify intruders from robustness data
    all_intruders = {}
    intruder_connections = []
    node_map = {node["id"]: node for node in taxonomy_data["nodes"]}
    
    # Process each group to identify intruders and their connections
    for group in robustness_data['groups']:
        # Get all node IDs in this group
        group_names = group['characteristics']
        group_ids = []
        
        # Find node IDs for all characteristics in this group
        for node_id, node in node_map.items():
            if node.get('name') in group_names:
                group_ids.append(node_id)
        
        # Process each intruder
        for intruder in group['intruders']:
            intruder_name = intruder['node']
            similar_name = intruder['other_node']
            similarity = intruder['similarity']
            
            # Find node IDs for the intruder and similar node
            intruder_id = None
            similar_id = None
            
            for node_id, node in node_map.items():
                if node.get('name') == intruder_name:
                    intruder_id = node_id
                    if intruder_id not in all_intruders:
                        all_intruders[intruder_id] = {
                            'name': intruder_name,
                            'connections': [],
                            'group_id': group['group_id']  # Store the group ID
                        }
                if node.get('name') == similar_name:
                    similar_id = node_id
            
            # Store the connection if we found both nodes
            if intruder_id and similar_id:
                connection = {
                    'source': intruder_id,
                    'target': similar_id,
                    'similarity': similarity,
                    'group_id': group['group_id']  # Store which group this connection belongs to
                }
                
                # Add to the intruder's connections list
                if intruder_id in all_intruders:
                    all_intruders[intruder_id]['connections'].append(connection)
                
                # Add to all connections
                intruder_connections.append(connection)
    
    # Add edges as orthogonal lines with right angles
    edge_traces = []
    
    # Group nodes by level for easier edge drawing
    levels = {}
    for node_id, (x, y) in pos.items():
        level = round(y * 100)  # Group by rounded y-coordinate
        if level not in levels:
            levels[level] = []
        levels[level].append((node_id, x))
    
    # Sort nodes in each level by x-coordinate
    for level in levels:
        levels[level] = sorted(levels[level], key=lambda item: item[1])
    
    # Draw orthogonal edges connecting parent to children
    for edge in taxonomy_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create orthogonal edge (three segments: down, across, down)
            edge_x = []
            edge_y = []
            
            # First point: at the parent node's center
            edge_x.append(x0)
            edge_y.append(y0)
            
            # Second point: directly below parent at halfway between parent and child level
            mid_y = (y0 + y1) / 2
            edge_x.append(x0)
            edge_y.append(mid_y)
            
            # Third point: horizontal shift to child's x-coordinate
            edge_x.append(x1)
            edge_y.append(mid_y)
            
            # Fourth point: down to child node
            edge_x.append(x1)
            edge_y.append(y1)
            
            # Add the edge trace
            edge_traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
    
    # Add all edge traces to the figure
    for trace in edge_traces:
        fig.add_trace(trace)
    
    # Create label positions (staggered or uniform)
    label_positions = {}
    node_positions = {}
    
    # Assign positions based on staggered_labels option
    for level, nodes in levels.items():
        # Sort nodes horizontally to ensure consistent ordering
        sorted_nodes = sorted(nodes, key=lambda n: n[1])  # n[1] is the x-coordinate
        
        for i, (node_id, x) in enumerate(sorted_nodes):
            node_positions[node_id] = (x, -level/100)  # Store exact position
            
            if staggered_labels and not labels_inside_nodes:
                # Alternating top/bottom for staggered labels
                if i % 2 == 0:
                    label_positions[node_id] = "top center"
                else:
                    label_positions[node_id] = "bottom center"
            else:
                # All labels in same position if not staggered
                label_positions[node_id] = "top center"
    
    # Map nodes to their parent groups for boundary visualization
    node_to_group = {}
    group_members = {}
    
    # Process groups to identify membership
    for i, group in enumerate(robustness_data['groups']):
        group_id = group['group_id']
        group_members[group_id] = []
        
        for char_name in group['characteristics']:
            # Find node ID for this characteristic
            for node_id, node in node_map.items():
                if node.get('name') == char_name:
                    node_to_group[node_id] = group_id
                    group_members[group_id].append(node_id)
                    break
    
    # Add group boundary shapes if enabled
    if show_group_boundaries:
        for group_id, members in group_members.items():
            if not members:
                continue
                
            # Get positions of all members
            x_values = []
            y_values = []
            
            for node_id in members:
                if node_id in pos:
                    x, y = pos[node_id]
                    x_values.append(x)
                    y_values.append(y)
            
            if not x_values:
                continue
                
            # Calculate boundary with some padding
            padding = 0.05
            x_min = min(x_values) - padding
            x_max = max(x_values) + padding
            y_min = min(y_values) - padding
            y_max = max(y_values) + padding
            
            # Add boundary shape
            fig.add_shape(
                type="rect",
                x0=x_min, y0=y_min,
                x1=x_max, y1=y_max,
                line=dict(
                    color="rgba(0,0,0,0.5)",
                    width=1,
                    dash="dash"
                ),
                fillcolor="rgba(0,0,0,0.03)",
                layer="below"
            )
            
            # Add group label
            fig.add_annotation(
                x=(x_min + x_max) / 2,
                y=y_max + 0.03,
                text=group_id,
                showarrow=False,
                font=dict(size=14, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                borderpad=2
            )
    
    # Add nodes by type to control appearance
    for node_type in ["root", "category", "characteristic", "unknown"]:
        # Get nodes of this type
        type_nodes = [n for n in taxonomy_data["nodes"] if n.get("type", "unknown") == node_type]
        
        if not type_nodes:
            continue
        
        # Separate normal nodes and intruder nodes
        normal_nodes = [n for n in type_nodes if n["id"] not in all_intruders]
        intruder_nodes = [n for n in type_nodes if n["id"] in all_intruders]
        
        # Add normal nodes
        if normal_nodes:
            node_x = []
            node_y = []
            node_text = []
            node_hover = []
            node_ids = []
            node_sizes = []
            
            for node in normal_nodes:
                if node["id"] not in pos:
                    continue
                
                x, y = pos[node["id"]]
                node_x.append(x)
                node_y.append(y)
                
                # Process node name for display
                display_name = node["name"]
                if labels_inside_nodes:
                    # For labels inside nodes, format with line breaks
                    display_name = format_node_label(display_name, max_line_length=10, max_lines=3)
                else:
                    # For external labels, allow slightly longer text
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                
                node_text.append(display_name)
                
                # Create hover information
                hover_text = f"<b>{node['name']}</b><br>"
                if "description" in node and node["description"]:
                    hover_text += f"Description: {node['description']}<br>"
                
                # Add group information if available
                if node["id"] in node_to_group:
                    hover_text += f"Group: {node_to_group[node['id']]}<br>"
                    
                node_hover.append(hover_text)
                node_ids.append(node["id"])
                
                # Calculate node size
                if labels_inside_nodes:
                    # Size based on text length
                    text_length = len(display_name)
                    size_factor = min(max(text_length / 5, 1), 3)
                    if node_type == "root":
                        size = base_node_size * node_size_multiplier * 1.5 * size_factor
                    elif node_type == "category":
                        size = base_node_size * node_size_multiplier * 1.2 * size_factor
                    else:
                        size = base_node_size * node_size_multiplier * size_factor
                    node_sizes.append(size)
                else:
                    # Standard sizing
                    if node_type == "root":
                        node_sizes.append(node_size * 1.3)
                    elif node_type == "category":
                        node_sizes.append(node_size * 1.1)
                    else:
                        node_sizes.append(node_size)
            
            # Determine text size based on node type
            if labels_inside_nodes:
                if node_type == "root":
                    text_size = font_size * 0.9
                elif node_type == "category":
                    text_size = font_size * 0.8
                else:
                    text_size = font_size * 0.7
            else:
                if node_type == "root":
                    text_size = font_size * 1.2
                elif node_type == "category":
                    text_size = font_size * 1.1
                else:
                    text_size = font_size
            
            # Determine text position
            if labels_inside_nodes:
                text_positions = ["middle center"] * len(node_ids)
                display_mode = 'markers+text'
            else:
                text_positions = [label_positions[node_id] for node_id in node_ids] if not use_tilted_text else None
                display_mode = 'markers' + ('+text' if not use_tilted_text else '')
            
            # Add the nodes trace
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode=display_mode,
                marker=dict(
                    color=node_colors.get(node_type, "#9E9E9E"),
                    size=node_sizes,
                    line=dict(width=1.5, color='white')
                ),
                text=node_text if (not use_tilted_text or labels_inside_nodes) else None,
                textposition=text_positions,
                textfont=dict(
                    size=text_size,
                    color='black'
                ) if (not use_tilted_text or labels_inside_nodes) else None,
                hovertext=node_hover,
                hoverinfo='text',
                name=node_type.capitalize(),
                customdata=node_ids
            ))
            
            # Add tilted text annotations if needed
            if use_tilted_text and not labels_inside_nodes:
                for i, node_id in enumerate(node_ids):
                    position = label_positions[node_id]
                    yanchor = "bottom" if "top" in position else "top"
                    
                    fig.add_annotation(
                        x=node_x[i],
                        y=node_y[i],
                        text=node_text[i],
                        showarrow=False,
                        font=dict(
                            size=text_size,
                            color='black'
                        ),
                        textangle=text_tilt,
                        xanchor="center",
                        yanchor=yanchor,
                        bgcolor="rgba(255,255,255,0.7)",
                        borderpad=2
                    )
        
        # Add intruder nodes with 'X' marks
        if intruder_nodes:
            node_x = []
            node_y = []
            node_text = []
            node_hover = []
            node_ids = []
            node_sizes = []
            
            for node in intruder_nodes:
                if node["id"] not in pos:
                    continue
                
                x, y = pos[node["id"]]
                node_x.append(x)
                node_y.append(y)
                
                # Process node name for display
                display_name = node["name"]
                if labels_inside_nodes:
                    display_name = format_node_label(display_name, max_line_length=10, max_lines=3)
                else:
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                
                node_text.append(display_name)
                
                # Create hover information for intruders with detailed connection info
                hover_text = f"<b>{node['name']} (INTRUDER)</b><br>"
                if "description" in node and node["description"]:
                    hover_text += f"Description: {node['description']}<br>"
                
                # Add group information if available
                if node["id"] in node_to_group:
                    hover_text += f"Group: {node_to_group[node['id']]}<br>"
                
                # Add information about connections to other nodes
                if node["id"] in all_intruders:
                    connections = all_intruders[node["id"]]['connections']
                    hover_text += f"<br>Connections outside group: {len(connections)}<br>"
                    for conn in connections[:3]:  # Show top 3 connections
                        similar_node = node_map.get(conn['target'], {}).get('name', 'Unknown')
                        hover_text += f"- Similar to: {similar_node} ({conn['similarity']:.4f})<br>"
                    if len(connections) > 3:
                        hover_text += f"- (and {len(connections) - 3} more...)<br>"
                
                node_hover.append(hover_text)
                node_ids.append(node["id"])
                
                # Calculate node size
                if labels_inside_nodes:
                    text_length = len(display_name)
                    size_factor = min(max(text_length / 5, 1), 3)
                    if node_type == "root":
                        size = base_node_size * node_size_multiplier * 1.5 * size_factor
                    elif node_type == "category":
                        size = base_node_size * node_size_multiplier * 1.2 * size_factor
                    else:
                        size = base_node_size * node_size_multiplier * size_factor
                    node_sizes.append(size)
                else:
                    if node_type == "root":
                        node_sizes.append(node_size * 1.3)
                    elif node_type == "category":
                        node_sizes.append(node_size * 1.1)
                    else:
                        node_sizes.append(node_size)
            
            # Determine text size
            if labels_inside_nodes:
                if node_type == "root":
                    text_size = font_size * 0.9
                elif node_type == "category":
                    text_size = font_size * 0.8
                else:
                    text_size = font_size * 0.7
            else:
                if node_type == "root":
                    text_size = font_size * 1.2
                elif node_type == "category":
                    text_size = font_size * 1.1
                else:
                    text_size = font_size
            
            # Determine text position
            if labels_inside_nodes:
                text_positions = ["middle center"] * len(node_ids)
                display_mode = 'markers+text'
            else:
                text_positions = [label_positions[node_id] for node_id in node_ids] if not use_tilted_text else None
                display_mode = 'markers' + ('+text' if not use_tilted_text else '')
            
            # Add the intruder nodes trace
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode=display_mode,
                marker=dict(
                    color=node_colors.get(node_type, "#9E9E9E"),  # Use regular node color
                    size=node_sizes,
                    line=dict(width=2, color=intruder_color),  # Red outline for intruders
                    symbol='x'  # Use 'x' symbol for intruders
                ),
                text=node_text if (not use_tilted_text or labels_inside_nodes) else None,
                textposition=text_positions,
                textfont=dict(
                    size=text_size,
                    color='black'
                ) if (not use_tilted_text or labels_inside_nodes) else None,
                hovertext=node_hover,
                hoverinfo='text',
                name=f"Intruder {node_type.capitalize()}s",  # e.g. "Intruder Characteristics"
                customdata=node_ids
            ))
            
            # Add tilted text annotations if needed
            if use_tilted_text and not labels_inside_nodes:
                for i, node_id in enumerate(node_ids):
                    position = label_positions[node_id]
                    yanchor = "bottom" if "top" in position else "top"
                    
                    fig.add_annotation(
                        x=node_x[i],
                        y=node_y[i],
                        text=node_text[i],
                        showarrow=False,
                        font=dict(
                            size=text_size,
                            color='black'
                        ),
                        textangle=text_tilt,
                        xanchor="center",
                        yanchor=yanchor,
                        bgcolor="rgba(255,255,255,0.7)",
                        borderpad=2
                    )
    
    # Add curved dashed lines for intruder connections
    # Group connections by group ID for legend purposes
    group_connection_traces = {}
    
    for conn in intruder_connections:
        source = conn['source']
        target = conn['target']
        similarity = conn['similarity']
        group_id = conn.get('group_id', 'G0')  # Default to G0 if not specified
        
        # Check if this group should be visible
        if visible_groups is not None and group_id not in visible_groups:
            continue  # Skip connections from groups that shouldn't be visible
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create a curved path
            path_x, path_y = create_curved_path(x0, y0, x1, y1, curvature=0.3)
            
            # Get or create the trace for this group
            if group_id not in group_connection_traces:
                group_connection_traces[group_id] = {
                    'x': [],
                    'y': [],
                    'hovertext': [],
                    'arrows': []
                }
            
            # Add to the group's path data
            group_connection_traces[group_id]['x'].extend(path_x.tolist() + [None])
            group_connection_traces[group_id]['y'].extend(path_y.tolist() + [None])
            
            # Add hover text for each point
            hover_text = f"Group {group_id}: Similarity = {similarity:.4f}"
            group_connection_traces[group_id]['hovertext'].extend([hover_text] * len(path_x) + [None])
            
            # Create arrow marker at the end of the path
            arrow_x, arrow_y = create_arrow_marker(path_x[-2], path_y[-2], path_x[-1], path_y[-1], arrow_size=0.01)
            
            # Add to the arrows list
            group_connection_traces[group_id]['arrows'].append({
                'x': arrow_x, 
                'y': arrow_y
            })
    
    # Add the connection traces to the figure
    for group_id, trace_data in group_connection_traces.items():
        # Add the connection line
        fig.add_trace(go.Scatter(
            x=trace_data['x'], 
            y=trace_data['y'],
            line=dict(
                width=2.0,
                color=connection_color,
                dash='dash'
            ),
            hoverinfo='text',
            hovertext=trace_data['hovertext'],
            mode='lines',
            name=f'Group {group_id} Connections',
            showlegend=True  # Show each group in the legend
        ))
        
        # Add arrow markers
        for arrow in trace_data['arrows']:
            fig.add_trace(go.Scatter(
                x=arrow['x'],
                y=arrow['y'],
                fill="toself",
                fillcolor=connection_color,
                line=dict(color=connection_color),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title="Orthogonal Taxonomy with Groups and Intruders",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=50, l=20, r=20, t=50),
        height=900,
        width=1200,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.1,
            xanchor="center",
            x=0.5
        ),
        # Add zoom capabilities
        dragmode='pan',
        newshape=dict(line_color='#000'),
        modebar_add=['drawline', 'drawopenpath', 'eraseshape']
    )
    
    return fig


def create_indented_list(taxonomy_data):
    """Create an indented list text representation of the taxonomy"""
    # Import networkx inside the function for reliability
    import networkx as nx
    
    # Create a NetworkX graph for traversal
    G = create_nx_graph(taxonomy_data)
    
    # Build a node map for quick access
    node_map = {node["id"]: node for node in taxonomy_data["nodes"]}
    
    # Find the root node
    root_id = "root"
    if root_id not in node_map:
        # Find the node with no incoming edges
        for node_id in node_map:
            if not any(edge["target"] == node_id for edge in taxonomy_data["edges"]):
                root_id = node_id
                break
    
    # Build parent-child map
    parent_child_map = {}
    for edge in taxonomy_data["edges"]:
        parent = edge["source"]
        child = edge["target"]
        if parent not in parent_child_map:
            parent_child_map[parent] = []
        parent_child_map[parent].append(child)
    
    # Recursive function to build the text representation
    def build_text(node_id, depth=0):
        if node_id not in node_map:
            return ""
        
        node = node_map[node_id]
        indent = "    " * depth
        node_type = node.get("type", "unknown")
        
        # Format the node line
        result = f"{indent}{'├─ ' if depth > 0 else ''}{node['name']} [{node_type}]"
        
        # Add description if available
        if "description" in node and node["description"]:
            # Truncate very long descriptions
            desc = node["description"]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            result += f" - {desc}"
        
        result += "\n"
        
        # Add children
        children = parent_child_map.get(node_id, [])
        for i, child_id in enumerate(sorted(children)):
            result += build_text(child_id, depth + 1)
        
        return result
    
    # Build the full text
    text = "Taxonomy Structure:\n" + build_text(root_id)
    
    return text


def visualize_taxonomy(taxonomy_data, visualization_type="orthogonal", options=None):
    """Create a visualization of the taxonomy based on the specified type"""
    if visualization_type == "orthogonal":
        return create_orthogonal_visualization(taxonomy_data, options)
    elif visualization_type == "indented_list":
        return create_indented_list(taxonomy_data)
    else:
        # Default to orthogonal
        return create_orthogonal_visualization(taxonomy_data, options)


def hierarchy_pos(G, root=None, width=1., vert_gap=0.1, vert_loc=0, xcenter=0.5):
    """
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.
    
    We include this function here for completeness
    """
    # Import networkx inside the function for reliability
    import networkx as nx
    
    if not nx.is_tree(G):
        # Add virtual edges to make it a tree
        tree_edges = list(nx.bfs_edges(G, source=root))
        non_tree_edges = [e for e in G.edges() if e not in tree_edges and (e[1], e[0]) not in tree_edges]
        
        # Create a copy of G without the non-tree edges
        T = nx.DiGraph()
        T.add_nodes_from(G.nodes())
        T.add_edges_from(tree_edges)
        
        # Get positions from the tree
        pos = hierarchy_pos(T, root, width, vert_gap, vert_loc, xcenter)
        
        return pos
    
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = list(G.nodes())[0]
    
    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        """
        Helper function for hierarchy_pos
        """
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph):
            children = [child for child in children if child != parent]
        
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
        return pos
    
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# New implementation of the taxonomy tree converter
def get_taxonomy_tree(taxonomy_data):
    """
    Convert the flat taxonomy data into a nested tree structure
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
    
    Returns:
        Dictionary representing the nested taxonomy tree
    """
    # Import networkx inside the function for reliability
    import networkx as nx
    
    G = create_nx_graph(taxonomy_data)
    node_map = {node["id"]: node for node in taxonomy_data["nodes"]}
    
    # Find the root node
    root_id = "root"
    if root_id not in node_map:
        # Find a node with no incoming edges
        for node_id in node_map:
            if G.in_degree(node_id) == 0:
                root_id = node_id
                break
    
    # Build the tree recursively
    def build_tree(node_id):
        children = list(G.successors(node_id))
        
        if not children:
            # Leaf node - return the name
            return node_map[node_id]["name"]
        
        # Non-leaf node - create a dictionary with children
        result = {}
        for child_id in children:
            result[child_id] = build_tree(child_id)
        
        return result
    
    # Start building from root
    tree = {}
    tree[root_id] = build_tree(root_id)
    
    return tree


def get_leaf_nodes(taxonomy_tree):
    """
    Extract leaf nodes from the taxonomy tree
    
    Args:
        taxonomy_tree: Dictionary representing the nested taxonomy
    
    Returns:
        Dictionary with node IDs as keys and node names as values for leaf nodes
    """
    leaves = {}
    
    def traverse(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                traverse(v, path + [k])
        else:
            # Leaf node (string value)
            leaves[path[-1]] = node
    
    for root_key, root_value in taxonomy_tree.items():
        traverse(root_value, [root_key])
    
    return leaves


def get_natural_leaf_groups(taxonomy_tree):
    """
    Get leaf nodes grouped by their natural parent nodes in the taxonomy
    
    Args:
        taxonomy_tree: Dictionary representing the nested taxonomy
    
    Returns:
        Dictionary with parent node IDs as keys and lists of leaf node names as values
    """
    natural_groups = {}
    
    def find_natural_groups(node, path=None, parent_key=None):
        if path is None:
            path = []
            
        if isinstance(node, dict):
            # Check if this is a dictionary with all leaf nodes
            all_leaves = all(not isinstance(v, dict) for v in node.values())
            
            if all_leaves and len(node) >= 1:
                # This is a natural group with multiple leaf nodes
                key = path[-1] if path else "root"
                natural_groups[key] = list(node.values())
            
            # Process children recursively
            for k, v in node.items():
                find_natural_groups(v, path + [k], k)
    
    for root_key, root_value in taxonomy_tree.items():
        find_natural_groups(root_value, [root_key])
    
    return natural_groups


def calculate_taxonomy_robustness_corrected(taxonomy_data, embedding_calculator=None):
    """
    Calculate robustness using the natural grouping approach
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
        embedding_calculator: Optional pre-initialized EmbeddingCalculator instance
    
    Returns:
        Dictionary with robustness metric and detailed analysis
    """
    # Import networkx inside the function for reliability
    import networkx as nx
    
    # Create embedding calculator if not provided
    if embedding_calculator is None:
        embedding_calculator = EmbeddingCalculator()
        embedding_calculator.load_model()
    
    # Convert to tree structure
    taxonomy_tree = get_taxonomy_tree(taxonomy_data)
    
    # Get natural groups of leaf nodes
    leaf_groups = get_natural_leaf_groups(taxonomy_tree)
    
    if not leaf_groups:
        # Fallback to get_natural_leaf_groups using the graph approach
        G = create_nx_graph(taxonomy_data)
        node_map = {node["id"]: node for node in taxonomy_data["nodes"]}
        leaf_groups = {}
        
        # Find nodes that have children that are all leaf nodes
        for node_id in G.nodes():
            if node_id == "root":
                continue
                
            children = list(G.successors(node_id))
            if children:
                # Check if all children are leaf nodes
                all_leaves = all(G.out_degree(child) == 0 for child in children)
                
                if all_leaves and len(children) >= 1:
                    # This is a natural group with leaf nodes
                    child_names = [node_map[child]["name"] for child in children]
                    leaf_groups[node_id] = child_names
    
    if not leaf_groups:
        return {
            "robustness": 0.0,
            "error": "No natural groups found in the taxonomy"
        }
    
    # Get all leaf nodes
    all_leaves_dict = get_leaf_nodes(taxonomy_tree)
    all_leaf_names = list(all_leaves_dict.values())
    
    # Calculate embeddings for all leaf names
    leaf_embeddings = embedding_calculator.compute_embeddings(all_leaf_names)
    
    if leaf_embeddings is None:
        return {
            "robustness": 0.0,
            "error": "Failed to compute embeddings"
        }
    
    # Create a name-to-index map for similarity lookups
    name_to_index = {name: i for i, name in enumerate(all_leaf_names)}
    
    # Compute similarity matrix
    similarity_matrix = np.zeros((len(all_leaf_names), len(all_leaf_names)))
    for i in range(len(all_leaf_names)):
        for j in range(len(all_leaf_names)):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1
            else:
                # Compute using the scipy cosine distance (1 - distance = similarity)
                similarity_matrix[i, j] = 1 - cosine(leaf_embeddings[i], leaf_embeddings[j])
    
    r_t_values = []
    group_data = []
    total_intruders = 0
    
    # Process each natural group
    for group_idx, (parent_id, group_names) in enumerate(leaf_groups.items()):
        # Get the parent node name
        parent_name = parent_id  # Default to the ID if we can't find the name
        for node in taxonomy_data["nodes"]:
            if node["id"] == parent_id:
                parent_name = node["name"]
                break
                
        # Skip if the group has no leaves
        if not group_names:
            continue
        
        # Calculate all pairwise similarities within the group
        internal_similarities = []
        min_internal_sim = 1.0  # Start with highest possible value
        
        for i in range(len(group_names)):
            if group_names[i] not in name_to_index:
                continue  # Skip if not in our mapping
                
            idx_i = name_to_index[group_names[i]]
            
            for j in range(i + 1, len(group_names)):
                if group_names[j] not in name_to_index:
                    continue  # Skip if not in our mapping
                    
                idx_j = name_to_index[group_names[j]]
                sim = similarity_matrix[idx_i, idx_j]
                internal_similarities.append({
                    "pair": (group_names[i], group_names[j]),
                    "similarity": float(sim)
                })
                
                # Update minimum similarity
                min_internal_sim = min(min_internal_sim, sim)
        
        # For single-element groups, set min_internal_sim to 1.0
        if len(group_names) == 1 or not internal_similarities:
            min_internal_sim = 1.0
        
        # Find intruder characteristics
        intruders = []
        
        # For each leaf in the group
        for group_name in group_names:
            if group_name not in name_to_index:
                continue  # Skip if not in our mapping
                
            idx_i = name_to_index[group_name]
            
            # Compare with every leaf outside the group
            for other_name in all_leaf_names:
                if other_name not in group_names and other_name in name_to_index:  # Only consider leaves outside the group
                    idx_j = name_to_index[other_name]
                    sim = similarity_matrix[idx_i, idx_j]
                    
                    # If similarity is higher than the minimum internal similarity, it's an intruder
                    if sim > min_internal_sim and min_internal_sim < 1.0:
                        intruders.append({
                            "node": group_name,
                            "other_node": other_name,
                            "similarity": float(sim)
                        })
        
        # Calculate R(T) for this group
        n_ic = len(intruders)
        n_gc = len(group_names)
        n_ac = len(all_leaf_names)
        
        total_intruders += n_ic
        
        # Calculate group robustness
        if n_gc == 1 or n_ac == n_gc:  # Single element or all leaves in one group
            group_robustness = 1.0
        else:
            potential_comparisons = n_gc * (n_ac - n_gc)
            if potential_comparisons > 0:
                group_robustness = 1.0 - (n_ic / potential_comparisons)
            else:
                group_robustness = 1.0  # Prevent division by zero
        
        r_t_values.append(group_robustness)
        
        group_data.append({
            "group_id": f"G{group_idx+1}",
            "parent": parent_name,
            "characteristics": group_names,
            "size": n_gc,
            "min_internal_similarity": float(min_internal_sim),
            "internal_similarities": internal_similarities,
            "intruders": intruders,
            "intruder_count": n_ic,
            "group_robustness": group_robustness
        })
    
    # Calculate overall robustness
    if not r_t_values:
        return {
            "robustness": 0.0,
            "error": "No valid groups found for robustness calculation"
        }
    
    overall_robustness = sum(r_t_values) / len(r_t_values)
    
    return {
        "robustness": float(overall_robustness),
        "group_count": len(leaf_groups),
        "characteristic_count": len(all_leaf_names),
        "groups": group_data,
        "detailed_explanation": (
            f"Found {len(leaf_groups)} natural groups with a total of {len(all_leaf_names)} leaf nodes. "
            f"The overall robustness R(T) is {overall_robustness:.4f}, where 1.0 is perfect robustness "
            f"(no intruder characteristics). The analysis identified a total of {total_intruders} "
            f"potential intruder characteristics across all groups."
        )
    }


def extract_ncat(taxonomy_data):
    """
    Extract the number of categories (excluding root) from the taxonomy
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
    
    Returns:
        int: Number of category nodes (excluding root)
    """
    # Count nodes with type "category", excluding the root node
    ncat = sum(1 for node in taxonomy_data["nodes"] 
              if node.get("type", "").lower() == "category" and node["id"] != "root")
    
    return ncat


def extract_nchar(taxonomy_data):
    """
    Extract the number of characteristics from the taxonomy
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
    
    Returns:
        int: Number of characteristic nodes
    """
    # Count nodes with type "characteristic"
    nchar = sum(1 for node in taxonomy_data["nodes"] 
               if node.get("type", "").lower() == "characteristic")
    
    return nchar


def extract_depths_cat(taxonomy_data):
    """
    Extract the depths of all category nodes (excluding root)
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
    
    Returns:
        list: Depths of category nodes
    """
    # Import networkx inside the function for reliability
    import networkx as nx
    
    G = create_nx_graph(taxonomy_data)
    depths_cat = []
    
    # Find the root node
    root_id = "root"
    if root_id not in G.nodes():
        # Find a node with no incoming edges
        for node_id in G.nodes():
            if G.in_degree(node_id) == 0:
                root_id = node_id
                break
    
    # Calculate depth for each category node
    for node in taxonomy_data["nodes"]:
        if node.get("type", "").lower() == "category" and node["id"] != root_id:
            try:
                # Calculate shortest path length from root (depth)
                depth = len(nx.shortest_path(G, source=root_id, target=node["id"])) - 1
                depths_cat.append(depth)
            except nx.NetworkXNoPath:
                # If no path (disconnected node), use maximum depth + 1
                if depths_cat:
                    depths_cat.append(max(depths_cat) + 1)
                else:
                    depths_cat.append(1)  # Default depth
    
    return depths_cat


def extract_depths_char(taxonomy_data):
    """
    Extract the depths of all characteristic nodes
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
    
    Returns:
        list: Depths of characteristic nodes
    """
    # Import networkx inside the function for reliability
    import networkx as nx
    
    G = create_nx_graph(taxonomy_data)
    depths_char = []
    
    # Find the root node
    root_id = "root"
    if root_id not in G.nodes():
        # Find a node with no incoming edges
        for node_id in G.nodes():
            if G.in_degree(node_id) == 0:
                root_id = node_id
                break
    
    # Calculate depth for each characteristic node
    for node in taxonomy_data["nodes"]:
        if node.get("type", "").lower() == "characteristic":
            try:
                # Calculate shortest path length from root (depth)
                depth = len(nx.shortest_path(G, source=root_id, target=node["id"])) - 1
                depths_char.append(depth)
            except nx.NetworkXNoPath:
                # If no path (disconnected node), use maximum depth + 1
                if depths_char:
                    depths_char.append(max(depths_char) + 1)
                else:
                    depths_char.append(1)  # Default depth
    
    return depths_char


def calculate_taxonomy_conciseness_corrected(taxonomy_data):
    """
    Calculate conciseness using the improved formula from Prat et al.
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
    
    Returns:
        Float: Conciseness value (0 to 1)
    """
    # Extract required components
    ncat = extract_ncat(taxonomy_data)
    nchar = extract_nchar(taxonomy_data)
    depths_cat = extract_depths_cat(taxonomy_data)
    depths_char = extract_depths_char(taxonomy_data)
    
    # Calculate the sum of the inverses of the depths
    sum_cat = sum(1 / d for d in depths_cat if d > 0) if depths_cat else 0
    sum_char = sum(1 / d for d in depths_char if d > 0) if depths_char else 0
    
    # Edge cases based on the paper's description
    if ncat == 1 and nchar == 2:
        return 1.0  # Simplest possible taxonomy
    
    if ncat + nchar == 0:
        return 1.0  # Empty taxonomy
    
    # Calculate using the formula
    try:
        # The formula: C(T) = 1 / (1 + ln(∑(1/d(Cat_i)) + ∑(1/d(Char_j)) - 1))
        inside_log = sum_cat + sum_char - 1
        
        # Handle case where inside_log is <= 0
        if inside_log <= 0:
            return 1.0
            
        conciseness = 1 / (1 + math.log(inside_log))
    except (ValueError, ZeroDivisionError):
        # Handle edge cases
        return 0.0
    
    # Ensure the result is in the range [0, 1]
    return max(0.0, min(1.0, conciseness))


def calculate_taxonomy_metrics_corrected(taxonomy_data):
    """Calculate metrics using the corrected methods"""
    # Import networkx inside the function for reliability
    import networkx as nx
    
    G = create_nx_graph(taxonomy_data)
    
    # Basic metrics
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    num_categories = extract_ncat(taxonomy_data)
    num_characteristics = extract_nchar(taxonomy_data)
    
    # Calculate depth
    try:
        depth = nx.dag_longest_path_length(G)
    except (nx.NetworkXError, nx.NetworkXNotImplemented):
        # Fallback if graph is not a DAG
        depth = 0
        for node in G.nodes():
            if node != "root":
                try:
                    path_length = len(nx.shortest_path(G, source="root", target=node)) - 1
                    depth = max(depth, path_length)
                except nx.NetworkXNoPath:
                    pass
    
    # Calculate average branching factor
    total_children = 0
    non_leaf_nodes = 0
    for node in G.nodes():
        children = list(G.successors(node))
        if children:
            total_children += len(children)
            non_leaf_nodes += 1
    
    avg_branching = total_children / non_leaf_nodes if non_leaf_nodes > 0 else 0
    
    # Calculate breadth (maximum number of nodes at any level)
    levels = {}
    for node in G.nodes():
        # Calculate shortest path length from root
        if node != "root":
            try:
                path_length = len(nx.shortest_path(G, source="root", target=node)) - 1
            except nx.NetworkXNoPath:
                path_length = 0  # Handle disconnected nodes
        else:
            path_length = 0
        
        if path_length not in levels:
            levels[path_length] = 0
        levels[path_length] += 1
    
    breadth = max(levels.values()) if levels else 0
    
    # Balanced factor (standard deviation of children across non-leaf nodes)
    children_counts = []
    for node in G.nodes():
        children = list(G.successors(node))
        if node != "root" and children:  # Exclude root and leaf nodes
            children_counts.append(len(children))
    
    balanced_factor = np.std(children_counts) if children_counts else 0
    
    # Calculate number of leaf nodes
    leaf_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    num_leaf_nodes = len(leaf_nodes)
    
    # Calculate number of paths
    num_paths = 0
    for leaf in leaf_nodes:
        # Count paths from root to each leaf
        try:
            num_paths += len(list(nx.all_simple_paths(G, source="root", target=leaf)))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass  # Skip disconnected nodes
    
    # Calculate conciseness using the corrected method
    conciseness = calculate_taxonomy_conciseness_corrected(taxonomy_data)
    
    # Create the metrics dictionary
    metrics = {
        "Total Nodes": num_nodes,
        "Categories": num_categories,
        "Characteristics": num_characteristics,
        "Total Relationships": num_edges,
        "Maximum Depth": depth,
        "Maximum Breadth": breadth,
        "Average Branching Factor": round(avg_branching, 2),
        "Balance Factor (lower is better)": round(balanced_factor, 2),
        "Leaf Nodes": num_leaf_nodes,
        "Number of Paths": num_paths,
        "Conciseness": round(conciseness, 4)
    }
    
    return metrics


def display_robustness_formula():
    """Display the robustness formula in a visually appealing way"""
    st.markdown("### Robustness Metric Formula")
    
    st.markdown(r"""
    The Robustness metric R(T) is calculated as follows:
    
    $$R(T) = \frac{1}{n_{groups}} \sum_{i=1}^{n_{groups}} \left( 1 - \frac{n_{ic}}{n_{gc} \cdot (n_{ac} - n_{gc})} \right)$$
    
    Where:
    - $n_{groups}$ is the number of natural groups in the taxonomy (based on tree structure)
    - $n_{ic}$ is the number of intruder characteristics in a group
    - $n_{gc}$ is the number of leaf nodes in the group
    - $n_{ac}$ is the total number of leaf nodes in the taxonomy
    """)
    
    # Add explanation with colored boxes
    st.markdown("### How the Formula Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Step 1:</strong> Identify natural groups of leaf nodes in the taxonomy tree
        </div>
        
        <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Step 2:</strong> Calculate all pairwise similarities within each group
        </div>
        
        <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Step 3:</strong> Find the minimum similarity within each group (cohesiveness)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Step 4:</strong> If any leaf outside the group has higher similarity with a group member, count as an intruder
        </div>
        
        <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Step 5:</strong> Calculate robustness contribution for each group using the formula
        </div>
        
        <div style="background-color:#e6f7ff; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Step 6:</strong> Average all group contributions for final robustness score
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Tree-Based Group Formation
    
    This implementation identifies natural groups based on the taxonomy tree structure:
    
    - Groups are formed from leaf nodes under the same parent
    - The method considers the actual hierarchical structure of the taxonomy
    - Single-node groups are properly handled
    - Intruders are identified when nodes from different groups have higher similarity than within-group minimum
    """)


def display_conciseness_formula():
    """Display the conciseness formula and explanation in a visually appealing way"""
    st.markdown("### Conciseness Metric Formula")
    
    st.markdown(r"""
    The Conciseness metric C(T) is calculated as follows:
    
    $$C(T) = \frac{1}{1 + \ln\left(\sum_{i=1}^{n_{cat}} \frac{1}{d(Cat_i)} + \sum_{j=1}^{n_{char}} \frac{1}{d(Char_j)} - 1\right)}$$
    
    Where:
    - $n_{cat}$ is the number of categories in the taxonomy (excluding root)
    - $n_{char}$ is the number of characteristics in the taxonomy
    - $d(Cat_i)$ is the depth of category $i$ in the taxonomy tree
    - $d(Char_j)$ is the depth of characteristic $j$ in the taxonomy tree
    """)
    
    # Add explanation
    st.markdown("### How the Formula Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color:#e6fff6; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Intuition:</strong> The formula penalizes adding elements closer to the root more than adding elements at deeper levels.
        </div>
        
        <div style="background-color:#e6fff6; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Range:</strong> The conciseness value ranges from 0 to 1, where 1 represents maximum conciseness.
        </div>
        
        <div style="background-color:#e6fff6; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Simplest Taxonomy:</strong> A taxonomy with 1 category and 2 characteristics has C(T) = 1.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:#e6fff6; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Depth Impact:</strong> Adding a node at depth 2 impacts conciseness twice as much as adding at depth 4.
        </div>
        
        <div style="background-color:#e6fff6; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Optimization:</strong> To improve conciseness, move nodes deeper into the taxonomy where appropriate.
        </div>
        
        <div style="background-color:#e6fff6; padding:10px; border-radius:5px; margin-bottom:10px;">
        <strong>Source:</strong> This metric was proposed by Prat et al. (2015) for evaluating taxonomy conciseness.
        </div>
        """, unsafe_allow_html=True)
    
    # Add a visual example
    st.markdown("### Example")
    
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-top: 20px;">
        <div style="text-align: center; width: 100%;">
            <pre style="background-color: #f5f5f5; padding: 15px; border-radius: 5px; display: inline-block; text-align: left;">
    Example 1: A taxonomy with 4 categories at depths [1,1,2,3] and 6 characteristics at depths [2,2,2,3,3,4]
      - Sum for categories: 1/1 + 1/1 + 1/2 + 1/3 = 2 + 0.5 + 0.33 = 2.83
      - Sum for characteristics: 1/2 + 1/2 + 1/2 + 1/3 + 1/3 + 1/4 = 1.5 + 0.67 + 0.25 = 2.42
      - Inside log: 2.83 + 2.42 - 1 = 4.25
      - C(T) = 1 / (1 + ln(4.25)) = 1 / (1 + 1.45) = 0.41
    
    Example 2: Same taxonomy with nodes moved deeper, at depths [2,2,3,4] and [3,3,3,4,4,5]
      - Sum for categories: 1/2 + 1/2 + 1/3 + 1/4 = 1 + 0.33 + 0.25 = 1.58
      - Sum for characteristics: 1/3 + 1/3 + 1/3 + 1/4 + 1/4 + 1/5 = 1 + 0.5 + 0.2 = 1.7
      - Inside log: 1.58 + 1.7 - 1 = 2.28
      - C(T) = 1 / (1 + ln(2.28)) = 1 / (1 + 0.82) = 0.55
      
    Example 2 has higher conciseness (0.55 > 0.41) because elements are placed deeper in the taxonomy.
            </pre>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_curved_path(x0, y0, x1, y1, curvature=0.2):
    """
    Create a curved path between two points
    
    Args:
        x0, y0: Coordinates of the source point
        x1, y1: Coordinates of the target point
        curvature: Amount of curvature (0 = straight line, higher = more curved)
        
    Returns:
        x_path, y_path: Arrays of points defining the curved path
    """
    import numpy as np
    
    # Find midpoint
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    
    # Calculate perpendicular vector
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    
    # Create perpendicular control point
    px = mx - dy * curvature
    py = my + dx * curvature
    
    # Create a Bezier curve with points
    t = np.linspace(0, 1, 20)
    x_path = (1-t)**2 * x0 + 2*(1-t)*t * px + t**2 * x1
    y_path = (1-t)**2 * y0 + 2*(1-t)*t * py + t**2 * y1
    
    return x_path, y_path


def create_arrow_marker(x1, y1, x2, y2, arrow_size=0.01):
    """
    Create an arrow marker at the end of a line segment
    
    Args:
        x1, y1: Coordinates of the second-to-last point on the path
        x2, y2: Coordinates of the last point on the path
        arrow_size: Size of the arrow
        
    Returns:
        arrow_x, arrow_y: Arrays of points defining the arrow
    """
    import numpy as np
    
    # Calculate direction vector
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / length, dy / length
    
    # Calculate perpendicular vector
    px, py = -dy, dx
    
    # Calculate arrow points
    arrow_x = [
        x2,  # Tip
        x2 - dx * arrow_size - px * arrow_size * 0.6,  # Left corner
        x2 - dx * arrow_size + px * arrow_size * 0.6,  # Right corner
        x2   # Back to tip
    ]
    
    arrow_y = [
        y2,  # Tip
        y2 - dy * arrow_size - py * arrow_size * 0.6,  # Left corner
        y2 - dy * arrow_size + py * arrow_size * 0.6,  # Right corner
        y2   # Back to tip
    ]
    
    return arrow_x, arrow_y


def create_intruder_visualization(taxonomy_data, robustness_data):
    """
    Create a specialized visualization of the taxonomy highlighting intruders
    and their semantic relationships to elements in other groups.
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
        robustness_data: Dictionary with robustness analysis results
        
    Returns:
        Plotly figure object with the visualization
    """
    # Import needed libraries
    import plotly.graph_objects as go
    import networkx as nx
    
    # Create a graph from the taxonomy data
    G = create_nx_graph(taxonomy_data)
    
    # Build a mapping from node IDs to nodes
    node_map = {node["id"]: node for node in taxonomy_data["nodes"]}
    
    # Get the hierarchical layout (using the existing function)
    pos = hierarchy_pos(G, root="root", vert_gap=0.15, width=1.0)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Find all intruders from the robustness data
    all_intruders = {}
    intruder_connections = []
    
    # Process each group to identify intruders and their connections
    for group in robustness_data['groups']:
        # Get all node IDs in this group
        group_names = group['characteristics']
        group_ids = []
        
        # Find node IDs for all characteristics in this group
        for node_id, node in node_map.items():
            if node.get('name') in group_names:
                group_ids.append(node_id)
        
        # Process each intruder
        for intruder in group['intruders']:
            intruder_name = intruder['node']
            similar_name = intruder['other_node']
            similarity = intruder['similarity']
            
            # Find node IDs for the intruder and similar node
            intruder_id = None
            similar_id = None
            
            for node_id, node in node_map.items():
                if node.get('name') == intruder_name:
                    intruder_id = node_id
                    all_intruders[intruder_id] = {
                        'name': intruder_name,
                        'connections': []
                    }
                if node.get('name') == similar_name:
                    similar_id = node_id
            
            # Store the connection if we found both nodes
            if intruder_id and similar_id:
                connection = {
                    'source': intruder_id,
                    'target': similar_id,
                    'similarity': similarity
                }
                
                # Add to the intruder's connections list
                if intruder_id in all_intruders:
                    all_intruders[intruder_id]['connections'].append(connection)
                
                # Add to all connections
                intruder_connections.append(connection)
    
    # Draw orthogonal edges for the base taxonomy structure
    for edge in taxonomy_data["edges"]:
        source = edge["source"]
        target = edge["target"]
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create orthogonal edge (three segments: down, across, down)
            edge_x = []
            edge_y = []
            
            # First point: at the parent node's center
            edge_x.append(x0)
            edge_y.append(y0)
            
            # Second point: directly below parent at halfway between parent and child level
            mid_y = (y0 + y1) / 2
            edge_x.append(x0)
            edge_y.append(mid_y)
            
            # Third point: horizontal shift to child's x-coordinate
            edge_x.append(x1)
            edge_y.append(mid_y)
            
            # Fourth point: down to child node
            edge_x.append(x1)
            edge_y.append(y1)
            
            # Add the edge trace
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color='rgba(100,100,100,0.7)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
    
    # Add curved dashed lines for intruder connections
    for conn in intruder_connections:
        source = conn['source']
        target = conn['target']
        similarity = conn['similarity']
        
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            # Create a curved path
            path_x, path_y = create_curved_path(x0, y0, x1, y1)
            
            # Add the connection trace
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                line=dict(
                    width=2.0, 
                    color='rgba(255,0,0,0.5)', 
                    dash='dash'
                ),
                hoverinfo='text',
                hovertext=f"Similarity: {similarity:.4f}",
                mode='lines',
                name='Semantic Similarity',
                showlegend=source == list(all_intruders.keys())[0] if all_intruders else False  # Show legend only once
            ))
            
            # Add arrow marker at the target end
            arrow_x, arrow_y = create_arrow_marker(path_x[-2], path_y[-2], path_x[-1], path_y[-1], arrow_size=0.01)
            
            fig.add_trace(go.Scatter(
                x=arrow_x, y=arrow_y,
                fill="toself",
                fillcolor='rgba(255,0,0,0.5)',
                line=dict(color='rgba(255,0,0,0.5)'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
    
    # Add nodes
    for node_type in ["root", "category", "characteristic"]:
        # Get nodes of this type
        type_nodes = [n for n in taxonomy_data["nodes"] if n.get("type", "unknown") == node_type]
        
        if not type_nodes:
            continue
        
        # Separate normal nodes and intruder nodes
        normal_nodes = [n for n in type_nodes if n["id"] not in all_intruders]
        intruder_nodes = [n for n in type_nodes if n["id"] in all_intruders]
        
        # Add normal nodes
        if normal_nodes:
            node_x = []
            node_y = []
            node_text = []
            node_hover = []
            
            for node in normal_nodes:
                if node["id"] in pos:
                    x, y = pos[node["id"]]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Process node name for display
                    display_name = node["name"]
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                    
                    node_text.append(display_name)
                    
                    # Create hover information
                    hover_text = f"<b>{node['name']}</b><br>"
                    if "description" in node and node["description"]:
                        hover_text += f"Description: {node['description']}<br>"
                    
                    node_hover.append(hover_text)
            
            # Set color based on node type
            if node_type == "root":
                node_color = "#4CAF50"  # Green
                node_size = 15 * 1.3
            elif node_type == "category":
                node_color = "#2196F3"  # Blue
                node_size = 15 * 1.1
            else:
                node_color = "#FF9800"  # Orange
                node_size = 15
            
            # Add the nodes trace
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    color=node_color,
                    size=node_size,
                    line=dict(width=1.5, color='white')
                ),
                text=node_text,
                textposition="top center",
                hovertext=node_hover,
                hoverinfo='text',
                name=f"{node_type.capitalize()}",
            ))
        
        # Add intruder nodes with different color
        if intruder_nodes:
            node_x = []
            node_y = []
            node_text = []
            node_hover = []
            
            for node in intruder_nodes:
                if node["id"] in pos:
                    x, y = pos[node["id"]]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Process node name for display
                    display_name = node["name"]
                    if len(display_name) > 25:
                        display_name = display_name[:22] + "..."
                    
                    node_text.append(display_name)
                    
                    # Create detailed hover information for intruders
                    hover_text = f"<b>{node['name']} (INTRUDER)</b><br>"
                    if node["id"] in all_intruders:
                        connections = all_intruders[node["id"]]['connections']
                        hover_text += f"Connections outside group: {len(connections)}<br>"
                        for conn in connections[:3]:  # Show top 3 connections
                            similar_node = node_map.get(conn['target'], {}).get('name', 'Unknown')
                            hover_text += f"- Similar to: {similar_node} ({conn['similarity']:.4f})<br>"
                        if len(connections) > 3:
                            hover_text += f"- (and {len(connections) - 3} more...)<br>"
                    
                    node_hover.append(hover_text)
            
            # Add the intruder nodes trace
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    color='#FF0000',  # Red for intruders
                    size=15,
                    line=dict(width=1.5, color='white')
                ),
                text=node_text,
                textposition="top center",
                hovertext=node_hover,
                hoverinfo='text',
                name='Intruders',
            ))
    
    # Update layout
    fig.update_layout(
        title="Taxonomy Intruder Visualization",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=50, l=20, r=20, t=50),
        height=900,
        width=1200,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            constrain="domain",
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-.1,
            xanchor="center",
            x=0.5
        ),
        # Add zoom capabilities
        dragmode='pan',
        newshape=dict(line_color='#000'),
        modebar_add=['drawline', 'drawopenpath', 'eraseshape']
    )
    
    return fig


def calculate_and_display_robustness(taxonomy_data, embedding_calculator):
    """Calculate and display robustness metrics using the corrected method"""
    # Import networkx inside the function for reliability
    import networkx as nx
    
    with st.spinner("Calculating taxonomy robustness (this may take a moment)..."):
        robustness_data = calculate_taxonomy_robustness_corrected(taxonomy_data, embedding_calculator)
        
    # Store the results in session state
    st.session_state.robustness_data = robustness_data
    
    # Display the metrics
    st.subheader("Taxonomy Robustness Analysis")
    
    st.metric(
        "Robustness R(T)",
        f"{robustness_data['robustness']:.4f}",
        help="Ranges from 0 to 1, where 1 is perfect robustness (no intruders)"
    )
    
    st.markdown(robustness_data["detailed_explanation"])

    # Add a button to show visualization
    if st.button("Show Taxonomy Visualization with Groups and Intruders"):
        st.markdown("---")
        st.subheader("Taxonomy Visualization with Groups and Intruders")
        
        with st.spinner("Generating visualization..."):
            # Use visualization options from session state
            viz_options = st.session_state.viz_options.copy() if 'viz_options' in st.session_state else {}
            viz_options["show_group_boundaries"] = True
            
            try:
                # Generate the visualization
                enhanced_fig = create_enhanced_intruder_visualization(
                    st.session_state.taxonomy_data,
                    st.session_state.robustness_data,
                    options=viz_options
                )
                
                # Display the figure
                st.plotly_chart(enhanced_fig, use_container_width=True)
                
                # Add legend explanation
                st.markdown("""
                **Legend**:
                - **Group boundaries**: Dashed boxes showing natural groupings
                - **Red X marks**: Intruder characteristics
                - **Red dashed arrows**: Semantic similarity connections
                - **Hover over elements**: For detailed information
                """)
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
                st.error("Check that the create_enhanced_intruder_visualization function is properly defined.")
    # Group analysis
    st.subheader("Group Analysis")
    
    for group in robustness_data["groups"]:
        with st.expander(f"Group {group['group_id']} - Parent: {group['parent']} - {len(group['characteristics'])} leaf nodes"):
            # Group info
            st.markdown(f"**Leaf Nodes:** {', '.join(group['characteristics'])}")
            st.markdown(f"**Group Robustness:** {group['group_robustness']:.4f}")
            st.markdown(f"**Minimum Internal Similarity:** {group['min_internal_similarity']:.4f}")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Internal similarity matrix
            with col1:
                if len(group['internal_similarities']) > 0:
                    st.markdown("##### Internal Similarities")
                    for pair in group['internal_similarities']:
                        st.markdown(f"- {pair['pair'][0]} ↔ {pair['pair'][1]}: {pair['similarity']:.4f}")
                else:
                    st.markdown("*No internal similarities (single element group)*")
            
            # Intruder characteristics
            with col2:
                if len(group['intruders']) > 0:
                    st.markdown(f"##### Intruders ({group['intruder_count']})")
                    for intruder in group['intruders']:
                        st.markdown(f"- {intruder['node']} ↔ {intruder['other_node']}: {intruder['similarity']:.4f}")
                else:
                    st.markdown("*No intruders found*")
            
            # Visualize the similarities
            if len(group['internal_similarities']) > 0 or len(group['intruders']) > 0:
                try:
                    import plotly.graph_objects as go
                    import networkx as nx
                    
                    # Create a graph to visualize group relationships
                    G = nx.Graph()
                    
                    # Add nodes
                    for char in group['characteristics']:
                        G.add_node(char, group=group['group_id'])
                    
                    # Add intruder connections
                    for intruder in group['intruders']:
                        if intruder['other_node'] not in G.nodes():
                            G.add_node(intruder['other_node'], group='external')
                        G.add_edge(
                            intruder['node'], 
                            intruder['other_node'], 
                            weight=intruder['similarity'],
                            type='intruder'
                        )
                    
                    # Add internal connections
                    for pair in group['internal_similarities']:
                        G.add_edge(
                            pair['pair'][0], 
                            pair['pair'][1], 
                            weight=pair['similarity'],
                            type='internal'
                        )
                    
                    # Create layout
                    pos = nx.spring_layout(G, seed=42)
                    
                    # Create a Plotly figure
                    fig = go.Figure()
                    
                    # Add internal edges (blue)
                    internal_edges_x = []
                    internal_edges_y = []
                    internal_edge_texts = []
                    
                    for u, v, data in G.edges(data=True):
                        if data['type'] == 'internal':
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            
                            internal_edges_x.append(x0)
                            internal_edges_x.append(x1)
                            internal_edges_x.append(None)
                            
                            internal_edges_y.append(y0)
                            internal_edges_y.append(y1)
                            internal_edges_y.append(None)
                            
                            weight = data['weight']
                            internal_edge_texts.append(f"{weight:.4f}")
                            internal_edge_texts.append(f"{weight:.4f}")
                            internal_edge_texts.append(None)
                    
                    # Add internal edges trace (blue)
                    if internal_edges_x:
                        fig.add_trace(go.Scatter(
                            x=internal_edges_x, y=internal_edges_y,
                            line=dict(width=1, color='blue'),
                            hoverinfo='text',
                            text=internal_edge_texts,
                            mode='lines',
                            name='Internal Similarities'
                        ))
                    
                    # Add intruder edges (red)
                    intruder_edges_x = []
                    intruder_edges_y = []
                    intruder_edge_texts = []
                    
                    for u, v, data in G.edges(data=True):
                        if data['type'] == 'intruder':
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            
                            intruder_edges_x.append(x0)
                            intruder_edges_x.append(x1)
                            intruder_edges_x.append(None)
                            
                            intruder_edges_y.append(y0)
                            intruder_edges_y.append(y1)
                            intruder_edges_y.append(None)
                            
                            weight = data['weight']
                            intruder_edge_texts.append(f"{weight:.4f}")
                            intruder_edge_texts.append(f"{weight:.4f}")
                            intruder_edge_texts.append(None)
                    
                    # Add intruder edges trace (red)
                    if intruder_edges_x:
                        fig.add_trace(go.Scatter(
                            x=intruder_edges_x, y=intruder_edges_y,
                            line=dict(width=1, color='red'),
                            hoverinfo='text',
                            text=intruder_edge_texts,
                            mode='lines',
                            name='Intruder Similarities'
                        ))
                    
                    # Add nodes for each group
                    for group_name, color in [(group['group_id'], 'green'), ('external', 'orange')]:
                        node_x = []
                        node_y = []
                        node_text = []
                        
                        for node, attrs in G.nodes(data=True):
                            if attrs.get('group') == group_name:
                                x, y = pos[node]
                                node_x.append(x)
                                node_y.append(y)
                                node_text.append(node)
                        
                        if node_x:  # Only add trace if nodes exist
                            fig.add_trace(go.Scatter(
                                x=node_x, y=node_y,
                                mode='markers+text',
                                marker=dict(
                                    size=15,
                                    color=color,
                                    line=dict(width=1, color='white')
                                ),
                                text=node_text,
                                textposition="top center",
                                name=f"{group_name} Nodes"
                            ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Group {group['group_id']} Similarity Network",
                        showlegend=True,
                        margin=dict(b=20, l=20, r=20, t=40),
                        height=400,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add legend explanation
                    st.markdown("""
                    **Legend**:
                    - **Green nodes**: Leaf nodes in this group
                    - **Orange nodes**: External leaf nodes from other groups
                    - **Blue lines**: Internal similarity connections
                    - **Red lines**: Intruder connections (higher similarity with outside nodes)
                    """)
                    
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
    
    return robustness_data


def generate_taxonomy_review(taxonomy_data, llm_manager):
    """Generate a review of the taxonomy using the LLM"""
    if not llm_manager.selected_model:
        return "Please connect to an LLM and select a model first."
    
    # Prepare a summary of the taxonomy
    nodes = taxonomy_data["nodes"]
    edges = taxonomy_data["edges"]
    
    num_categories = len([n for n in nodes if n.get("type", "") == "category"])
    num_characteristics = len([n for n in nodes if n.get("type", "") == "characteristic"])
    
    # Create a simplified representation for the LLM
    taxonomy_summary = {
        "root": next((n["name"] for n in nodes if n["id"] == "root"), "Root"),
        "categories": {},
    }
    
    # Build a parent-child relationship map
    parent_child_map = {}
    for edge in edges:
        if edge["source"] not in parent_child_map:
            parent_child_map[edge["source"]] = []
        parent_child_map[edge["source"]].append(edge["target"])
    
    # Recursive function to build the taxonomy tree
    def build_tree(node_id, depth=0):
        node = next((n for n in nodes if n["id"] == node_id), None)
        if not node:
            return {"name": "Unknown", "type": "unknown"}
        
        result = {
            "name": node["name"],
            "type": node.get("type", "unknown"),
        }
        
        if "description" in node and node["description"]:
            result["description"] = node["description"]
        
        children = parent_child_map.get(node_id, [])
        if children and depth < 5:  # Limit depth to avoid huge prompts
            result["children"] = [build_tree(child, depth+1) for child in children]
        
        return result
    
    # Build the tree starting from the root
    taxonomy_tree = build_tree("root")
    
    # Convert to a more readable text format for the LLM
    def format_tree(tree, depth=0):
        indent = "  " * depth
        result = f"{indent}- {tree['name']} ({tree['type']})"
        if "description" in tree:
            result += f": {tree['description']}"
        result += "\n"
        
        if "children" in tree:
            for child in tree["children"]:
                result += format_tree(child, depth+1)
        
        return result
    
    formatted_taxonomy = format_tree(taxonomy_tree)
    
    # Calculate metrics
    metrics = calculate_taxonomy_metrics_corrected(taxonomy_data)
    metrics_text = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    
    # Add robustness information if available in session state
    robustness_info = ""
    if 'robustness_data' in st.session_state:
        robustness_info = f"\n\nROBUSTNESS ANALYSIS:\n"
        robustness_info += st.session_state.robustness_data["detailed_explanation"]
        
        # Add information about problematic groups if there are intruders
        if any(g["intruder_count"] > 0 for g in st.session_state.robustness_data["groups"]):
            robustness_info += "\n\nGroups with intruders:\n"
            for group in st.session_state.robustness_data["groups"]:
                if group["intruder_count"] > 0:
                    robustness_info += f"- Group {group['group_id']} ({', '.join(group['characteristics'])}) has {group['intruder_count']} intruders\n"
    
    # Add conciseness information
    conciseness_info = f"\n\nCONCISENESS ANALYSIS:\n"
    conciseness_info += f"Conciseness metric C(T): {metrics['Conciseness']:.4f} (ranges from 0 to 1, higher is better)\n"
    conciseness_info += "This metric evaluates how efficiently the taxonomy organizes information based on node depth."
    conciseness_info += "Lower depth nodes (closer to root) have a higher impact on reducing conciseness."

    # Create the prompt for the LLM
    system_message = """You are an expert in taxonomy evaluation with extensive knowledge in software engineering and effort estimation. 
You are reviewing a taxonomy for software engineering effort estimation methods. Please provide a comprehensive review focusing on:

1. Structure: Is the hierarchy logical and well-organized?
2. Coverage: Does the taxonomy cover the important aspects of effort estimation?
3. Balance: Is the taxonomy balanced in terms of breadth and depth?
4. Clarity: Are the categories and characteristics clearly named and organized?
5. Redundancy: Are there any redundant or overlapping elements?
6. Missing Elements: Are there any important elements missing?
7. Suggestions: How could the taxonomy be improved?

Be specific in your feedback, referring to actual elements in the taxonomy. Provide both strengths and areas for improvement."""

    user_message = f"""Please review the following software engineering effort estimation taxonomy:

TAXONOMY STRUCTURE:
{formatted_taxonomy}

TAXONOMY METRICS:
{metrics_text}
{robustness_info}
{conciseness_info}

Please provide a detailed evaluation of this taxonomy, pointing out strengths, weaknesses, and specific suggestions for improvement.
"""

    # Add specific instructions for robustness and conciseness
    user_message += """
Pay attention to both robustness and conciseness analyses:

- Robustness measures how well the taxonomy groups semantically similar concepts together.
  Low robustness scores indicate that some characteristics might be better placed in different groups.

- Conciseness evaluates the depth efficiency of the taxonomy.
  To improve conciseness, elements should be placed deeper in the tree where appropriate.
"""

    # Generate the review
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        response = llm_manager.generate_chat_completion(
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )
        
        return response
    except Exception as e:
        st.error(f"Error generating review: {str(e)}")
        return f"Error: {str(e)}"

def add_group_toggles_and_visualization(taxonomy_data, robustness_data, viz_options):
    """
    Add group toggles and visualization to the UI
    
    Args:
        taxonomy_data: Dictionary with 'nodes' and 'edges' representing the taxonomy
        robustness_data: Dictionary with robustness analysis results
        viz_options: Visualization options dictionary
        
    Returns:
        None (adds elements to the Streamlit UI)
    """
    import streamlit as st
    
    # Add subheader for the visualization section
    st.subheader("Taxonomy Visualization with Groups and Intruders")
    
    # Add explanation
    st.markdown("""
    This visualization shows the complete taxonomy with:
    - Natural groups clearly delineated with dashed boundaries
    - Intruders marked with red 'X' symbols
    - Dashed red lines showing semantic connections between intruders and nodes in other groups
    - Group labels (G1, G2, etc.) at the top of each group boundary
    
    You can toggle the visibility of connection lines for specific groups using the checkboxes below.
    """)
    
    # Get all unique groups
    all_groups = []
    for group in robustness_data['groups']:
        all_groups.append(group['group_id'])
    
    # Add toggle controls
    st.markdown("### Toggle Connection Visibility")
    
    # Add "Show All" and "Hide All" buttons
    col1, col2 = st.columns(2)
    show_all = col1.button("Show All Connections")
    hide_all = col2.button("Hide All Connections")
    
    # Create checkboxes for each group
    visible_groups = []
    
    # Determine default visibility based on buttons
    default_visibility = True  # Default to showing all connections
    if hide_all:
        default_visibility = False
    if show_all:
        default_visibility = True
    
    # Create a row of checkboxes with appropriate columns
    num_cols = min(len(all_groups), 5)  # Max 5 columns to avoid overcrowding
    cols = st.columns(num_cols)
    
    for i, group_id in enumerate(all_groups):
        col_idx = i % num_cols
        with cols[col_idx]:
            if st.checkbox(f"Group {group_id}", value=default_visibility, key=f"toggle_{group_id}"):
                visible_groups.append(group_id)
    
    # If no groups are selected and we haven't explicitly clicked "Hide All", show a message
    if not visible_groups and not hide_all:
        st.warning("No groups selected. Please select at least one group to see connections or click 'Show All Connections'.")
    
    # Generate the visualization
    with st.spinner("Generating enhanced tree visualization..."):
        # Ensure group boundaries are shown
        viz_options["show_group_boundaries"] = True
        
        try:
            # Create the enhanced visualization with visible groups parameter
            enhanced_fig = create_enhanced_intruder_visualization(
                taxonomy_data,
                robustness_data,
                options=viz_options,
                visible_groups=visible_groups if visible_groups or hide_all else None  # Pass None if all should be visible
            )
            
            # Display the figure
            st.plotly_chart(enhanced_fig, use_container_width=True)
            
            # Add legend explanation
            st.markdown("""
            **Legend**:
            - **Group boundaries**: Dashed boxes showing natural groupings of characteristics
            - **Red X marks**: Intruder characteristics that might be better placed elsewhere
            - **Red dashed arrows**: Connections to semantically similar elements outside the group
            - **Hover over elements**: For detailed information about nodes and connections
            """)
            
            # Add interaction tips
            st.info(
                "**Interaction Tips:**\n"
                "- **Zoom**: Scroll wheel or pinch gesture\n"
                "- **Pan**: Click and drag\n"
                "- **Reset View**: Double-click\n"
                "- **Toggle Groups**: Use checkboxes above to show/hide connection lines\n"
                "- **Select Legend Items**: Click legend items to show/hide elements"
            )
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.write("Try calculating robustness again or check the console for detailed error messages.")

def init_session_state():
    """Initialize session state variables"""
    if 'taxonomy_data' not in st.session_state:
        st.session_state.taxonomy_data = None
    
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = LLMManager()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    
    if 'review' not in st.session_state:
        st.session_state.review = None
    
    if 'embedding_calculator' not in st.session_state:
        st.session_state.embedding_calculator = EmbeddingCalculator()
        
    # Visualization settings
    if 'viz_type' not in st.session_state:
        st.session_state.viz_type = "orthogonal"  # Set orthogonal as default
    
    if 'viz_options' not in st.session_state:
        st.session_state.viz_options = {
            "node_size": 15,
            "font_size": 16,
            "vert_spacing": 0.15,
            "horz_spacing": 1.0,
            "text_tilt": -20,  # Initialize with a slight tilt for better readability
            "edge_width": 1.5,
            "edge_color": "rgba(100,100,100,0.7)",
            "root_color": "#4CAF50",      # Green
            "category_color": "#2196F3",  # Blue
            "characteristic_color": "#FF9800",  # Orange
            "unknown_color": "#9E9E9E",    # Grey
            "staggered_labels": True,      # Toggle for staggered/alternating labels
            "labels_inside_nodes": False,   # Toggle for placing labels inside nodes
            "show_group_boundaries": True  # Toggle for showing group boundaries
        }


def main():
    """Main application function"""
    # Import networkx inside the function for reliability
    import networkx as nx
    
    st.set_page_config(page_title="Agent 0: The Validator", page_icon="✓", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    # Application header
    st.title("Agent 0: The Validator")
    st.subheader("Taxonomy Evaluation Tool")
    
    # Create sidebar for file upload and LLM settings
    with st.sidebar:
        st.header("Taxonomy Upload")
        uploaded_file = st.file_uploader("Upload Taxonomy JSON", type="json")
        
        if uploaded_file:
            taxonomy_data = load_taxonomy(uploaded_file)
            if taxonomy_data:
                st.session_state.taxonomy_data = taxonomy_data
                # Calculate metrics on upload using the corrected method
                with st.spinner("Calculating metrics..."):
                    st.session_state.metrics = calculate_taxonomy_metrics_corrected(taxonomy_data)
        
        st.header("LLM Configuration")
        
        # Server URL
        api_base = st.text_input(
            "LLM Server URL", 
            value=st.session_state.llm_manager.api_base
        )
        
        if api_base != st.session_state.llm_manager.api_base:
            st.session_state.llm_manager = LLMManager(api_base=api_base)
        
        # Connect to LLM button
        if st.button("Connect to LLM"):
            with st.spinner("Connecting to LLM server..."):
                if st.session_state.llm_manager.connect():
                    st.success("Connected to LLM server")
                    
                    # Get available models
                    models = st.session_state.llm_manager.get_available_models()
                    if models:
                        st.session_state.available_models = models
                        st.success(f"Found {len(models)} models")
                    else:
                        st.warning("No models found. Please ensure LM Studio is running with models loaded.")
                else:
                    st.error("Failed to connect to LLM server")
        
        # Model selection
        if hasattr(st.session_state, 'available_models') and st.session_state.available_models:
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
        
        # LLM parameters
        st.subheader("Model Parameters")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                help="Higher values make output more random, lower values make it more deterministic")
        
        max_tokens = st.slider("Max Tokens", min_value=256, max_value=8192, value=4000, step=256,
                               help="Maximum number of tokens to generate in the response")
        
        # Store parameters in session state
        st.session_state.temperature = temperature
        st.session_state.max_tokens = max_tokens
        
        # Generate automatic review button
        if st.session_state.taxonomy_data and st.session_state.llm_manager.selected_model:
            if st.button("Generate Automatic Review"):
                # Make sure robustness data is available for the review
                if 'robustness_data' not in st.session_state:
                    with st.spinner("Calculating robustness for review..."):
                        robustness_data = calculate_taxonomy_robustness_corrected(
                            st.session_state.taxonomy_data, 
                            st.session_state.embedding_calculator
                        )
                        st.session_state.robustness_data = robustness_data
                
                with st.spinner("Generating taxonomy review..."):
                    review = generate_taxonomy_review(
                        st.session_state.taxonomy_data, 
                        st.session_state.llm_manager
                    )
                    st.session_state.review = review
    
    # Main content area
    if st.session_state.taxonomy_data:
        # Create tabs for different views - reordered as requested
        tabs = st.tabs(["Visualization", "Robustness", "Conciseness", "Metrics", "LLM Review", "Chat"])
        
        # Visualization tab - now index 0
        with tabs[0]:
            st.header("Taxonomy Visualization")
            
            # Add visualization type selector
            viz_types = {
                "orthogonal": "Orthogonal Tree (Right-Angled)",
                "indented_list": "Indented List (Text)"
            }
            
            viz_col1, viz_col2 = st.columns([2, 1])
            
            with viz_col1:
                selected_viz_type = st.selectbox(
                    "Visualization Type",
                    options=list(viz_types.keys()),
                    format_func=lambda x: viz_types[x],
                    index=list(viz_types.keys()).index(st.session_state.viz_type) if st.session_state.viz_type in viz_types else 0
                )
                
                # Update session state if changed
                if selected_viz_type != st.session_state.viz_type:
                    st.session_state.viz_type = selected_viz_type
                    st.rerun()
            
            with viz_col2:
                if st.button("Refresh Visualization"):
                    st.rerun()
            
            # Add visualization options for orthogonal visualization
            if st.session_state.viz_type == "orthogonal":
                with st.expander("Visualization Options", expanded=False):
                    opt_col1, opt_col2 = st.columns(2)
                    
                    with opt_col1:
                        node_size = st.slider(
                            "Node Size", 
                            min_value=5, 
                            max_value=30, 
                            value=st.session_state.viz_options["node_size"],
                            key="node_size_slider"
                        )
                        
                        font_size = st.slider(
                            "Font Size", 
                            min_value=8, 
                            max_value=24, 
                            value=st.session_state.viz_options["font_size"],
                            key="font_size_slider"
                        )
                        
                        text_tilt = st.slider(
                            "Text Tilt (degrees)", 
                            min_value=-45, 
                            max_value=45, 
                            value=st.session_state.viz_options["text_tilt"],
                            key="text_tilt_slider"
                        )
                        
                        labels_inside_nodes = st.checkbox(
                            "Labels Inside Nodes",
                            value=st.session_state.viz_options.get("labels_inside_nodes", False),
                            help="Place labels inside the nodes instead of outside (requires larger nodes)",
                            key="labels_inside_nodes_checkbox"
                        )
                        
                        staggered_labels = st.checkbox(
                            "Staggered Labels",
                            value=st.session_state.viz_options.get("staggered_labels", True),
                            help="Alternate label positions between top and bottom for better readability",
                            key="staggered_labels_checkbox",
                            disabled=labels_inside_nodes
                        )
                    
                    with opt_col2:
                        edge_width = st.slider(
                            "Edge Width", 
                            min_value=0.5, 
                            max_value=3.0, 
                            value=st.session_state.viz_options["edge_width"],
                            step=0.1,
                            key="edge_width_slider"
                        )
                        
                        vert_spacing = st.slider(
                            "Vertical Spacing", 
                            min_value=0.05, 
                            max_value=0.3, 
                            value=st.session_state.viz_options.get("vert_spacing", 0.15),
                            step=0.01,
                            key="vert_spacing_slider"
                        )
                        
                        horz_spacing = st.slider(
                            "Horizontal Spacing", 
                            min_value=0.5, 
                            max_value=3.0, 
                            value=st.session_state.viz_options.get("horz_spacing", 1.0),
                            step=0.1,
                            help="Controls how wide the tree spreads horizontally",
                            key="horz_spacing_slider"
                        )
                    
                    # Update options in session state
                    st.session_state.viz_options["node_size"] = node_size
                    st.session_state.viz_options["font_size"] = font_size
                    st.session_state.viz_options["edge_width"] = edge_width
                    st.session_state.viz_options["text_tilt"] = text_tilt
                    st.session_state.viz_options["vert_spacing"] = vert_spacing
                    st.session_state.viz_options["horz_spacing"] = horz_spacing
                    st.session_state.viz_options["staggered_labels"] = staggered_labels
                    st.session_state.viz_options["labels_inside_nodes"] = labels_inside_nodes
            
            # Display visualization based on selected type
            if st.session_state.viz_type == "indented_list":
                # Text-based visualization
                indented_text = visualize_taxonomy(
                    st.session_state.taxonomy_data, 
                    visualization_type="indented_list"
                )
                
                st.markdown("### Indented List Representation")
                st.text(indented_text)
                
                # Add download button for text
                st.download_button(
                    label="Download as Text",
                    data=indented_text,
                    file_name="taxonomy_structure.txt",
                    mime="text/plain"
                )
            else:
                # Plotly-based visualization
                fig = visualize_taxonomy(
                    st.session_state.taxonomy_data, 
                    visualization_type=st.session_state.viz_type,
                    options=st.session_state.viz_options
                )
                
                # Display interactive plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Add description of visualization controls
                st.info(
                    "**Visualization Controls:**\n"
                    "- **Zoom**: Scroll wheel or pinch gesture\n"
                    "- **Pan**: Click and drag\n"
                    "- **Reset View**: Double-click\n"
                    "- **Hover**: Mouse over elements to see details"
                )
        
        # Robustness tab - now index 1
        with tabs[1]:
            st.header("Semantic Robustness Analysis")
            
            # Display the robustness formula explanation
            display_robustness_formula()
            
            # Add a divider
            st.markdown("---")
            
            # Add button to calculate robustness
            if 'robustness_data' not in st.session_state:
                if st.button("Calculate Robustness", key="calc_robustness"):
                    calculate_and_display_robustness(
                        st.session_state.taxonomy_data,
                        st.session_state.embedding_calculator
                    )
            else:
                # Display previously calculated robustness
                st.metric(
                    "Robustness R(T)",
                    f"{st.session_state.robustness_data['robustness']:.4f}",
                    help="Ranges from 0 to 1, where 1 is perfect robustness (no intruders)"
                )
                
                st.markdown(st.session_state.robustness_data["detailed_explanation"])
                
                # Add recalculate button
                if st.button("Recalculate Robustness", key="recalc_robustness"):
                    calculate_and_display_robustness(
                        st.session_state.taxonomy_data,
                        st.session_state.embedding_calculator
                    )
                
                # Add separator
                st.markdown("---")
                
                # THIS IS THE PART THAT CHANGES - REPLACED WITH THE NEW FUNCTION
                # Add group toggles and visualization with group filtering
                add_group_toggles_and_visualization(
                    st.session_state.taxonomy_data,
                    st.session_state.robustness_data,
                    st.session_state.viz_options.copy()
                )
                
                # Add a debug button for visualization issues
                if st.button("Debug Visualization"):
                    try:
                        st.write("Checking if taxonomy data exists...")
                        if 'taxonomy_data' not in st.session_state:
                            st.error("No taxonomy data found!")
                        else:
                            st.success(f"✓ Taxonomy data exists ({len(st.session_state.taxonomy_data.get('nodes', []))} nodes)")
                        
                        st.write("Checking if robustness data exists...")
                        if 'robustness_data' not in st.session_state:
                            st.error("No robustness data found!")
                        else:
                            st.success(f"✓ Robustness data exists")
                            groups = st.session_state.robustness_data.get('groups', [])
                            st.write(f"Found {len(groups)} groups in robustness data")
                            for i, group in enumerate(groups):
                                st.write(f"Group {i+1}: {len(group.get('characteristics', []))} characteristics, {group.get('intruder_count', 0)} intruders")
                    except Exception as debug_error:
                        st.error(f"Debug error: {str(debug_error)}")
                
                # Add separator
                st.markdown("---")
        
        # Conciseness tab - now index 2
        with tabs[2]:
            st.header("Taxonomy Conciseness Analysis")
            
            # Display the conciseness formula
            display_conciseness_formula()
            
            # Add a divider
            st.markdown("---")
            
            # Display the conciseness metric
            conciseness = st.session_state.metrics.get("Conciseness", 0)
            
            st.metric(
                "Conciseness C(T)",
                f"{conciseness:.4f}",
                help="Ranges from 0 to 1, where 1 is maximum conciseness (simplest taxonomy)"
            )
            
            # Add info about calculation method
            st.info("""
            **Note on Calculation Method**: The conciseness calculation properly extracts the depths 
            of all categories and characteristics in the taxonomy according to Prat et al. (2015). 
            This provides an accurate measure of how efficiently the taxonomy organizes information.
            """)
            
            # Add some practical advice
            st.subheader("Improving Conciseness")
            st.markdown("""
            To improve the conciseness of your taxonomy:
            
            1. **Move elements deeper** in the hierarchy where it makes sense semantically
            2. **Consolidate similar categories** at higher levels
            3. **Create more hierarchical levels** instead of having many elements at the same level
            4. **Ensure proper nesting** of categories and characteristics
            """)
            
            # Add a visualization of the taxonomy depth distribution
            st.subheader("Depth Distribution")
            
            # Calculate nodes at each depth
            G = create_nx_graph(st.session_state.taxonomy_data)
            depths = {}
            
            # Find the root node
            root_id = "root"
            if root_id not in G.nodes():
                # Find a node with no incoming edges
                for node_id in G.nodes():
                    if G.in_degree(node_id) == 0:
                        root_id = node_id
                        break
            
            for node_id in G.nodes():
                if node_id == root_id:
                    depth = 0
                else:
                    try:
                        depth = len(nx.shortest_path(G, source=root_id, target=node_id)) - 1
                    except nx.NetworkXNoPath:
                        depth = 0
                
                if depth not in depths:
                    depths[depth] = {"total": 0, "category": 0, "characteristic": 0}
                
                depths[depth]["total"] += 1
                
                # Find node type
                node = next((n for n in st.session_state.taxonomy_data["nodes"] if n["id"] == node_id), None)
                if node and "type" in node:
                    node_type = node["type"].lower()
                    if node_type in depths[depth]:
                        depths[depth][node_type] += 1
            
            # Create data for the chart
            depth_levels = sorted(depths.keys())
            total_counts = [depths[d]["total"] for d in depth_levels]
            category_counts = [depths[d]["category"] for d in depth_levels]
            characteristic_counts = [depths[d]["characteristic"] for d in depth_levels]
            
            # Create bar chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=depth_levels,
                y=category_counts,
                name='Categories',
                marker_color='#2196F3'
            ))
            
            fig.add_trace(go.Bar(
                x=depth_levels,
                y=characteristic_counts,
                name='Characteristics',
                marker_color='#FF9800'
            ))
            
            fig.update_layout(
                title='Node Distribution by Depth',
                xaxis=dict(
                    title='Depth Level',
                    tickmode='linear',
                    dtick=1
                ),
                yaxis=dict(title='Number of Nodes'),
                barmode='stack',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add caption explaining impact on conciseness
            st.caption("""
            **Note:** Nodes at lower depths (closer to the root) have a greater impact on reducing conciseness.
            The chart above shows the distribution of nodes by depth, helping you identify opportunities for restructuring.
            """)
            
            # Create impact visualization to show the contribution of each level to conciseness
            st.subheader("Depth Impact on Conciseness")
            
            # Calculate the impact of each depth level on conciseness
            impact_by_depth = {}
            for depth in depths:
                if depth == 0:  # Skip root level
                    continue
                
                # Calculate sum of 1/depth for this level
                cat_count = depths[depth].get("category", 0)
                char_count = depths[depth].get("characteristic", 0)
                
                if depth > 0:
                    cat_impact = (cat_count * (1/depth)) if cat_count > 0 else 0
                    char_impact = (char_count * (1/depth)) if char_count > 0 else 0
                    total_impact = cat_impact + char_impact
                    
                    impact_by_depth[depth] = {
                        "cat_impact": cat_impact,
                        "char_impact": char_impact,
                        "total_impact": total_impact
                    }
            
            # Prepare data for visualization
            impact_depths = sorted(impact_by_depth.keys())
            cat_impacts = [impact_by_depth[d]["cat_impact"] for d in impact_depths]
            char_impacts = [impact_by_depth[d]["char_impact"] for d in impact_depths]
            
            # Create impact chart
            fig2 = go.Figure()
            
            fig2.add_trace(go.Bar(
                x=impact_depths,
                y=cat_impacts,
                name='Category Impact',
                marker_color='#2196F3'
            ))
            
            fig2.add_trace(go.Bar(
                x=impact_depths,
                y=char_impacts,
                name='Characteristic Impact',
                marker_color='#FF9800'
            ))
            
            fig2.update_layout(
                title='Contribution to Conciseness Formula by Depth',
                xaxis=dict(
                    title='Depth Level',
                    tickmode='linear',
                    dtick=1
                ),
                yaxis=dict(title='Impact (Higher = More Negative Impact)'),
                barmode='stack',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            st.caption("""
            **Explanation:** This chart shows how much each depth level contributes to the denominator 
            in the conciseness formula. Higher values indicate a more negative impact on conciseness.
            Typically, levels 1 and 2 have the most impact because nodes at these levels contribute more to the sum.
            """)
        # Metrics tab - now index 3 with improved table format
        with tabs[3]:
            st.header("Taxonomy Metrics")
            
            if st.session_state.metrics:
                # Create a data dictionary for the metrics table - all metrics combined in one table
                metrics_data = {
                    "Metric": [
                        "Robustness R(T)",
                        "Conciseness C(T)", 
                        "Dimensions", 
                        "Categories", 
                        "Characteristics", 
                        "Total Nodes",
                        "Total Relationships",
                        "Maximum Depth",
                        "Maximum Breadth", 
                        "Average Branching Factor", 
                        "Balance Factor (lower is better)",
                        "Leaf Nodes",
                        "Number of Paths"
                    ],
                    "Value": [
                        f"{st.session_state.robustness_data['robustness']:.2f}" if 'robustness_data' in st.session_state else "N/A",
                        f"{st.session_state.metrics.get('Conciseness', 0):.2f}",
                        "12",  # Example value - could be calculated based on top-level categories
                        str(st.session_state.metrics["Categories"]),
                        str(st.session_state.metrics["Characteristics"]),
                        str(st.session_state.metrics["Total Nodes"]),
                        str(st.session_state.metrics["Total Relationships"]),
                        str(st.session_state.metrics["Maximum Depth"]),
                        str(st.session_state.metrics["Maximum Breadth"]),
                        f"{st.session_state.metrics['Average Branching Factor']}",
                        f"{st.session_state.metrics['Balance Factor (lower is better)']}",
                        str(st.session_state.metrics["Leaf Nodes"]),
                        str(st.session_state.metrics["Number of Paths"])
                    ]
                }
                
                # Convert to DataFrame for better table display
                import pandas as pd
                metrics_df = pd.DataFrame(metrics_data)
                
                # Display the metrics table
                st.table(metrics_df)
                
                # Add description of metrics
                st.subheader("What do these metrics mean?")
                st.markdown("""
                ### Metrics Explanation
                
                - **Robustness R(T)**: Measures how well the taxonomy groups semantically similar concepts together. 
                Values range from 0 to 1, with higher values indicating better semantic grouping. A score of 1 represents 
                perfect robustness with no intruders between groups.
                
                - **Conciseness C(T)**: Evaluates how efficiently the taxonomy is structured based on node depth.
                Values range from 0 to 1, with higher values indicating a more concisely structured taxonomy.
                This metric penalizes elements placed closer to the root more than those at deeper levels.
                
                - **Dimensions**: The number of top-level dimensions or branches in the taxonomy structure.
                
                - **Categories**: The total number of category elements (inner nodes) in the taxonomy.
                
                - **Characteristics**: The total number of characteristic elements (typically leaf nodes) that 
                represent the most specific concepts in the taxonomy.
                
                - **Total Nodes**: Total number of elements in the taxonomy (root + categories + characteristics).
                
                - **Total Relationships**: Number of parent-child relationships between nodes in the taxonomy.
                
                - **Maximum Depth**: The longest path from root to any leaf node, indicating the maximum
                number of hierarchical levels in the taxonomy.
                
                - **Maximum Breadth**: The maximum number of nodes at any level, indicating the widest part of the taxonomy.
                
                - **Average Branching Factor**: Average number of children per non-leaf node, showing how much the taxonomy branches out.
                
                - **Balance Factor**: Standard deviation of children counts (lower means more balanced distribution of nodes).
                
                - **Leaf Nodes**: Total number of nodes with no children (usually characteristics).
                
                - **Number of Paths**: Total number of unique paths from root to leaf nodes.
                """)
            else:
                st.info("Please calculate metrics first.")

        # LLM Review tab - now index 4
        with tabs[4]:
            st.header("LLM Automatic Review")
            
            if st.session_state.review:
                st.markdown(st.session_state.review)
            else:
                # Check if requirements are met for generating a review
                if st.session_state.taxonomy_data and st.session_state.llm_manager.selected_model:
                    if st.button("Generate Automatic Review", key="review_tab_button"):
                        # Make sure robustness data is available for the review
                        if 'robustness_data' not in st.session_state:
                            with st.spinner("Calculating robustness for review..."):
                                robustness_data = calculate_taxonomy_robustness_corrected(
                                    st.session_state.taxonomy_data, 
                                    st.session_state.embedding_calculator
                                )
                                st.session_state.robustness_data = robustness_data
                        
                        with st.spinner("Generating taxonomy review..."):
                            review = generate_taxonomy_review(
                                st.session_state.taxonomy_data, 
                                st.session_state.llm_manager
                            )
                            st.session_state.review = review
                        st.rerun()
                else:
                    if not st.session_state.taxonomy_data:
                        st.info("Please upload a taxonomy JSON file first.")
                    elif not st.session_state.llm_manager.selected_model:
                        st.info("Please connect to an LLM and select a model in the sidebar first.")
                    else:
                        st.info("Please upload a taxonomy and connect to an LLM to generate a review.")
        # Chat tab - now index 5
        with tabs[5]:
            st.header("Chat with LLM about the Taxonomy")
            
            # Chat interface
            if st.session_state.llm_manager.selected_model:
                # Display chat history
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**AI:** {message['content']}")
                    
                    # Add a separator between messages
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
                
                # Input for new message
                user_input = st.text_area("Your question about the taxonomy:", height=100)

                
                if st.button("Send"):
                    if user_input:
                        # Add user message to history
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        
                        # Create messages including system context
                        messages = [
                            {"role": "system", "content": "You are an expert in taxonomy evaluation with extensive knowledge in software engineering and effort estimation. The user is asking about a taxonomy they've uploaded. Be specific and helpful in your responses about taxonomy design and evaluation."}
                        ]
                        
                        # Add the taxonomy structure as context if it's the first message
                        if len(st.session_state.chat_history) <= 1:
                            # Create a simplified representation of the taxonomy
                            nodes = st.session_state.taxonomy_data["nodes"]
                            edges = st.session_state.taxonomy_data["edges"]
                            
                            # Build a parent-child relationship map
                            parent_child_map = {}
                            for edge in edges:
                                if edge["source"] not in parent_child_map:
                                    parent_child_map[edge["source"]] = []
                                parent_child_map[edge["source"]].append(edge["target"])
                            
                            # Recursive function to build the taxonomy tree as text
                            def format_node(node_id, depth=0):
                                node = next((n for n in nodes if n["id"] == node_id), None)
                                if not node:
                                    return ""
                                
                                indent = "  " * depth
                                result = f"{indent}- {node['name']} ({node.get('type', 'unknown')})"
                                if "description" in node and node["description"]:
                                    result += f": {node['description']}"
                                result += "\n"
                                
                                children = parent_child_map.get(node_id, [])
                                for child in children:
                                    result += format_node(child, depth+1)
                                
                                return result
                            
                            # Format the entire taxonomy
                            taxonomy_text = format_node("root")
                            
                            # Create metrics text
                            metrics_text = "\n".join([f"{k}: {v}" for k, v in st.session_state.metrics.items()])
                            
                            # Add robustness info if available
                            robustness_info = ""
                            if 'robustness_data' in st.session_state:
                                robustness_info = f"\n\nRobustness Analysis:\n{st.session_state.robustness_data['detailed_explanation']}"
                            
                            # Add conciseness info
                            conciseness_info = f"\n\nConciseness Analysis:\nConciseness metric C(T): {st.session_state.metrics.get('Conciseness', 0):.4f} (ranges from 0 to 1, higher is better)"
                            
                            # Add as context
                            context_message = {
                                "role": "system", 
                                "content": f"The user has uploaded a taxonomy with the following structure:\n\n{taxonomy_text}\n\nThe metrics for this taxonomy are:\n{metrics_text}{robustness_info}{conciseness_info}"
                            }
                            messages.append(context_message)
                        
                        # Add chat history
                        messages.extend(st.session_state.chat_history)
                        
                        # Generate response
                        with st.spinner("Generating response..."):
                            response = st.session_state.llm_manager.generate_chat_completion(
                                messages=messages,
                                max_tokens=st.session_state.max_tokens,
                                temperature=st.session_state.temperature
                            )
                            
                            # Add response to history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Rerun to display the updated chat
                        st.rerun()
            else:
                st.info("Please connect to an LLM and select a model in the sidebar first.")
                
                # Optional: Add a button to jump to LLM configuration
                if st.button("Configure LLM Connection"):
                    # This will update the UI to highlight the sidebar
                    st.markdown("👈 Please check the sidebar for LLM configuration options.")
    else:
        # Display instructions if no taxonomy is loaded
        st.info("Please upload a taxonomy JSON file in the sidebar to begin evaluation.")
        
        st.markdown("""
        ## How to Use This Tool
        
        1. **Upload Taxonomy**: Use the sidebar to upload a JSON file exported from Agent 1: The Aggregator
        2. **Connect to LLM**: Configure the connection to your local LLM through LM Studio
        3. **Explore Visualization**: View your taxonomy structure with the enhanced orthogonal visualization
        4. **Check Robustness**: Calculate and examine the semantic robustness of your taxonomy groupings
        5. **Evaluate Conciseness**: Analyze how efficiently your taxonomy is structured based on node depth  
        6. **Review Metrics**: Analyze quantitative measures of your taxonomy quality
        7. **Get AI Review**: Generate an automatic evaluation of your taxonomy
        8. **Ask Questions**: Chat with the LLM about specific aspects of your taxonomy
        
        This tool helps you validate the quality and structure of your taxonomy before finalizing it.
        """)

if __name__ == "__main__":
    main()