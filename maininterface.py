import streamlit as st
import importlib
import sys
import os
from pathlib import Path
import json
import pandas as pd
import io
import tempfile


# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Integrated Taxonomy Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS to improve the UI
st.markdown("""
<style>
    .agent-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .pipeline-status {
        background: linear-gradient(90deg, #e3f2fd 0%, #f3e5f5 50%, #e8f5e9 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
    }
    .completed {
        background-color: #d1ffd2;
    }
    .active {
        border: 3px solid #4e8cff;
        box-shadow: 0 0 10px rgba(78, 140, 255, 0.3);
    }
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 10px;
        color: #1E3A8A;
    }
    .status-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
    }
    .status-complete {
        background-color: #4CAF50;
    }
    .status-in-progress {
        background-color: #FFC107;
        animation: pulse 1.5s infinite;
    }
    .status-pending {
        background-color: #9E9E9E;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for data flow between agents
def init_session_state():
    """Initialize all session state variables"""
    if 'current_agent' not in st.session_state:
        st.session_state.current_agent = None
    if 'agent2_output' not in st.session_state:
        st.session_state.agent2_output = None
    if 'agent1_output' not in st.session_state:
        st.session_state.agent1_output = None
    if 'agent0_output' not in st.session_state:
        st.session_state.agent0_output = None
    if 'agent_states' not in st.session_state:
        st.session_state.agent_states = {
            'agent2': {},
            'agent1': {},
            'agent0': {}
        }
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = {
            'agent2': False,
            'agent1': False,
            'agent0': False
        }

init_session_state()

# Function to save agent state
def save_agent_state(agent_name, important_keys):
    """Save important session state variables for an agent"""
    saved_state = {}
    for key in important_keys:
        if key in st.session_state:
            # Handle special objects that might not be JSON serializable
            try:
                # Test if the value is JSON serializable
                json.dumps(st.session_state[key])
                saved_state[key] = st.session_state[key]
            except (TypeError, ValueError):
                # For non-serializable objects, try to extract relevant data
                if hasattr(st.session_state[key], 'export_taxonomy'):
                    # Handle taxonomy_builder object
                    saved_state[key + '_exported'] = st.session_state[key].export_taxonomy(format="json")
                elif hasattr(st.session_state[key], '__dict__'):
                    # Try to save object attributes
                    saved_state[key + '_attributes'] = {
                        k: v for k, v in st.session_state[key].__dict__.items()
                        if not k.startswith('_')
                    }
    
    st.session_state.agent_states[agent_name] = saved_state
    return saved_state

# Function to restore agent state
def restore_agent_state(agent_name):
    """Restore saved session state variables for an agent"""
    if agent_name in st.session_state.agent_states:
        for key, value in st.session_state.agent_states[agent_name].items():
            if key not in st.session_state:
                st.session_state[key] = value

# Define a dummy function to replace set_page_config
def dummy_set_page_config(*args, **kwargs):
    pass

# Function to safely import an agent module
def safe_import_agent(module_name, file_path=None):
    """Import an agent module while preventing duplicate set_page_config calls"""
    try:
        if file_path:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            
            # Temporarily replace streamlit's set_page_config
            original_set_page_config = st.set_page_config
            st.set_page_config = dummy_set_page_config
            
            try:
                spec.loader.exec_module(module)
            finally:
                # Restore original function
                st.set_page_config = original_set_page_config
            
            # Also patch the module's st reference if it exists
            if hasattr(module, 'st'):
                module.st.set_page_config = dummy_set_page_config
                
            return module
    except Exception as e:
        st.error(f"Error importing {module_name}: {str(e)}")
        return None

# Function to find agent files
def find_agent_file(agent_number):
    """Find the appropriate agent file based on common naming patterns"""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    if agent_number == 2:
        patterns = ["A2_preparator.py", "agent2.py", "preparator.py"]
    elif agent_number == 1:
        patterns = ["A1_aggregator.py", "agent1.py", "aggregator.py"]
    elif agent_number == 0:
        patterns = ["A0_validator.py", "agent0.py", "validator.py"]
    else:
        return None
    
    for pattern in patterns:
        file_path = os.path.join(dir_path, pattern)
        if os.path.exists(file_path):
            return file_path
    
    return None

# Function to transfer data from Agent 2 to Agent 1
def transfer_data_2_to_1():
    """Transfer processed data from Agent 2 to Agent 1"""
    try:
        # Get the processed data from Agent 2
        characteristics = st.session_state.get('processed_characteristics', 
                                              st.session_state.get('characteristics', []))
        categories = st.session_state.get('processed_categories', 
                                         st.session_state.get('categories', []))
        
        if not characteristics and not categories:
            st.warning("No data to transfer from Agent 2")
            return False
        
        # Prepare the data in the format Agent 1 expects
        output_data = {
            'characteristics': characteristics,
            'categories': categories,
            'metadata': {
                'source': 'Agent 2: The Preparator',
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save as JSON string
        st.session_state.agent2_output = json.dumps(output_data, indent=2)
        
        # Also save to a temp file for Agent 1 to load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(output_data, f)
            st.session_state.agent2_output_file = f.name
        
        st.success(f"✅ Transferred {len(characteristics)} characteristics and {len(categories)} categories to Agent 1")
        return True
        
    except Exception as e:
        st.error(f"Error transferring data from Agent 2: {str(e)}")
        return False

# Function to transfer data from Agent 1 to Agent 0
def transfer_data_1_to_0():
    """Transfer taxonomy data from Agent 1 to Agent 0"""
    try:
        # Check if taxonomy_builder exists
        if 'taxonomy_builder' in st.session_state:
            # Export the taxonomy
            taxonomy_json = st.session_state.taxonomy_builder.export_taxonomy(format="json")
            st.session_state.agent1_output = taxonomy_json
            
            # Parse the JSON to check node count
            taxonomy_data = json.loads(taxonomy_json)
            node_count = len(taxonomy_data.get('nodes', []))
            edge_count = len(taxonomy_data.get('edges', []))
            
            # Save to a temp file for Agent 0 to load
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(taxonomy_json)
                st.session_state.agent1_output_file = f.name
            
            st.success(f"✅ Transferred taxonomy with {node_count} nodes and {edge_count} edges to Agent 0")
            return True
        else:
            st.warning("No taxonomy data found. Please build a taxonomy in Agent 1 first.")
            return False
            
    except Exception as e:
        st.error(f"Error transferring data from Agent 1: {str(e)}")
        return False

# Enhanced function to run Agent 2
def run_agent2():
    agent2_file = find_agent_file(2)
    
    if not agent2_file:
        st.error("Could not find Agent 2 (A2_preparator.py)")
        return
    
    module_name = os.path.basename(agent2_file).replace('.py', '')
    agent2 = safe_import_agent(module_name, agent2_file)
    
    if not agent2:
        return
    
    # Initialize agent state
    if hasattr(agent2, 'initialize_state'):
        agent2.initialize_state()
    
    # Restore saved state
    restore_agent_state('agent2')
    
    # UI Header
    st.markdown("<h2 style='color: #1E3A8A;'>🔍 Agent 2: The Preparator</h2>", unsafe_allow_html=True)
    st.markdown("*Prepare taxonomy elements by identifying and processing synonyms*")
    
    # Navigation and data transfer controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📤 Transfer Data to Agent 1", type="primary", use_container_width=True):
            # Save current state
            save_agent_state('agent2', [
                'characteristics', 'categories', 'processed_characteristics',
                'processed_categories', 'decisions', 'embedding_calculator'
            ])
            
            # Transfer data
            if transfer_data_2_to_1():
                st.session_state.current_agent = 1
                st.session_state.data_loaded['agent1'] = False  # Reset load flag
                st.rerun()
    
    # Run agent's main function
    if hasattr(agent2, 'main'):
        agent2.main()

# Enhanced function to run Agent 1
def run_agent1():
    agent1_file = find_agent_file(1)
    
    if not agent1_file:
        st.error("Could not find Agent 1 (A1_aggregator.py)")
        return
    
    module_name = os.path.basename(agent1_file).replace('.py', '')
    agent1 = safe_import_agent(module_name, agent1_file)
    
    if not agent1:
        return
    
    # Check if Agent 1 is already initialized
    agent1_initialized = st.session_state.get('agent1_initialized', False)
    
    if not agent1_initialized:
        # FIRST TIME: Initialize everything
        st.write("🔄 Initializing Agent 1...")
        
        # Initialize agent state
        if hasattr(agent1, 'initialize_state'):
            agent1.initialize_state()
        
        # Force re-initialization of embedding calculator with proper class
        if hasattr(agent1, 'EmbeddingCalculator'):
            st.session_state.embedding_calculator = agent1.EmbeddingCalculator()
        
        # Mark as initialized
        st.session_state.agent1_initialized = True
        
    else:
        # SUBSEQUENT TIMES: Just ensure classes are properly linked
        # Don't call initialize_state() again!
        
        # Only fix embedding calculator if it's missing methods
        if ('embedding_calculator' in st.session_state and 
            hasattr(agent1, 'EmbeddingCalculator') and
            not hasattr(st.session_state.embedding_calculator, 'perform_hdbscan_clustering')):
            
            # Backup existing data
            existing_embeddings = getattr(st.session_state.embedding_calculator, 'embeddings', {})
            existing_element_map = getattr(st.session_state.embedding_calculator, 'element_map', {})
            
            # Create new instance with proper class
            st.session_state.embedding_calculator = agent1.EmbeddingCalculator()
            
            # Restore data
            st.session_state.embedding_calculator.embeddings = existing_embeddings
            st.session_state.embedding_calculator.element_map = existing_element_map
    
    # Restore saved state (this should happen after checking initialization)
    restore_agent_state('agent1')
    
    # Auto-load data from Agent 2 if not already loaded
    if st.session_state.agent2_output and not st.session_state.data_loaded['agent1']:
        try:
            # Parse the JSON data
            data = json.loads(st.session_state.agent2_output)
            
            # Set the data in session state for Agent 1
            st.session_state.categories = data.get('categories', [])
            st.session_state.characteristics = data.get('characteristics', [])
            
            # Create unassigned elements list
            unassigned = []
            for cat in data.get('categories', []):
                unassigned.append({
                    "id": cat["id"],
                    "name": cat["name"],
                    "type": "category",
                    "source": cat.get("source", "Unknown"),
                    "description": cat.get("description", "")
                })
            
            for char in data.get('characteristics', []):
                unassigned.append({
                    "id": char["id"],
                    "name": char["name"],
                    "type": "characteristic",
                    "source": char.get("source", "Unknown"),
                    "description": char.get("description", "")
                })
            
            st.session_state.unassigned_elements = unassigned
            st.session_state.data_loaded['agent1'] = True
            
            st.success(f"✅ Automatically loaded {len(data.get('categories', []))} categories and {len(data.get('characteristics', []))} characteristics from Agent 2")
            
        except Exception as e:
            st.error(f"Error loading data from Agent 2: {str(e)}")
    
    # UI Header
    st.markdown("<h2 style='color: #1E3A8A;'>🏗️ Agent 1: The Aggregator</h2>", unsafe_allow_html=True)
    st.markdown("*Build a hierarchical taxonomy structure from prepared elements*")
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Back to Agent 2", use_container_width=True):
            save_agent_state('agent1', [
                'categories', 'characteristics', 'taxonomy_builder',
                'category_clusters', 'characteristic_clusters', 'unassigned_elements',
                'current_step', 'agent1_initialized'
            ])
            st.session_state.current_agent = 2
            st.rerun()
    
    with col3:
        if st.button("📤 Transfer Taxonomy to Agent 0", type="primary", use_container_width=True):
            save_agent_state('agent1', [
                'categories', 'characteristics', 'taxonomy_builder',
                'category_clusters', 'characteristic_clusters',
                'current_step', 'agent1_initialized'
            ])
            
            if transfer_data_1_to_0():
                st.session_state.current_agent = 0
                st.session_state.data_loaded['agent0'] = False
                st.rerun()
    
    # Run agent's main function
    if hasattr(agent1, 'main'):
        agent1.main()

# Enhanced function to run Agent 0
def run_agent0():
    agent0_file = find_agent_file(0)
    
    if not agent0_file:
        st.error("Could not find Agent 0 (A0_validator.py)")
        return
    
    module_name = os.path.basename(agent0_file).replace('.py', '')
    agent0 = safe_import_agent(module_name, agent0_file)
    
    if not agent0:
        return
    
    # Check if Agent 0 is already initialized
    agent0_initialized = st.session_state.get('agent0_initialized', False)
    
    if not agent0_initialized:
        # FIRST TIME: Let Agent 0 initialize itself completely
        st.write("🔄 Initializing Agent 0...")
        
        # Clear any conflicting session state that might interfere
        keys_to_clear = ['metrics', 'robustness_data', 'review', 'embedding_calculator']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Let Agent 0 initialize itself properly
        if hasattr(agent0, 'init_session_state'):
            agent0.init_session_state()
        elif hasattr(agent0, 'initialize_state'):
            agent0.initialize_state()
        
        # Mark as initialized
        st.session_state.agent0_initialized = True
    
    # Restore saved state (this should happen after Agent 0 initializes itself)
    restore_agent_state('agent0')
    
    # Auto-load taxonomy from Agent 1 if not already loaded
    if st.session_state.agent1_output and not st.session_state.data_loaded['agent0']:
        try:
            # Parse the taxonomy JSON
            taxonomy_data = json.loads(st.session_state.agent1_output)
            
            # Set the taxonomy data in session state
            st.session_state.taxonomy_data = taxonomy_data
            st.session_state.data_loaded['agent0'] = True
            
            node_count = len(taxonomy_data.get('nodes', []))
            edge_count = len(taxonomy_data.get('edges', []))
            
            # IMPORTANT: Calculate metrics when taxonomy is loaded
            # This is what was missing and causing the error
            with st.spinner("Calculating taxonomy metrics..."):
                try:
                    # Use Agent 0's metric calculation function
                    if hasattr(agent0, 'calculate_taxonomy_metrics_corrected'):
                        st.session_state.metrics = agent0.calculate_taxonomy_metrics_corrected(taxonomy_data)
                    else:
                        st.error("Agent 0 does not have the required metrics calculation function")
                        return
                except Exception as e:
                    st.error(f"Error calculating metrics: {str(e)}")
                    # Set empty metrics to prevent further errors
                    st.session_state.metrics = {
                        "Total Nodes": 0,
                        "Categories": 0,
                        "Characteristics": 0,
                        "Total Relationships": 0,
                        "Maximum Depth": 0,
                        "Maximum Breadth": 0,
                        "Average Branching Factor": 0,
                        "Balance Factor (lower is better)": 0,
                        "Leaf Nodes": 0,
                        "Number of Paths": 0,
                        "Conciseness": 0
                    }
            
            st.success(f"✅ Automatically loaded taxonomy with {node_count} nodes and {edge_count} edges from Agent 1")
            
        except Exception as e:
            st.error(f"Error loading taxonomy from Agent 1: {str(e)}")
    
    # UI Header
    st.markdown("<h2 style='color: #1E3A8A;'>✅ Agent 0: The Validator</h2>", unsafe_allow_html=True)
    st.markdown("*Validate and evaluate the quality of your taxonomy structure*")
    
    # Navigation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Back to Agent 1", use_container_width=True):
            save_agent_state('agent0', [
                'taxonomy_data', 'metrics', 'robustness_data', 'review', 'agent0_initialized'
            ])
            # Reset both agent initialization flags when switching
            st.session_state.agent1_initialized = False
            st.session_state.agent0_initialized = False
            st.session_state.current_agent = 1
            st.rerun()
    
    with col2:
        if st.session_state.get('taxonomy_data'):
            # Export options
            st.download_button(
                label="📥 Download Final Taxonomy",
                data=json.dumps(st.session_state.taxonomy_data, indent=2),
                file_name="validated_taxonomy.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Run agent's main function
    if hasattr(agent0, 'main'):
        agent0.main()

# Main application
def main():
    # Header
    st.markdown("<h1 class='main-header'>🔄 Integrated Taxonomy Pipeline</h1>", unsafe_allow_html=True)
    
    # Pipeline status indicator
    st.markdown("<div class='pipeline-status'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_class = "status-complete" if st.session_state.agent2_output else ("status-in-progress" if st.session_state.current_agent == 2 else "status-pending")
        st.markdown(f"""
        <div style='text-align: center;'>
            <span class='status-indicator {status_class}'></span>
            <b>Agent 2: Preparator</b><br>
            {f"✅ {len(json.loads(st.session_state.agent2_output).get('characteristics', [])) + len(json.loads(st.session_state.agent2_output).get('categories', []))} elements" if st.session_state.agent2_output else "⏳ Pending"}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status_class = "status-complete" if st.session_state.agent1_output else ("status-in-progress" if st.session_state.current_agent == 1 else "status-pending")
        st.markdown(f"""
        <div style='text-align: center;'>
            <span class='status-indicator {status_class}'></span>
            <b>Agent 1: Aggregator</b><br>
            {f"✅ Taxonomy built" if st.session_state.agent1_output else "⏳ Pending"}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status_class = "status-complete" if st.session_state.agent0_output else ("status-in-progress" if st.session_state.current_agent == 0 else "status-pending")
        st.markdown(f"""
        <div style='text-align: center;'>
            <span class='status-indicator {status_class}'></span>
            <b>Agent 0: Validator</b><br>
            {f"✅ Validated" if st.session_state.agent0_output else "⏳ Pending"}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Navigation")
        
        # Quick navigation buttons
        st.write("### Select Agent")
        
        if st.button("🔍 Agent 2: Preparator", use_container_width=True, 
                    type="primary" if st.session_state.current_agent == 2 else "secondary"):
            st.session_state.current_agent = 2
            st.rerun()
        
        if st.button("🏗️ Agent 1: Aggregator", use_container_width=True,
                    type="primary" if st.session_state.current_agent == 1 else "secondary",
                    disabled=not st.session_state.agent2_output):
            st.session_state.current_agent = 1
            st.rerun()
        
        if st.button("✅ Agent 0: Validator", use_container_width=True,
                    type="primary" if st.session_state.current_agent == 0 else "secondary",
                    disabled=not st.session_state.agent1_output):
            st.session_state.current_agent = 0
            st.rerun()
        
        # Reset options
        st.write("### Pipeline Control")
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # File status
        st.write("### File Status")
        for i, name in [(2, "Preparator"), (1, "Aggregator"), (0, "Validator")]:
            file = find_agent_file(i)
            if file:
                st.success(f"✅ Agent {i}: Found")
            else:
                st.error(f"❌ Agent {i}: Missing")
    
    # Main content area
    if st.session_state.current_agent is None:
        # Welcome screen
        st.write("## Welcome to the Integrated Taxonomy Pipeline")
        
        st.info("""
        ### 🚀 Quick Start Guide
        
        1. **Start with Agent 2** to prepare and clean your taxonomy elements
        2. **Continue to Agent 1** to build your taxonomy structure  
        3. **Finish with Agent 0** to validate and evaluate quality
        
        Data flows automatically between agents - no manual export/import needed!
        """)
        
        # Start button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Pipeline", type="primary", use_container_width=True):
                st.session_state.current_agent = 2
                st.rerun()
    
    # Run the selected agent
    elif st.session_state.current_agent == 2:
        run_agent2()
    elif st.session_state.current_agent == 1:
        run_agent1()
    elif st.session_state.current_agent == 0:
        run_agent0()

if __name__ == "__main__":
    main()