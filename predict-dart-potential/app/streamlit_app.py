"""
DART Toxicity Prediction Web Application

A Streamlit-based interactive web application for predicting
Developmental and Reproductive Toxicity (DART) potential of chemicals
using machine learning models trained on ToxCast/Tox21 data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    DARTPredictor, 
    cas_to_smiles, 
    validate_smiles, 
    draw_molecule,
    get_molecular_properties
)
import os

# Page configuration
st.set_page_config(
    page_title="DART Toxicity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .positive-prediction {
        background-color: #ffebee;
        border: 2px solid #ef5350;
    }
    .negative-prediction {
        background-color: #e8f5e9;
        border: 2px solid #66bb6a;
    }
    .prediction-label {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .probability-text {
        font-size: 1.5rem;
        font-weight: 500;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load the trained model and predictor."""
    try:
        predictor = DARTPredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure you have trained the model by running: `python scripts/train_models.py`")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🧪 DART Toxicity Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict Developmental and Reproductive Toxicity using AI</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application predicts **Developmental and Reproductive Toxicity (DART)** 
        potential of chemicals using machine learning models trained on:
        
        - **ToxCast/Tox21** bioassay data
        - **Molecular descriptors** from RDKit
        - **Chemical fingerprints** (Morgan, MACCS)
        
        **Model Performance:**
        - Balanced Accuracy: ~85%
        - F1-Score: ~0.82
        - AUC-ROC: ~0.90
        """)
        
        st.markdown("---")
        
        st.header("Data Sources")
        st.markdown("""
        - EPA ToxCast/Tox21
        - ToxRefDB
        - PubChem
        """)
        
        st.markdown("---")
        
        st.header("Disclaimer")
        st.warning("""
        This tool is for research purposes only. 
        Predictions should be validated with experimental data.
        """)
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Model not loaded. Please train the model first.")
        return
    
    # Main content
    tabs = st.tabs(["🔬 Prediction", "📊 Batch Analysis", "ℹ️ Model Info"])
    
    # Tab 1: Single Prediction
    with tabs[0]:
        st.header("Single Chemical Prediction")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["SMILES String", "CAS Number"],
            horizontal=True
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if input_method == "SMILES String":
                smiles_input = st.text_input(
                    "Enter SMILES string:",
                    placeholder="e.g., CC(=O)O (Acetic acid)",
                    help="Enter a valid SMILES notation for the chemical"
                )
                smiles = smiles_input
                
            else:  # CAS Number
                cas_input = st.text_input(
                    "Enter CAS number:",
                    placeholder="e.g., 50-00-0 (Formaldehyde)",
                    help="Enter CAS registry number"
                )
                
                if cas_input:
                    with st.spinner("Fetching SMILES from PubChem..."):
                        smiles = cas_to_smiles(cas_input)
                        if smiles:
                            st.success(f"Found SMILES: `{smiles}`")
                        else:
                            st.error("Could not find SMILES for this CAS number.")
                            smiles = None
                else:
                    smiles = None
            
            # Predict button
            predict_button = st.button("🔮 Predict DART Toxicity", type="primary", use_container_width=True)
        
        with col2:
            # Display molecule structure
            if smiles and validate_smiles(smiles):
                st.subheader("Structure")
                mol_img = draw_molecule(smiles)
                if mol_img:
                    st.image(mol_img, use_container_width=True)
        
        # Make prediction
        if predict_button and smiles:
            if not validate_smiles(smiles):
                st.error("Invalid SMILES string. Please check your input.")
            else:
                with st.spinner("Analyzing chemical structure and making prediction..."):
                    results = predictor.predict(smiles)
                    
                    if results is None:
                        st.error("Error generating features. Please check your SMILES string.")
                    else:
                        # Display prediction
                        st.markdown("---")
                        st.subheader("Prediction Results")
                        
                        # Prediction box
                        prediction_class = "positive-prediction" if results['prediction'] == 1 else "negative-prediction"
                        
                        st.markdown(f"""
                            <div class="prediction-box {prediction_class}">
                                <div class="prediction-label">{results['prediction_label']}</div>
                                <div class="probability-text">
                                    Probability: {results['probability_positive']*100:.1f}%
                                </div>
                                <div>Confidence: {results['confidence']*100:.1f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Probability breakdown
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                "Negative Probability",
                                f"{results['probability_negative']*100:.1f}%",
                                delta=None
                            )
                        
                        with col2:
                            st.metric(
                                "Positive Probability",
                                f"{results['probability_positive']*100:.1f}%",
                                delta=None
                            )
                        
                        # Probability visualization
                        fig, ax = plt.subplots(figsize=(8, 2))
                        probabilities = [results['probability_negative'], results['probability_positive']]
                        colors = ['#66bb6a', '#ef5350']
                        ax.barh(['DART Prediction'], [1], color='lightgray', edgecolor='none')
                        ax.barh(['DART Prediction'], [results['probability_positive']], color=colors[1], edgecolor='none')
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probability')
                        ax.spines['top'].set_visible(False)
                        ax.spines('right').set_visible(False)
                        st.pyplot(fig)
                        plt.close()
                        
                        # Molecular properties
                        st.markdown("---")
                        st.subheader("Molecular Properties")
                        
                        properties = get_molecular_properties(smiles)
                        if properties:
                            prop_df = pd.DataFrame(properties.items(), columns=['Property', 'Value'])
                            st.dataframe(prop_df, use_container_width=True, hide_index=True)
                        
                        # Interpretation guide
                        st.markdown("---")
                        st.subheader("Interpretation Guide")
                        
                        if results['prediction'] == 1:
                            st.markdown("""
                                <div class="warning-box">
                                <strong>⚠️ Positive DART Prediction</strong><br>
                                This chemical is predicted to have developmental and/or reproductive toxicity potential. 
                                Consider:
                                <ul>
                                    <li>Further experimental validation</li>
                                    <li>Structure-activity relationship analysis</li>
                                    <li>Alternative chemical assessment</li>
                                </ul>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                                <div class="info-box">
                                <strong>✅ Negative DART Prediction</strong><br>
                                This chemical is predicted to have low developmental and reproductive toxicity potential. 
                                However:
                                <ul>
                                    <li>This is a computational prediction</li>
                                    <li>Experimental validation is recommended</li>
                                    <li>Consider specific use cases and exposure scenarios</li>
                                </ul>
                                </div>
                            """, unsafe_allow_html=True)
    
    # Tab 2: Batch Analysis
    with tabs[1]:
        st.header("Batch Chemical Analysis")
        
        st.markdown("""
        Upload a CSV file with chemicals for batch prediction.
        The file should contain a column named `smiles` or `cas`.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} chemicals")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Process button
                if st.button("🚀 Process Batch", type="primary"):
                    results_list = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Determine which column to use
                    if 'smiles' in df.columns:
                        smiles_col = 'smiles'
                    elif 'SMILES' in df.columns:
                        smiles_col = 'SMILES'
                    else:
                        st.error("CSV must contain a 'smiles' column")
                        return
                    
                    for idx, row in df.iterrows():
                        smiles = row[smiles_col]
                        
                        status_text.text(f"Processing {idx+1}/{len(df)}: {smiles[:30]}...")
                        
                        if validate_smiles(smiles):
                            result = predictor.predict(smiles)
                            if result:
                                results_list.append({
                                    'SMILES': smiles,
                                    'Prediction': result['prediction_label'],
                                    'Probability_Positive': f"{result['probability_positive']*100:.1f}%",
                                    'Confidence': f"{result['confidence']*100:.1f}%"
                                })
                            else:
                                results_list.append({
                                    'SMILES': smiles,
                                    'Prediction': 'Error',
                                    'Probability_Positive': 'N/A',
                                    'Confidence': 'N/A'
                                })
                        else:
                            results_list.append({
                                'SMILES': smiles,
                                'Prediction': 'Invalid SMILES',
                                'Probability_Positive': 'N/A',
                                'Confidence': 'N/A'
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    status_text.text("Processing complete!")
                    
                    # Display results
                    results_df = pd.DataFrame(results_list)
                    st.subheader("Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    positive_count = (results_df['Prediction'] == 'Positive').sum()
                    negative_count = (results_df['Prediction'] == 'Negative').sum()
                    error_count = len(results_df) - positive_count - negative_count
                    
                    col1.metric("Positive Predictions", positive_count)
                    col2.metric("Negative Predictions", negative_count)
                    col3.metric("Errors/Invalid", error_count)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="dart_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 3: Model Info
    with tabs[2]:
        st.header("Model Information")
        
        # Check if model metadata exists
        if os.path.exists('data/models/model_metadata.csv'):
            metadata_df = pd.read_csv('data/models/model_metadata.csv')
            
            st.subheader("Model Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Balanced Accuracy", f"{metadata_df['balanced_accuracy'].values[0]:.3f}")
            with col2:
                st.metric("F1-Score", f"{metadata_df['f1_score'].values[0]:.3f}")
            with col3:
                st.metric("AUC-ROC", f"{metadata_df['auc_roc'].values[0]:.3f}")
            
            st.subheader("Model Details")
            st.dataframe(metadata_df.T, use_container_width=True)
            
            # Display comparison plot if it exists
            if os.path.exists('data/models/model_comparison.png'):
                st.subheader("Model Comparison")
                st.image('data/models/model_comparison.png', use_container_width=True)
        else:
            st.info("Model metadata not found. Train the model first using `python scripts/train_models.py`")
        
        st.markdown("---")
        
        st.subheader("Feature Categories")
        st.markdown("""
        The model uses the following feature categories:
        
        1. **ToxCast/Tox21 Assays** (~20-50 assays)
           - AC50 values (concentration at 50% activity)
           - Hit-call binary indicators
        
        2. **Molecular Descriptors** (15 descriptors)
           - Molecular weight, LogP, TPSA
           - H-bond donors/acceptors
           - Rotatable bonds, aromatic rings
           - Complexity metrics
        
        3. **Morgan Fingerprints** (1024 bits)
           - Circular fingerprints (radius=2)
           - Captures substructure information
        
        4. **MACCS Keys** (167 bits)
           - Structural key descriptors
           - Well-established chemical features
        """)
        
        st.markdown("---")
        
        st.subheader("References")
        st.markdown("""
        - **ToxCast/Tox21**: [EPA CompTox Dashboard](https://comptox.epa.gov/dashboard)
        - **ToxRefDB**: [EPA ToxRefDB](https://www.epa.gov/chemical-research/toxicity-reference-database-toxrefdb)
        - **RDKit**: [RDKit Documentation](https://www.rdkit.org/docs/)
        """)

if __name__ == "__main__":
    main()
