"""
Utility functions for the DART prediction web application.

This module contains helper functions for:
- Chemical structure validation and processing
- Feature generation from SMILES/CAS
- Model predictions
- Data visualization
"""

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Draw
from rdkit import RDLogger
import requests

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class DARTPredictor:
    """Main predictor class for DART toxicity predictions."""
    
    def __init__(self, model_path='data/models/dart_model.pkl',
                 scaler_path='data/models/scaler.pkl',
                 features_path='data/models/feature_names.pkl'):
        """Initialize the predictor with trained model and scaler."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        
    def smiles_to_features(self, smiles):
        """
        Convert SMILES string to feature vector.
        
        Args:
            smiles: SMILES string representation of molecule
            
        Returns:
            Feature vector (numpy array)
        """
        # Generate molecular descriptors
        descriptors = self._generate_molecular_descriptors(smiles)
        if descriptors is None:
            return None
        
        # Generate Morgan fingerprint
        morgan_fp = self._generate_morgan_fingerprint(smiles)
        
        # Generate MACCS keys
        maccs_fp = self._generate_maccs_keys(smiles)
        
        # Combine all features
        # Note: ToxCast assay values would need to be fetched from database
        # For now, we'll use zeros (indicating no activity data available)
        toxcast_features = self._get_placeholder_toxcast_features()
        
        # Combine all features in correct order
        all_features = np.concatenate([
            toxcast_features,
            list(descriptors.values()),
            morgan_fp,
            maccs_fp
        ])
        
        return all_features
    
    def _generate_molecular_descriptors(self, smiles):
        """Generate molecular descriptors from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'FractionCSP3': Descriptors.FractionCsp3(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'MolMR': Descriptors.MolMR(mol),
            'BalabanJ': Descriptors.BalabanJ(mol),
            'BertzCT': Descriptors.BertzCT(mol),
        }
        return descriptors
    
    def _generate_morgan_fingerprint(self, smiles, radius=2, n_bits=1024):
        """Generate Morgan fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    
    def _generate_maccs_keys(self, smiles):
        """Generate MACCS keys."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)
        
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    
    def _get_placeholder_toxcast_features(self):
        """
        Get placeholder ToxCast features.
        
        In production, this would query an actual ToxCast database.
        For now, returns zeros (no activity).
        """
        # Number of ToxCast features (10 assays × 2 features per assay)
        return np.zeros(20)
    
    def predict(self, smiles):
        """
        Make DART prediction for a chemical.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with prediction results
        """
        # Generate features
        features = self.smiles_to_features(smiles)
        if features is None:
            return None
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        
        results = {
            'prediction': int(prediction),
            'prediction_label': 'Positive' if prediction == 1 else 'Negative',
            'probability_negative': float(probability[0]),
            'probability_positive': float(probability[1]),
            'confidence': float(max(probability)),
            'feature_importance': feature_importance
        }
        
        return results

def cas_to_smiles(cas_number):
    """
    Convert CAS number to SMILES using PubChem API.
    
    Args:
        cas_number: CAS registry number
        
    Returns:
        SMILES string or None if not found
    """
    try:
        # Query PubChem API
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            return smiles
        else:
            return None
    except Exception as e:
        print(f"Error fetching SMILES: {e}")
        return None

def validate_smiles(smiles):
    """
    Validate SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Boolean indicating if SMILES is valid
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def draw_molecule(smiles, size=(400, 400)):
    """
    Generate molecular structure image.
    
    Args:
        smiles: SMILES string
        size: Image size tuple (width, height)
        
    Returns:
        PIL Image object
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    img = Draw.MolToImage(mol, size=size)
    return img

def get_molecular_properties(smiles):
    """
    Get basic molecular properties for display.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        'Molecular Formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
        'Molecular Weight': f"{Descriptors.MolWt(mol):.2f} g/mol",
        'LogP': f"{Descriptors.MolLogP(mol):.2f}",
        'H-Bond Donors': Descriptors.NumHDonors(mol),
        'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
        'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
        'Aromatic Rings': Descriptors.NumAromaticRings(mol),
        'TPSA': f"{Descriptors.TPSA(mol):.2f} Ų"
    }
    
    return properties
