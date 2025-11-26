"""
Generate molecular descriptors and merge with ToxCast data.

This script:
1. Loads ToxCast assay data and DART labels
2. Generates molecular descriptors using RDKit
3. Creates chemical fingerprints (MACCS, Morgan)
4. Merges all features into a final training dataset
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from tqdm import tqdm

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Create output directory
os.makedirs('data/processed', exist_ok=True)

def generate_molecular_descriptors(smiles):
    """
    Generate molecular descriptors from SMILES string using RDKit.
    
    Args:
        smiles: SMILES string representation of molecule
        
    Returns:
        Dictionary of molecular descriptors
    """
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

def generate_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    """Generate Morgan (circular) fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def generate_maccs_keys(smiles):
    """Generate MACCS keys fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)

def process_features():
    """Main function to process all features."""
    print("=" * 60)
    print("Feature Generation Pipeline")
    print("=" * 60)
    
    # Load ToxCast data
    print("\n1. Loading ToxCast data...")
    toxcast_df = pd.read_csv('data/raw/toxcast_data.csv')
    print(f"   Loaded {len(toxcast_df)} chemicals")
    
    # Load DART labels
    print("\n2. Loading DART labels...")
    labels_df = pd.read_csv('data/raw/dart_labels.csv')
    print(f"   Loaded {len(labels_df)} labeled chemicals")
    
    # Merge datasets on CAS number
    print("\n3. Merging datasets...")
    merged_df = pd.merge(toxcast_df, labels_df[['casrn', 'dart_label']], 
                         on='casrn', how='inner')
    print(f"   Merged dataset: {len(merged_df)} chemicals with labels")
    
    # Generate molecular descriptors
    print("\n4. Generating molecular descriptors...")
    descriptor_list = []
    
    for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        smiles = row['smiles']
        descriptors = generate_molecular_descriptors(smiles)
        
        if descriptors:
            descriptors['casrn'] = row['casrn']
            descriptor_list.append(descriptors)
    
    descriptors_df = pd.DataFrame(descriptor_list)
    print(f"   Generated {len(descriptors_df.columns)-1} molecular descriptors")
    
    # Merge descriptors
    merged_df = pd.merge(merged_df, descriptors_df, on='casrn', how='left')
    
    # Generate Morgan fingerprints
    print("\n5. Generating Morgan fingerprints...")
    morgan_fps = []
    for smiles in tqdm(merged_df['smiles']):
        fp = generate_morgan_fingerprint(smiles)
        morgan_fps.append(fp)
    
    morgan_df = pd.DataFrame(
        morgan_fps, 
        columns=[f'Morgan_{i}' for i in range(1024)]
    )
    morgan_df['casrn'] = merged_df['casrn'].values
    
    # Generate MACCS keys
    print("\n6. Generating MACCS keys...")
    maccs_fps = []
    for smiles in tqdm(merged_df['smiles']):
        fp = generate_maccs_keys(smiles)
        maccs_fps.append(fp)
    
    maccs_df = pd.DataFrame(
        maccs_fps,
        columns=[f'MACCS_{i}' for i in range(167)]
    )
    maccs_df['casrn'] = merged_df['casrn'].values
    
    # Save intermediate files
    print("\n7. Saving processed datasets...")
    
    # Main dataset with descriptors
    output_path = 'data/processed/dart_dataset_with_descriptors.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")
    
    # Morgan fingerprints
    morgan_path = 'data/processed/morgan_fingerprints.csv'
    morgan_df.to_csv(morgan_path, index=False)
    print(f"   Saved: {morgan_path}")
    
    # MACCS keys
    maccs_path = 'data/processed/maccs_fingerprints.csv'
    maccs_df.to_csv(maccs_path, index=False)
    print(f"   Saved: {maccs_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Feature Generation Summary")
    print("=" * 60)
    print(f"Total samples: {len(merged_df)}")
    print(f"ToxCast features: {len([c for c in merged_df.columns if 'ac50' in c or 'hitcall' in c])}")
    print(f"Molecular descriptors: {len(descriptors_df.columns)-1}")
    print(f"Morgan fingerprint bits: 1024")
    print(f"MACCS keys: 167")
    print(f"Total features: {len(merged_df.columns) + 1024 + 167 - 5}")  # excluding metadata
    print("=" * 60)
    
    return merged_df

if __name__ == '__main__':
    try:
        df = process_features()
        print("\nFeature generation complete!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run fetch_toxcast_data.py and fetch_dart_labels.py first.")
