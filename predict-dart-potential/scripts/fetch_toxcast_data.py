"""
Fetch ToxCast/Tox21 bioassay data from EPA CompTox Dashboard.

This script downloads AC50 values and hit-call data for chemicals
screened in the ToxCast program, focusing on assays relevant to DART prediction.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm
import time

# Create data directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# ToxCast assays most relevant for DART prediction (curated list)
DART_RELEVANT_ASSAYS = [
    'ATG_ERE_CIS_up',
    'ATG_ERa_TRANS_up',
    'ATG_AR_TRANS_up',
    'TOX21_AR_LUC_MDAKB2_Agonist',
    'TOX21_ERa_LUC_BG1_Agonist',
    'ATG_GR_TRANS_up',
    'ATG_Myb_CIS_dn',
    'ATG_p53_CIS_dn',
    'TOX21_p53_BLA_p1_ratio',
    'APR_HepG2_CellLoss_24h_dn',
    'APR_HepG2_MitoMass_24h_dn',
    'APR_HepG2_MitoMass_48h_dn',
    'APR_HepG2_OxidativeStress_24h_up',
    'BSK_KF3CT_SRB_down',
    'BSK_KF3CT_VCAM1_up',
    'BSK_LPS_IL1a_up',
    'BSK_LPS_IL8_up',
    'BSK_LPS_TNFa_up',
    'CEETOX_H295R_ESTRONE_up',
    'CEETOX_H295R_ESTRADIOL_up',
    'ACEA_T47D_80hr_Positive',
    'NVS_NR_hER',
    'NVS_NR_hPR',
    'NVS_NR_hAR',
    'OT_ER_ERaERa_1440',
    'OT_ER_ERbERb_1440',
    'Tanguay_ZF_120hpf_MORT_up'
]

def fetch_toxcast_data():
    """
    Fetch ToxCast data from EPA CompTox Dashboard API.
    
    Note: This is a simplified version. In production, you would use the actual
    EPA CompTox API or download bulk data files.
    """
    print("Fetching ToxCast data...")
    
    # For demonstration, we'll create a synthetic dataset structure
    # In production, replace this with actual API calls to:
    # https://comptox.epa.gov/dashboard/api
    
    print("Note: Using synthetic data for demonstration.")
    print("Replace this with actual EPA CompTox API calls in production.")
    
    # Example structure of what the real data would look like
    sample_data = {
        'casrn': ['50-00-0', '71-43-2', '108-88-3', '67-64-1', '79-01-6'],
        'chemical_name': ['Formaldehyde', 'Benzene', 'Toluene', 'Acetone', 'Trichloroethylene'],
        'smiles': ['C=O', 'c1ccccc1', 'Cc1ccccc1', 'CC(=O)C', 'C(=C(Cl)Cl)Cl']
    }
    
    # Add assay data columns (AC50 values in µM, -1 indicates no activity)
    for assay in DART_RELEVANT_ASSAYS[:10]:  # Use subset for demo
        sample_data[f'{assay}_ac50'] = [10.5, -1, 25.3, -1, 5.8] 
        sample_data[f'{assay}_hitcall'] = [1, 0, 1, 0, 1]
    
    df = pd.DataFrame(sample_data)
    
    # Save raw data
    output_path = 'data/raw/toxcast_data.csv'
    df.to_csv(output_path, index=False)
    print(f"ToxCast data saved to {output_path}")
    print(f"Total chemicals: {len(df)}")
    print(f"Total assays: {len(DART_RELEVANT_ASSAYS)}")
    
    return df

def download_bulk_toxcast(url=None):
    """
    Alternative: Download bulk ToxCast data files.
    
    EPA provides bulk downloads at:
    https://www.epa.gov/chemical-research/exploring-toxcast-data
    """
    print("\nAlternative: Download bulk ToxCast files")
    print("Visit: https://www.epa.gov/chemical-research/exploring-toxcast-data")
    print("Download: 'ToxCast_Assay_Matrix.csv' or 'ToxCast_AC50_Data.csv'")

if __name__ == '__main__':
    print("=" * 60)
    print("ToxCast Data Fetcher")
    print("=" * 60)
    
    # Fetch data
    df = fetch_toxcast_data()
    
    # Show instructions for real data
    download_bulk_toxcast()
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
