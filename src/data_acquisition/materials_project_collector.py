from mp_api.client import MPRester
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name)
class MaterialsProjectCollector:
"""
Collector for Materials Project data with robust error handling
and comprehensive ceramic material property extraction
"""
def __init__(self, api_key: str):
    """Initialize MP API client"""
    self.api_key = api_key
    self.mpr = MPRester(api_key)
    self.target_systems = ["Si-C", "B-C", "W-C", "Ti-C", "Al-O"]
    self.collected_data = []
    
def collect_ceramic_data(self) -> pd.DataFrame:
    """
    Collect comprehensive data for target ceramic systems
    Returns DataFrame with all ceramic materials and properties
    """
    logger.info("Starting Materials Project data collection...")
    all_data = []
    
    for system in tqdm(self.target_systems, desc="Collecting ceramic systems"):
        try:
            # Search for materials in each system
            results = self.mpr.materials.search(
                chemsys=system,
                fields=[
                    "material_id", "formula_pretty", "composition",
                    "structure", "density", "formation_energy_per_atom",
                    "energy_above_hull", "is_stable", "ordering",
                    "total_magnetization", "band_gap", "efermi",
                    "elastic", "piezo", "dielectric"
                ],
                num_chunks=None  # Get all materials
            )
            
            logger.info(f"Found {len(results)} materials for system {system}")
            all_data.extend(results)
            
            # Rate limiting to avoid API throttling
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error collecting data for system {system}: {str(e)}")
            continue
    
    # Process collected data into DataFrame
    df = self._process_mp_data(all_data)
    logger.info(f"Successfully collected {len(df)} ceramic materials")
    
    return df

def _process_mp_data(self, raw_data: List) -> pd.DataFrame:
    """Convert MP data to structured DataFrame with feature extraction"""
    processed_records = []
    
    for entry in tqdm(raw_data, desc="Processing MP data"):
        try:
            record = {
                # Basic identifiers
                'material_id': entry.material_id,
                'formula': entry.formula_pretty,
                'composition': entry.composition,
                
                # Thermodynamic properties
                'density': entry.density if entry.density else np.nan,
                'formation_energy': entry.formation_energy_per_atom,
                'energy_above_hull': entry.energy_above_hull,
                'is_stable': entry.is_stable,
                
                # Structural properties
                'crystal_system': entry.structure.lattice.crystal_system if entry.structure else None,
                'space_group': entry.structure.get_space_group_info()[1] if entry.structure else None,
                'volume': entry.structure.volume if entry.structure else np.nan,
                'nsites': entry.structure.num_sites if entry.structure else np.nan,
                
                # Electronic properties
                'band_gap': entry.band_gap if hasattr(entry, 'band_gap') else np.nan,
                'efermi': entry.efermi if hasattr(entry, 'efermi') else np.nan,
                
                # Store structure for feature engineering
                'structure': entry.structure
            }
            
            # Extract elastic properties if available
            if entry.elastic:
                elastic_props = self._extract_elastic_properties(entry.elastic)
                record.update(elastic_props)
            
            processed_records.append(record)
            
        except Exception as e:
            logger.warning(f"Error processing entry {entry.material_id}: {str(e)}")
            continue
    
    return pd.DataFrame(processed_records)

def _extract_elastic_properties(self, elastic_data) -> Dict:
    """Extract elastic tensor properties"""
    props = {}
    
    try:
        if hasattr(elastic_data, 'bulk_modulus'):
            props['bulk_modulus_vrh'] = elastic_data.bulk_modulus.vrh
            props['shear_modulus_vrh'] = elastic_data.shear_modulus.vrh
            props['youngs_modulus'] = elastic_data.youngs_modulus
            props['poisson_ratio'] = elastic_data.poisson_ratio
            
            # Calculate additional elastic properties
            if elastic_data.bulk_modulus.vrh and elastic_data.shear_modulus.vrh:
                props['pugh_ratio'] = elastic_data.shear_modulus.vrh / elastic_data.bulk_modulus.vrh
                props['cauchy_pressure'] = self._calculate_cauchy_pressure(elastic_data)
                
    except Exception as e:
        logger.debug(f"Could not extract elastic properties: {str(e)}")
        
    return props

def _calculate_cauchy_pressure(self, elastic_data) -> float:
    """Calculate Cauchy pressure (C12 - C44) for bonding character"""
    try:
        c_matrix = elastic_data.c_matrix
        c12 = c_matrix[0][1]
        c44 = c_matrix[3][3]
        return c12 - c44
    except:
        return np.nan
