"""
Demographic utility functions for TBsim population initialization.

This module provides functions for loading and processing demographic data
from various sources, creating age and sex distributions, and validating
population structures.
"""

import pandas as pd
import numpy as np
import os
import starsim as ss

__all__ = [
    'load_age_data',
    'load_demographic_data', 
    'create_age_distribution',
    'create_sex_distribution',
    'validate_demographics',
    'create_people_from_demographics',
]


def load_age_data(source='default', file_path=''):
    """
    Load population data from a CSV file or use default data.
    
    Args:
        source: 'default' for UN WPP 1960 data, 'json' for custom JSON file
        file_path: Path to JSON file if source='json'
        
    Returns:
        pd.DataFrame with 'age' and 'value' columns
    """
    if source == 'default':
        # Default population data
        # Gathered from WPP, https://population.un.org/wpp/Download/Standard/MostUsed/
        age_data = pd.DataFrame({ 
            'age': np.arange(0, 101, 5),
            'value': [5791, 4446, 3130, 2361, 2279, 2375, 2032, 1896, 1635, 1547, 1309, 1234, 927, 693, 460, 258, 116, 36, 5, 1, 0]  # 1960
        })
    elif source == 'json':
        if not file_path:
            raise ValueError("file_path must be provided when source is 'json'.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        data = pd.read_json(file_path)
        age_data = pd.DataFrame(data)
    else:
        raise ValueError("Invalid source. Use 'default' or 'json'.")
    return age_data


def load_demographic_data(filepath):
    """
    Load age-sex cross-tabulated demographic data from a CSV file.
    
    Expected CSV format:
        age_min, age_max, male_count, female_count
        0, 5, 450, 430
        5, 10, 480, 460
        ...
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        pd.DataFrame with demographic data
        
    Raises:
        ValueError: If CSV is missing required columns
        FileNotFoundError: If filepath does not exist
        
    Examples:
        >>> demo_data = load_demographic_data('data/demographics/south_africa.csv')
        >>> print(demo_data.head())
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Demographic file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    required_cols = ['age_min', 'age_max', 'male_count', 'female_count']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(
            f"CSV must contain columns: {required_cols}. "
            f"Found: {list(df.columns)}"
        )
    
    return df


def create_age_distribution(demo_data):
    """
    Convert demographic data into age distribution format for Starsim.
    
    Starsim expects age data in histogram format with 'age' and 'value' columns,
    where 'age' represents bin edges and 'value' represents counts or proportions.
    
    Args:
        demo_data: pd.DataFrame with columns: age_min, age_max, male_count, female_count
        
    Returns:
        pd.DataFrame with 'age' and 'value' columns suitable for ss.People(age_data=...)
        
    Examples:
        >>> demo_data = load_demographic_data('demographics.csv')
        >>> age_dist = create_age_distribution(demo_data)
        >>> people = ss.People(n_agents=10000, age_data=age_dist)
    """
    age_bins = []
    age_counts = []
    
    for _, row in demo_data.iterrows():
        age_bins.append(row['age_min'])
        total_count = row['male_count'] + row['female_count']
        age_counts.append(total_count)
    
    # Add upper edge for the last age bin
    if len(demo_data) > 0:
        last_row = demo_data.iloc[-1]
        age_bins.append(last_row['age_max'])
        age_counts.append(0)  # Zero count to mark upper boundary
    
    age_dist_df = pd.DataFrame({
        'age': age_bins,
        'value': age_counts
    })
    
    return age_dist_df


def create_sex_distribution(demo_data):
    """
    Calculate the proportion of females from demographic data.
    
    Args:
        demo_data: pd.DataFrame with columns: male_count, female_count
        
    Returns:
        float: Proportion of females (0.0 to 1.0) for use with ss.bernoulli
        
    Examples:
        >>> demo_data = load_demographic_data('demographics.csv')
        >>> p_female = create_sex_distribution(demo_data)
        >>> print(f"Female proportion: {p_female:.1%}")
    """
    total_male = demo_data['male_count'].sum()
    total_female = demo_data['female_count'].sum()
    total = total_male + total_female
    
    if total == 0:
        return 0.5  # Default to 50/50 if no data
    
    return total_female / total


def validate_demographics(people, demo_data, tolerance=0.1):
    """
    Validate that a population matches expected demographic data.
    
    Since Starsim samples from distributions, exact matches are not expected.
    Validation checks if actual values are within tolerance of expected values.
    
    Args:
        people: ss.People or sim.people object to validate
        demo_data: pd.DataFrame with expected demographics (from load_demographic_data)
        tolerance: Acceptable relative deviation (default 0.1 = 10%)
        
    Returns:
        dict: Validation results with keys:
            - total_population: dict with expected, actual, match, rel_diff, within_tolerance
            - sex_distribution: dict with male/female validation
            - age_sex_distribution: list of dicts with per-group validation
            
    Examples:
        >>> people = mtb.TBPeople.from_demographics('demographics.csv')
        >>> demo_data = load_demographic_data('demographics.csv')
        >>> validation = validate_demographics(people, demo_data)
        >>> print(f"Sex distribution valid: {validation['sex_distribution']['female']['within_tolerance']}")
    """
    validation = {}
    
    # Check total population
    total_expected = demo_data['male_count'].sum() + demo_data['female_count'].sum()
    total_actual = len(people)
    rel_diff = abs(total_actual - total_expected) / total_expected if total_expected > 0 else 0
    validation['total_population'] = {
        'expected': int(total_expected),
        'actual': int(total_actual),
        'match': total_expected == total_actual,
        'rel_diff': rel_diff,
        'within_tolerance': rel_diff <= tolerance
    }
    
    # Check sex distribution
    male_expected = demo_data['male_count'].sum()
    female_expected = demo_data['female_count'].sum()
    male_actual = (~people.female).sum()
    female_actual = people.female.sum()
    
    male_rel_diff = abs(male_actual - male_expected) / male_expected if male_expected > 0 else 0
    female_rel_diff = abs(female_actual - female_expected) / female_expected if female_expected > 0 else 0
    
    validation['sex_distribution'] = {
        'male': {
            'expected': int(male_expected), 
            'actual': int(male_actual),
            'rel_diff': male_rel_diff,
            'within_tolerance': male_rel_diff <= tolerance
        },
        'female': {
            'expected': int(female_expected), 
            'actual': int(female_actual),
            'rel_diff': female_rel_diff,
            'within_tolerance': female_rel_diff <= tolerance
        }
    }
    
    # Check age distribution by group
    age_validation = []
    for _, row in demo_data.iterrows():
        age_min, age_max = row['age_min'], row['age_max']
        mask = (people.age >= age_min) & (people.age < age_max)
        
        actual_male = (~people.female & mask).sum()
        actual_female = (people.female & mask).sum()
        
        male_exp = row['male_count']
        female_exp = row['female_count']
        
        male_rel_diff = abs(actual_male - male_exp) / male_exp if male_exp > 0 else 0
        female_rel_diff = abs(actual_female - female_exp) / female_exp if female_exp > 0 else 0
        
        age_validation.append({
            'age_range': f"{age_min}-{age_max}",
            'male': {
                'expected': int(male_exp), 
                'actual': int(actual_male),
                'rel_diff': male_rel_diff,
                'within_tolerance': male_rel_diff <= tolerance
            },
            'female': {
                'expected': int(female_exp), 
                'actual': int(actual_female),
                'rel_diff': female_rel_diff,
                'within_tolerance': female_rel_diff <= tolerance
            }
        })
    
    validation['age_sex_distribution'] = age_validation
    
    return validation


def create_people_from_demographics(filepath, n_agents=None, scale_factor=1.0, 
                                   method='exact', people_class=None, **kwargs):
    """
    Create a People object from demographic CSV file.
    
    This is a low-level function. Most users should use TBPeople.from_demographics()
    classmethod instead.
    
    Args:
        filepath: Path to demographic data CSV file
        n_agents: Number of agents (calculated from CSV if None)
        scale_factor: Factor to scale the population (1.0 = use counts as-is)
        method: 'exact' for precise matching, 'starsim' for distribution sampling
        people_class: People class to instantiate (default: ss.People)
        **kwargs: Additional arguments passed to People constructor
        
    Returns:
        People instance with specified demographic structure (not yet linked to sim)
        
    Examples:
        >>> people = create_people_from_demographics('demographics.csv', method='exact')
        >>> sim = ss.Sim(people=people, diseases=[...])
    """
    if people_class is None:
        people_class = ss.People
    
    # Load demographic data
    demo_data = load_demographic_data(filepath)
    
    # Scale the demographic data if needed
    if scale_factor != 1.0:
        demo_data = demo_data.copy()
        demo_data['male_count'] = (demo_data['male_count'] * scale_factor).astype(int)
        demo_data['female_count'] = (demo_data['female_count'] * scale_factor).astype(int)
    
    if method == 'exact':
        # Create exact age-sex distribution
        ages = []
        sexes = []
        
        for _, row in demo_data.iterrows():
            age_min = row['age_min']
            age_max = row['age_max']
            male_count = int(row['male_count'])
            female_count = int(row['female_count'])
            
            # Generate ages uniformly within age group
            male_ages = np.random.uniform(age_min, age_max, male_count)
            female_ages = np.random.uniform(age_min, age_max, female_count)
            
            ages.extend(male_ages)
            ages.extend(female_ages)
            sexes.extend([False] * male_count)  # False = male
            sexes.extend([True] * female_count)  # True = female
        
        # Calculate actual n_agents from generated data
        actual_n_agents = len(ages)
        
        # Use n_agents if provided, otherwise use actual count
        if n_agents is None:
            n_agents = actual_n_agents
        elif n_agents != actual_n_agents:
            # Warn if there's a mismatch
            import warnings
            warnings.warn(
                f"Requested n_agents={n_agents} but demographic data has {actual_n_agents} agents. "
                f"Using {actual_n_agents}."
            )
            n_agents = actual_n_agents
        
        # Create People object
        people = people_class(n_agents=n_agents, **kwargs)
        
        # Store for post-initialization
        people._demographic_ages = np.array(ages)
        people._demographic_sexes = np.array(sexes)
        people._demographic_method = 'exact'
        
    else:  # method == 'starsim'
        # Calculate total population if not provided
        if n_agents is None:
            n_agents = int(demo_data['male_count'].sum() + demo_data['female_count'].sum())
        
        # Create age distribution for Starsim
        age_dist_df = create_age_distribution(demo_data)
        
        # Calculate sex distribution (proportion female)
        p_female = create_sex_distribution(demo_data)
        
        # Create People object with custom age and sex distributions
        people = people_class(n_agents=n_agents, age_data=age_dist_df, **kwargs)
        
        # Override default female distribution with our calculated proportion
        people.female.default = ss.bernoulli(p=p_female, name='female')
        people._demographic_method = 'starsim'
    
    return people
