"""
Tests for demographic utility functions and TBPeople.from_demographics()
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import tbsim as mtb
import starsim as ss
from tbsim.utils.demographics import (
    load_demographic_data,
    create_age_distribution,
    create_sex_distribution,
    validate_demographics,
    create_people_from_demographics,
)


@pytest.fixture
def sample_demo_data():
    """Create sample demographic data for testing."""
    data = pd.DataFrame({
        'age_min': [0, 5, 10, 15, 20],
        'age_max': [5, 10, 15, 20, 25],
        'male_count': [100, 110, 120, 115, 105],
        'female_count': [95, 105, 125, 120, 110]
    })
    return data


@pytest.fixture
def demo_csv_file(sample_demo_data):
    """Create a temporary CSV file with demographic data."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        sample_demo_data.to_csv(f.name, index=False)
        filepath = f.name
    yield filepath
    # Cleanup
    if os.path.exists(filepath):
        os.remove(filepath)


def test_load_demographic_data(demo_csv_file):
    """Test loading demographic data from CSV."""
    data = load_demographic_data(demo_csv_file)
    
    assert isinstance(data, pd.DataFrame)
    assert 'age_min' in data.columns
    assert 'age_max' in data.columns
    assert 'male_count' in data.columns
    assert 'female_count' in data.columns
    assert len(data) == 5


def test_load_demographic_data_missing_file():
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_demographic_data('nonexistent_file.csv')


def test_load_demographic_data_missing_columns(tmp_path):
    """Test that loading CSV with missing columns raises ValueError."""
    bad_csv = tmp_path / "bad.csv"
    bad_data = pd.DataFrame({
        'age_min': [0, 5],
        'age_max': [5, 10]
        # Missing male_count and female_count
    })
    bad_data.to_csv(bad_csv, index=False)
    
    with pytest.raises(ValueError, match="CSV must contain columns"):
        load_demographic_data(str(bad_csv))


def test_create_age_distribution(sample_demo_data):
    """Test creating age distribution from demographic data."""
    age_dist = create_age_distribution(sample_demo_data)
    
    assert isinstance(age_dist, pd.DataFrame)
    assert 'age' in age_dist.columns
    assert 'value' in age_dist.columns
    assert len(age_dist) == 6  # 5 groups + 1 upper edge
    assert age_dist['age'].iloc[0] == 0
    assert age_dist['age'].iloc[-1] == 25  # Upper edge
    assert age_dist['value'].iloc[-1] == 0  # Upper edge has 0 count


def test_create_sex_distribution(sample_demo_data):
    """Test calculating sex distribution proportion."""
    p_female = create_sex_distribution(sample_demo_data)
    
    assert isinstance(p_female, float)
    assert 0 <= p_female <= 1
    
    total_male = sample_demo_data['male_count'].sum()
    total_female = sample_demo_data['female_count'].sum()
    expected = total_female / (total_male + total_female)
    
    assert abs(p_female - expected) < 1e-10


def test_create_sex_distribution_empty():
    """Test sex distribution with zero population."""
    empty_data = pd.DataFrame({
        'age_min': [0],
        'age_max': [5],
        'male_count': [0],
        'female_count': [0]
    })
    p_female = create_sex_distribution(empty_data)
    assert p_female == 0.5  # Should default to 50/50


def test_create_people_from_demographics_exact(demo_csv_file):
    """Test creating people with exact method."""
    people = create_people_from_demographics(demo_csv_file, method='exact')
    
    assert isinstance(people, ss.People)
    assert hasattr(people, '_demographic_method')
    assert people._demographic_method == 'exact'
    assert hasattr(people, '_demographic_ages')
    assert hasattr(people, '_demographic_sexes')


def test_create_people_from_demographics_starsim(demo_csv_file):
    """Test creating people with starsim method."""
    people = create_people_from_demographics(demo_csv_file, method='starsim')
    
    assert isinstance(people, ss.People)
    assert hasattr(people, '_demographic_method')
    assert people._demographic_method == 'starsim'


def test_create_people_from_demographics_scale(demo_csv_file):
    """Test population scaling."""
    people_full = create_people_from_demographics(demo_csv_file, scale_factor=1.0)
    people_half = create_people_from_demographics(demo_csv_file, scale_factor=0.5)
    
    # Half should have approximately half the population
    assert people_half.n_agents_init < people_full.n_agents_init
    assert abs(people_half.n_agents_init / people_full.n_agents_init - 0.5) < 0.1


def test_tbpeople_from_demographics_exact(demo_csv_file):
    """Test TBPeople.from_demographics() with exact method."""
    people = mtb.TBPeople.from_demographics(demo_csv_file, method='exact')
    
    assert isinstance(people, mtb.TBPeople)
    assert people.n_agents_init == 1105  # Sum of all counts
    
    # Create sim and initialize to apply demographics
    sim = ss.Sim(people=people, diseases=[mtb.TB()])
    sim.init()
    
    # Check that demographics were applied
    assert len(sim.people) == 1105
    assert sim.people.age.min() >= 0
    assert sim.people.age.max() < 25


def test_tbpeople_from_demographics_starsim(demo_csv_file):
    """Test TBPeople.from_demographics() with starsim method."""
    people = mtb.TBPeople.from_demographics(demo_csv_file, method='starsim')
    
    assert isinstance(people, mtb.TBPeople)
    
    # Create sim and initialize
    sim = ss.Sim(people=people, diseases=[mtb.TB()])
    sim.init()
    
    # Check that demographics exist
    assert len(sim.people) == 1105
    assert sim.people.age.min() >= 0


def test_tbpeople_from_demographics_with_extra_states(demo_csv_file):
    """Test TBPeople.from_demographics() with additional custom states."""
    custom_states = [ss.FloatArr('SES', default=0.0)]
    people = mtb.TBPeople.from_demographics(
        demo_csv_file, 
        method='exact',
        extra_states=custom_states
    )
    
    assert isinstance(people, mtb.TBPeople)
    
    # Create sim and initialize
    sim = ss.Sim(people=people, diseases=[mtb.TB()])
    sim.init()
    
    # Check that custom state exists
    assert hasattr(sim.people, 'SES')
    assert len(sim.people.SES) == len(sim.people)


def test_validate_demographics_exact(demo_csv_file):
    """Test demographic validation with exact method."""
    # Create people with exact method
    people = mtb.TBPeople.from_demographics(demo_csv_file, method='exact')
    sim = ss.Sim(people=people, diseases=[mtb.TB()])
    sim.init()
    
    # Load expected data
    demo_data = load_demographic_data(demo_csv_file)
    
    # Validate
    validation = validate_demographics(sim.people, demo_data, tolerance=0.02)
    
    # Should match exactly
    assert validation['total_population']['match']
    assert validation['total_population']['within_tolerance']
    assert validation['sex_distribution']['male']['within_tolerance']
    assert validation['sex_distribution']['female']['within_tolerance']


def test_validate_demographics_starsim(demo_csv_file):
    """Test demographic validation with starsim method."""
    # Create people with starsim method
    people = mtb.TBPeople.from_demographics(demo_csv_file, method='starsim')
    sim = ss.Sim(people=people, diseases=[mtb.TB()])
    sim.init()
    
    # Load expected data
    demo_data = load_demographic_data(demo_csv_file)
    
    # Validate (with higher tolerance for sampling)
    validation = validate_demographics(sim.people, demo_data, tolerance=0.2)
    
    # Total should match exactly
    assert validation['total_population']['match']
    # Sex distribution should be close
    assert validation['sex_distribution']['male']['within_tolerance']
    assert validation['sex_distribution']['female']['within_tolerance']


def test_validate_demographics_structure():
    """Test structure of validation results."""
    # Create simple demo data
    demo_data = pd.DataFrame({
        'age_min': [0, 10],
        'age_max': [10, 20],
        'male_count': [100, 90],
        'female_count': [95, 85]
    })
    
    # Create people
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        demo_data.to_csv(f.name, index=False)
        filepath = f.name
    
    try:
        people = mtb.TBPeople.from_demographics(filepath, method='exact')
        sim = ss.Sim(people=people, diseases=[mtb.TB()])
        sim.init()
        
        validation = validate_demographics(sim.people, demo_data)
        
        # Check structure
        assert 'total_population' in validation
        assert 'sex_distribution' in validation
        assert 'age_sex_distribution' in validation
        
        assert 'expected' in validation['total_population']
        assert 'actual' in validation['total_population']
        assert 'rel_diff' in validation['total_population']
        assert 'within_tolerance' in validation['total_population']
        
        assert 'male' in validation['sex_distribution']
        assert 'female' in validation['sex_distribution']
        
        assert len(validation['age_sex_distribution']) == 2  # 2 age groups
        
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_tbpeople_has_tb_states(demo_csv_file):
    """Test that TBPeople from demographics has all TB states."""
    people = mtb.TBPeople.from_demographics(demo_csv_file)
    sim = ss.Sim(people=people, diseases=[mtb.TB()])
    sim.init()
    
    # Check that TB states exist
    assert hasattr(sim.people, 'sought_care')
    assert hasattr(sim.people, 'diagnosed')
    assert hasattr(sim.people, 'symptomatic')
    assert hasattr(sim.people, 'hiv_positive')
    assert hasattr(sim.people, 'hhid')


def test_integration_full_simulation(demo_csv_file):
    """Test full simulation with demographics."""
    # Create people from demographics
    people = mtb.TBPeople.from_demographics(demo_csv_file, method='exact', scale_factor=0.1)
    
    # Create TB disease
    tb = mtb.TB(pars={'beta': ss.peryear(0.01), 'init_prev': 0.1})
    
    # Create simulation
    sim = ss.Sim(
        people=people,
        diseases=[tb],
        dt=ss.days(7),
        start=ss.date('2000-01-01'),
        stop=ss.date('2001-01-01')
    )
    
    # Run simulation
    sim.init()
    sim.run()
    
    # Check that simulation ran
    assert sim.initialized
    assert sim.complete
    assert sim.ti > 0  # Use ti (time index) instead of t (Timeline object)


def test_package_sample_data():
    """Test using package sample demographic data."""
    # Get path to package data
    data_dir = os.path.join(os.path.dirname(mtb.__file__), 'data', 'demographics')
    sample_file = os.path.join(data_dir, 'sample.csv')
    
    # Should exist
    assert os.path.exists(sample_file)
    
    # Should load
    people = mtb.TBPeople.from_demographics(sample_file, method='exact', scale_factor=0.01)
    sim = ss.Sim(people=people, diseases=[mtb.TB()])
    sim.init()
    
    assert len(sim.people) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

