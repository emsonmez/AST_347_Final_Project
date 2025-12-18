import numpy as np
import pytest
from astropy.table import Table
from galaxycorr.utils import CosmologyUtils

@pytest.fixture
def small_table():
    """Small synthetic RA/Dec/z table for coordinate conversion tests."""
    # Three mock sky positions with increasing redshift
    ra = [0, 45, 90]
    dec = [0, 10, -10]
    z = [0.0, 0.05, 0.1]
    # Construct an astropy Table like a minimal SDSS slice
    return Table([ra, dec, z], names=("ra", "dec", "z"))


def test_E_positive():
    """E(z) should be strictly positive for a range of redshifts."""
    z_vals = np.array([0.0, 0.1, 0.5, 1.0])
    # fiducial ΛCDM parameters (ΩM, Ωk, ΩΛ, Ωr)
    cosmo = (0.3, 0.0, 0.7, 1e-4)
    for z in z_vals:
        # Expansion rate factor must never go negative
        assert CosmologyUtils.E(z, *cosmo) > 0


def test_comoving_distance_increasing():
    """Comoving distance should be zero at z=0 and increase monotonically."""
    cosmo = (0.3, 0.0, 0.7, 1e-4)

    # Distance at zero redshift should vanish
    d0 = CosmologyUtils.comoving_distance(0.0, *cosmo)
    # Higher redshifts should map to larger comoving radii
    d1 = CosmologyUtils.comoving_distance(0.1, *cosmo)
    d2 = CosmologyUtils.comoving_distance(0.5, *cosmo)

    assert d0 == pytest.approx(0.0)
    assert d1 > d0
    assert d2 > d1


def test_comoving_distances_array():
    """Array version of comoving distances should return non-negative values."""
    # Four increasing redshift values
    z_arr = np.array([0.0, 0.05, 0.1, 0.2])
    # Should broadcast into an array of same length
    chi = CosmologyUtils.comoving_distances_array(z_arr, 0.3, 0.0, 0.7, 1e-4)

    assert len(chi) == len(z_arr)
    # Distances must never become negative
    assert np.all(chi >= 0)


def test_cartesian_from_table_shape(small_table):
    """Cartesian conversion should match (N,3) and produce finite values."""
    # Should convert each RA/Dec/z triplet → x,y,z in comoving Mpc/h
    coords = CosmologyUtils.cartesian_from_table(small_table)

    assert coords.shape == (len(small_table), 3)
    # No NaNs or infs from trig or distance integrals
    assert np.all(np.isfinite(coords))