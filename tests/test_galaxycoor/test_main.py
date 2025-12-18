import numpy as np
import pytest
from unittest.mock import patch
from astropy.table import Table
from sklearn.neighbors import KernelDensity
from galaxycorr import main as main_module  
from galaxycorr.correlation import CorrelationCalculator

@pytest.fixture
def mock_table():
    ra  = np.array([0,1,2,3,4])
    dec = np.array([0,1,2,3,4])
    z   = np.array([0.01,0.02,0.03,0.04,0.05])
    return Table([ra, dec, z], names=("ra","dec","z"))


@patch("galaxycorr.main.Table.read")
@patch("matplotlib.pyplot.show")     # suppress plotting
def test_main_full(mock_show, mock_read, mock_table):
    # Use synthetic galaxies in place of SDSS FITS
    mock_read.return_value = mock_table

    # Ensure the script runs end-to-end
    main_module.main()

    # File access expected
    mock_read.assert_called_once_with("galaxycorr/data/sdss_galaxies.fits")

    # Cartesian conversion sanity
    xyz = CorrelationCalculator.from_fits_table(mock_table)
    assert xyz.shape == (5, 3)

    # Neighborhood check relative to first point
    d = np.linalg.norm(xyz - xyz[0], axis=1)
    max_d = np.max(d)
    fracs = np.linspace(0.001, 0.25, 20)
    counts = np.array([np.sum(d <= f*max_d) - 1 for f in fracs])
    assert np.all(counts >= 0)

    # Random catalog KDE sampling shape
    kde = KernelDensity(bandwidth=5.0)
    kde.fit(xyz)
    rnd = kde.sample(10000, random_state=35)
    assert rnd.shape == (10000, 3)

    # Correlation arrays finite and correct length
    r_bins = np.linspace(0, 200, 41)
    calc_d = CorrelationCalculator(xyz)
    calc_r = CorrelationCalculator(rnd)

    dd = calc_d.normalized_counts(r_bins=r_bins)
    rr = calc_r.normalized_counts(r_bins=r_bins)
    dr = calc_d.normalized_counts(r_bins=r_bins, other_data=rnd)
    xi = CorrelationCalculator.landy_szalay(dd, dr, rr)

    n = len(r_bins) - 1
    assert len(dd) == len(rr) == len(dr) == len(xi) == n
    assert np.all(np.isfinite(xi))