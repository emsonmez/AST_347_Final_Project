import numpy as np
import pytest
from galaxycorr.correlation import CorrelationCalculator


@pytest.fixture
def simple_data():
    """Synthetic dataset of four points along x-axis."""
    return np.array([[0,0,0],
                     [1,0,0],
                     [2,0,0],
                     [3,0,0]])

@pytest.fixture
def random_data(simple_data):
    """Shift synthetic data to create a mock random catalog."""
    return simple_data + 10


def test_count_pairs_in_bins(simple_data):
    """Test raw pair counts in radial bins for simple synthetic data."""
    calc = CorrelationCalculator(simple_data)
    r_bins = np.array([0, 1.5, 3, 5])
    counts = calc.count_pairs_in_bins(simple_data, simple_data, r_bins, self_pairs=True)

    # Expected bin counts for (0–1.5, 1.5–3, 3–5)
    expected = np.array([3, 2, 1])
    assert np.array_equal(counts, expected)


def test_normalized_counts_dd(simple_data):
    """Test DD normalization equals pair counts divided by Nd*(Nd−1)/2."""
    calc = CorrelationCalculator(simple_data)
    r_bins = np.array([0, 1.5, 3, 5])
    dd = calc.normalized_counts(r_bins=r_bins)

    Nd = len(simple_data)
    expected = np.array([3, 2, 1]) / (Nd * (Nd - 1) / 2)
    np.testing.assert_allclose(dd, expected)


def test_normalized_counts_dr_rr(simple_data, random_data):
    """Test RR and DR normalization against analytic expectations."""
    calc_data = CorrelationCalculator(simple_data)
    calc_random = CorrelationCalculator(random_data)
    r_bins = np.array([0, 1.5, 3, 20])

    # RR normalization
    rr = calc_random.normalized_counts(r_bins=r_bins)
    Nr = len(random_data)
    rr_counts = calc_random.count_pairs_in_bins(random_data, random_data, r_bins, self_pairs=True)
    expected_rr = rr_counts / (Nr * (Nr - 1) / 2)
    np.testing.assert_allclose(rr, expected_rr)

    # DR normalization
    dr = calc_data.normalized_counts(r_bins=r_bins, other_data=random_data)
    expected_dr = calc_data.count_pairs_in_bins(simple_data, random_data, r_bins) / (len(simple_data) * len(random_data))
    np.testing.assert_allclose(dr, expected_dr)


def test_landy_szalay_behavior():
    """Test Landy–Szalay output is finite in bins with nonzero RR."""
    data = np.array([[0,0,0], [1,0,0], [2,0,0]])
    random = np.array([[0.5,0,0], [1.5,0,0], [2.5,0,0]])
    r_bins = np.array([0, 1, 2, 3, 4])

    calc_d = CorrelationCalculator(data)
    calc_r = CorrelationCalculator(random)

    dd = calc_d.normalized_counts(r_bins=r_bins)
    rr = calc_r.normalized_counts(r_bins=r_bins)
    dr = calc_d.normalized_counts(r_bins=r_bins, other_data=random)

    with np.errstate(divide='ignore', invalid='ignore'):
        xi = CorrelationCalculator.landy_szalay(dd, dr, rr)

    nonzero = rr > 0
    assert np.all(np.isfinite(xi[nonzero]))

    # For the last bin, value should be tiny if RR > 0
    if rr[-1] > 0:
        assert abs(xi[-1]) < 1e-12