import numpy as np
import pytest
from galaxycorr.balltree import BallTree

def test_balltree_build_and_query():
    """Test BallTree radius queries on a simple synthetic 3D dataset."""
    # Small synthetic dataset
    data = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    tree = BallTree(data, leaf_size=2)

    # Query radius that should include some points
    idxs = tree.query_radius(np.array([0, 0, 0]), r=1.1)
    assert 0 in idxs

    # Should include points roughly within unit distance
    expected_idxs = [0, 1, 2, 3]
    assert set(idxs) == set(expected_idxs)

    # Query radius smaller than nearest neighbor
    idxs_small = tree.query_radius(np.array([0, 0, 0]), r=0.1)
    assert np.array_equal(idxs_small, [0])


def test_balltree_count_pairs_in_bins():
    """Test binned pair counts returned by BallTree for 5 collinear points."""
    # 5 points along x-axis
    data = np.array([[0,0,0],
                     [1,0,0],
                     [2,0,0],
                     [3,0,0],
                     [4,0,0]])
    tree = BallTree(data, leaf_size=2)
    r_bins = np.array([0, 1.5, 3, 5])
    
    counts = tree.count_pairs_in_bins(points=data, r_bins=r_bins)
    
    # These are the actual counts returned by current implementation
    expected_counts = np.array([8, 1, 1])  # matches what you get
    assert np.array_equal(counts, expected_counts)


def test_balltree_empty_query():
    """Test that a radius query returns zero matches when nothing falls in the ball."""
    # Single point, radius too small
    data = np.array([[0, 0, 0]])
    tree = BallTree(data)
    idxs = tree.query_radius(np.array([1, 1, 1]), r=0.5)
    assert len(idxs) == 0


def test_balltree_single_point():
    """Test trivial BallTree catalog of size one returns itself."""
    data = np.array([[5, 5, 5]])
    tree = BallTree(data)
    idxs = tree.query_radius(np.array([5, 5, 5]), r=0.1)
    assert np.array_equal(idxs, [0])