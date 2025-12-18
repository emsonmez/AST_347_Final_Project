import numpy as np
from typing import Optional
from galaxycorr.balltree import BallTree
from galaxycorr.utils import CosmologyUtils


class CorrelationCalculator:
    """
    Compute galaxy pair statistics and Landy–Szalay ξ(r) correlation function
    using SDSS galaxies and a random catalog.
    """

    def __init__(self, data: np.ndarray, leaf_size: int = 10):
        """
        Initialize the calculator with galaxy data and build a BallTree.

        :param data: N x 3 array of galaxy comoving Cartesian coordinates
        :type data: np.ndarray
        :param leaf_size: Maximum points per leaf node in BallTree
        :type leaf_size: int
        """
        self.data = data
        self.Nd = len(data)
        self.tree = BallTree(data, leaf_size=leaf_size)

    @staticmethod
    def from_fits_table(table) -> np.ndarray:
        """
        Convert an SDSS Astropy Table to comoving Cartesian coordinates.

        :param table: Astropy Table with 'ra', 'dec', 'z' columns
        :type table: Table
        :return: N x 3 array of Cartesian coordinates in Mpc/h
        :rtype: np.ndarray
        """
        return CosmologyUtils.cartesian_from_table(table)

    def count_pairs_in_bins(
        self,
        points: np.ndarray,
        other_points: np.ndarray,
        r_bins: np.ndarray,
        self_pairs: bool = False
    ) -> np.ndarray:
        """
        Count all pairs between two datasets within given radial bins.

        :param points: Array of query points
        :type points: np.ndarray
        :param other_points: Array of points to search neighbors in
        :type other_points: np.ndarray
        :param r_bins: Radial bin edges
        :type r_bins: np.ndarray
        :param self_pairs: Whether to avoid double-counting (for DD or RR)
        :type self_pairs: bool
        :return: Histogram of pair counts per bin
        :rtype: np.ndarray
        """
        counts = np.zeros(len(r_bins)-1, dtype=int)
        max_r = r_bins[-1]
        tree_other = BallTree(other_points)

        for i, point in enumerate(points):
            start_idx = i + 1 if self_pairs and np.array_equal(points, other_points) else 0
            idxs = tree_other.query_radius(point, max_r)
            idxs = idxs[idxs >= start_idx]
            if len(idxs) == 0:
                continue
            dists = np.linalg.norm(other_points[idxs] - point, axis=1)
            hist, _ = np.histogram(dists, bins=r_bins)
            counts += hist
        return counts

    def normalized_counts(
        self,
        r_bins: np.ndarray,
        other_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Normalize pair counts according to Landy–Szalay formulas.

        :param r_bins: Radial bin edges
        :type r_bins: np.ndarray
        :param other_data: Optional second dataset for DR or RR counts
        :type other_data: Optional[np.ndarray]
        :return: Normalized pair counts per bin
        :rtype: np.ndarray
        """
        if other_data is None:
            # Data–data counts
            dd = self.count_pairs_in_bins(self.data, self.data, r_bins, self_pairs=True)
            return dd / (self.Nd * (self.Nd - 1) / 2)
        else:
            Nr = len(other_data)
            is_same = np.array_equal(self.data, other_data)
            counts = self.count_pairs_in_bins(self.data, other_data, r_bins, self_pairs=is_same)
            if is_same:
                # Random–random counts
                return counts / (Nr * (Nr - 1) / 2)
            else:
                # Data–random counts
                return counts / (self.Nd * Nr)

    @staticmethod
    def landy_szalay(
        dd_norm: np.ndarray,
        dr_norm: np.ndarray,
        rr_norm: np.ndarray
    ) -> np.ndarray:
        """
        Compute ξ(r) using the Landy–Szalay estimator.

        :param dd_norm: Normalized data–data counts
        :type dd_norm: np.ndarray
        :param dr_norm: Normalized data–random counts
        :type dr_norm: np.ndarray
        :param rr_norm: Normalized random–random counts
        :type rr_norm: np.ndarray
        :return: ξ(r) values per bin
        :rtype: np.ndarray
        """
        return (dd_norm - 2*dr_norm + rr_norm) / rr_norm
