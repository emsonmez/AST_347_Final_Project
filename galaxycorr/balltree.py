import numpy as np
from typing import List, Optional


class BallTree:
    """
    BallTree for efficient fixed-radius neighbor searches in 3D.
    Used for fast pair counting in galaxy correlation calculations.
    """

    class Node:
        """
        Node in the BallTree representing a subset of points within a spherical region.

        :param idxs: Indices of points contained in this node
        :type idxs: np.ndarray
        :param center: Center of the ball (mean of points)
        :type center: np.ndarray
        :param radius: Radius of the ball enclosing all points
        :type radius: float
        """
        def __init__(self, idxs: np.ndarray, center: np.ndarray, radius: float):
            self.idxs = idxs
            self.center = center
            self.radius = radius
            self.left: Optional['BallTree.Node'] = None
            self.right: Optional['BallTree.Node'] = None
            self.is_leaf = False

    def __init__(self, data: np.ndarray, leaf_size: int = 50):
        """
        Build a BallTree from 3D point data.

        :param data: N x 3 array of 3D points
        :type data: np.ndarray
        :param leaf_size: Maximum number of points per leaf node
        :type leaf_size: int
        """
        self.data = data
        self.leaf_size = leaf_size
        self.N = data.shape[0]
        self.root = self._build(np.arange(self.N))

    def _build(self, idxs: np.ndarray) -> 'BallTree.Node':
        """
        Recursively construct a BallTree node.

        :param idxs: Indices of points for this node
        :type idxs: np.ndarray
        :return: Constructed BallTree node
        :rtype: BallTree.Node
        """
        pts = self.data[idxs]
        center = np.mean(pts, axis=0)
        radius = np.max(np.linalg.norm(pts - center, axis=1))
        node = self.Node(idxs, center, radius)

        if len(idxs) <= self.leaf_size:
            node.is_leaf = True
            return node

        # Split along axis of maximum variance for balanced partition
        axis = np.argmax(np.var(pts, axis=0))
        median = np.median(pts[:, axis])
        left_mask = pts[:, axis] <= median
        right_mask = ~left_mask

        node.left = self._build(idxs[left_mask])
        node.right = self._build(idxs[right_mask])
        return node

    def _query_radius(self, node: 'BallTree.Node', x: np.ndarray, r: float, result: List[int]):
        """
        Recursively search for points within radius r of query point x.

        :param node: Current node in the tree
        :type node: BallTree.Node
        :param x: Query point
        :type x: np.ndarray
        :param r: Search radius
        :type r: float
        :param result: List to accumulate point indices
        :type result: List[int]
        """
        dist_to_center = np.linalg.norm(node.center - x)

        # Prune entire node if outside radius
        if dist_to_center - node.radius > r:
            return

        if node.is_leaf:
            pts = self.data[node.idxs]
            dists = np.linalg.norm(pts - x, axis=1)
            mask = dists <= r
            result.extend(node.idxs[mask])
            return

        if node.left is not None:
            self._query_radius(node.left, x, r, result)
        if node.right is not None:
            self._query_radius(node.right, x, r, result)

    def query_radius(self, x: np.ndarray, r: float) -> np.ndarray:
        """
        Return indices of points within radius r of query point x.

        :param x: Query point
        :type x: np.ndarray
        :param r: Search radius
        :type r: float
        :return: Indices of points within radius
        :rtype: np.ndarray
        """
        result: List[int] = []
        self._query_radius(self.root, x, r, result)
        return np.array(result, dtype=int)

    def count_pairs_in_bins(self, points: Optional[np.ndarray] = None, r_bins: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Count all pairs between points and self.data within radial bins.

        :param points: Query points (defaults to self.data if None)
        :type points: Optional[np.ndarray]
        :param r_bins: Array of bin edges
        :type r_bins: Optional[np.ndarray]
        :return: Number of pairs in each radial bin
        :rtype: np.ndarray
        """
        if points is None:
            points = self.data
        if r_bins is None:
            raise ValueError("r_bins must be provided.")

        counts = np.zeros(len(r_bins) - 1, dtype=int)
        max_r = r_bins[-1]

        for point in points:
            idxs = self.query_radius(point, max_r)
            if len(idxs) == 0:
                continue
            dists = np.linalg.norm(self.data[idxs] - point, axis=1)
            hist, _ = np.histogram(dists, bins=r_bins)
            counts += hist

        # Remove self-counts for data-data pairs
        if np.array_equal(points, self.data):
            counts = counts - len(points)

        return counts