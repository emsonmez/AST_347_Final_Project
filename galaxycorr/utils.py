import numpy as np
from scipy.integrate import quad
from astropy.table import Table


class CosmologyUtils:
    """
    Cosmology utility functions for converting redshift to comoving distances
    and Cartesian coordinates.
    """

    @staticmethod
    def E(z: float,
          Omega_M: float,
          Omega_k: float,
          Omega_L: float,
          Omega_r: float) -> float:
        """
        Compute the dimensionless Hubble parameter E(z) at redshift z.

        :param z: Redshift
        :type z: float
        :param Omega_M: Matter density parameter
        :type Omega_M: float
        :param Omega_k: Curvature density parameter
        :type Omega_k: float
        :param Omega_L: Cosmological constant density parameter
        :type Omega_L: float
        :param Omega_r: Radiation density parameter
        :type Omega_r: float
        :return: Dimensionless Hubble parameter
        :rtype: float
        """
        return np.sqrt(
            Omega_r * (1 + z)**4 + # Radiation
            Omega_M * (1 + z)**3 + # Matter
            Omega_k * (1 + z)**2 + # Curvature
            Omega_L # Cosmological constant
        )

    @staticmethod
    def comoving_distance(z: float,
                          Omega_M: float,
                          Omega_k: float,
                          Omega_L: float,
                          Omega_r: float) -> float:
        """
        Compute the line-of-sight comoving distance to redshift z in Mpc/h.

        :param z: Redshift
        :type z: float
        :param Omega_M: Matter density parameter
        :type Omega_M: float
        :param Omega_k: Curvature density parameter
        :type Omega_k: float
        :param Omega_L: Cosmological constant density parameter
        :type Omega_L: float
        :param Omega_r: Radiation density parameter
        :type Omega_r: float
        :return: Comoving distance in Mpc/h
        :rtype: float
        """
        integrand = lambda zp: 1.0 / CosmologyUtils.E(zp, Omega_M, Omega_k, Omega_L, Omega_r)
        chi, _ = quad(integrand, 0, z)

        # Convert dimensionless distance to Mpc/h
        c = 299792.458  # km/s
        H0 = 67.74      # km/s/Mpc
        chi *= c / H0

        return chi

    @staticmethod
    def comoving_distances_array(z_array: np.ndarray,
                                 Omega_M: float,
                                 Omega_k: float,
                                 Omega_L: float,
                                 Omega_r: float) -> np.ndarray:
        """
        Compute comoving distances for an array of redshifts in Mpc/h.

        :param z_array: Array of redshifts
        :type z_array: np.ndarray
        :param Omega_M: Matter density parameter
        :type Omega_M: float
        :param Omega_k: Curvature density parameter
        :type Omega_k: float
        :param Omega_L: Cosmological constant density parameter
        :type Omega_L: float
        :param Omega_r: Radiation density parameter
        :type Omega_r: float
        :return: Array of comoving distances in Mpc/h
        :rtype: np.ndarray
        """
        chi_array = np.array([
            CosmologyUtils.comoving_distance(z, Omega_M, Omega_k, Omega_L, Omega_r)
            for z in z_array
        ])

        # Debug: inspect distances
        print("Comoving distances (first 10):", chi_array[:10])
        print("Min/max comoving distances:", np.min(chi_array), np.max(chi_array))

        return chi_array

    @staticmethod
    def cartesian_from_table(table: Table,
                             Omega_M: float = 0.315,
                             Omega_k: float = 0.0,
                             Omega_L: float = 0.685,
                             Omega_r: float = 9.3e-5) -> np.ndarray:
        """
        Convert an Astropy Table with 'ra', 'dec', 'z' columns into 3D comoving Cartesian coordinates.

        :param table: Astropy Table containing galaxy RA, Dec, and redshift
        :type table: Table
        :param Omega_M: Matter density parameter
        :type Omega_M: float
        :param Omega_k: Curvature density parameter
        :type Omega_k: float
        :param Omega_L: Cosmological constant density parameter
        :type Omega_L: float
        :param Omega_r: Radiation density parameter
        :type Omega_r: float
        :return: N x 3 array of Cartesian coordinates in Mpc/h
        :rtype: np.ndarray
        """
        ra = np.array(table['ra'])
        dec = np.array(table['dec'])
        z = np.array(table['z'])

        chi = CosmologyUtils.comoving_distances_array(z, Omega_M, Omega_k, Omega_L, Omega_r)

        # Convert to radians
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        # Spherical to Cartesian conversion
        x = chi * np.cos(dec_rad) * np.cos(ra_rad)
        y = chi * np.cos(dec_rad) * np.sin(ra_rad)
        z_coord = chi * np.sin(dec_rad)

        # Debug: inspect coordinates
        print("Cartesian coordinates min:", x.min(), y.min(), z_coord.min())
        print("Cartesian coordinates max:", x.max(), y.max(), z_coord.max())

        return np.vstack([x, y, z_coord]).T
    
        