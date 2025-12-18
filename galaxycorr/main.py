import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from astropy.table import Table
from galaxycorr.correlation import CorrelationCalculator
from scipy.signal import find_peaks


def main():
    """
    Main script to analyze SDSS galaxy clustering.
    """

    # Load SDSS galaxy data
    table = Table.read("galaxycorr/data/sdss_galaxies.fits")
    data_cartesian = CorrelationCalculator.from_fits_table(table)
    Nd = len(data_cartesian)

    # First graph: galaxy counts vs fractional radius for SDSS galaxies
    central = data_cartesian[0]  # reference galaxy
    distances = np.linalg.norm(data_cartesian - central, axis=1)
    max_dist = np.max(distances)
    fractions = np.linspace(0.001, 0.25, 20) # 0.1% to 25% maximal seperation
    counts = np.array([np.sum(distances <= f*max_dist) - 1 for f in fractions])  # exclude central galaxy

    plt.figure(figsize=(8,5))
    plt.plot(fractions*100, counts, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='Galaxy Counts')
    plt.fill_between(fractions*100, counts*0.95, counts*1.05, color='#1f77b4', alpha=0.2, label='±5% range')
    plt.xlabel("Fractional Radius (%)", fontsize=12)
    plt.ylabel("Number of Galaxies", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Generate random catalog using 3D KDE
    Nr = 10000  # number of random points
    kde = KernelDensity(bandwidth=5.0, kernel='gaussian')  # bandwidth in Mpc/h
    kde.fit(data_cartesian)
    random_data = kde.sample(Nr, random_state=35)

    # Radial bins
    r_bins = np.linspace(0, 200, 41)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

    # Build correlation calculators
    calc_data = CorrelationCalculator(data_cartesian)
    calc_random = CorrelationCalculator(random_data)

    # Compute normalized counts and ξ(r) using Landy–Szalay formulas 
    dd_norm = calc_data.normalized_counts(r_bins=r_bins) # Data–data
    rr_norm = calc_random.normalized_counts(r_bins=r_bins) # Random–random
    dr_norm = calc_data.normalized_counts(r_bins=r_bins, other_data=random_data) # Data–random
    xi = CorrelationCalculator.landy_szalay(dd_norm, dr_norm, rr_norm)

    # Detect BAO peak within expected range
    peaks, _ = find_peaks(xi, height=0)
    peak_rs = r_centers[peaks]
    peak_xis = xi[peaks]

    bao_min, bao_max = 120, 180 # Expected range to find BAO peak 
    bao_mask = (peak_rs >= bao_min) & (peak_rs <= bao_max)
    bao_rs_in_range = peak_rs[bao_mask]
    bao_xis_in_range = peak_xis[bao_mask]

    if len(bao_rs_in_range) == 0:
        print("No BAO peak detected within expected range.")
        bao_peak_r = None
    else:
        max_idx = np.argmax(bao_xis_in_range)
        bao_peak_r = bao_rs_in_range[max_idx]
        bao_peak_xi = bao_xis_in_range[max_idx]
        print(f"BAO peak detected at r = {bao_peak_r:.2f} Mpc/h with ξ = {bao_peak_xi:.4f}")

    # Plot ξ(r)
    plt.figure(figsize=(8,5))
    plt.plot(r_centers, xi, marker='o', linestyle='-', color='#d62728', linewidth=2, label='ξ(r)')
    plt.axhline(0, color='k', linestyle='--', alpha=0.7)
    plt.axvline(150, color='g', linestyle='--', alpha=0.7, label='Expected BAO (150 Mpc/h)')
    if bao_peak_r is not None:
        plt.axvline(bao_peak_r, color='b', linestyle=':', alpha=0.9,
                    label=f'Measured BAO ({bao_peak_r:.1f} Mpc/h)')
    plt.xlabel("Comoving Distance r [Mpc/h]", fontsize=12)
    plt.ylabel("ξ(r)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()