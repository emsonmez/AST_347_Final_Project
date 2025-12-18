from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.table import Table
import astropy.units as u
import os

def fetch_sdss_galaxies(output_path: str = "galaxycorr/data/sdss_galaxies.fits",
                        n_max: int = 20000,
                        dr: int = 17) -> Table:
    """
    Fetch SDSS galaxies with spectroscopic redshifts and save them to a FITS file.

    :param output_path: Path to save the galaxy catalog
    :type output_path: str
    :param n_max: Maximum number of galaxies to fetch
    :type n_max: int
    :param dr: SDSS data release number to query
    :type dr: int
    :return: Astropy Table containing the RA, Dec, and z of galaxies
    :rtype: astropy.table.Table
    """
    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}.")
        table = Table.read(output_path)
        print(f"Loaded {len(table)} galaxies from existing FITS file")
        return table

    # Define the sky window for RA and Dec
    ra_min, ra_max = 150, 160
    dec_min, dec_max = 0, 10

    # Construct and query the SQL query to select galaxies
    query = f"""
    SELECT TOP {n_max} ra, dec, z
    FROM SpecObj
    WHERE class = 'GALAXY'
      AND z > 0 AND z < 0.2
      AND ra BETWEEN {ra_min} AND {ra_max}
      AND dec BETWEEN {dec_min} AND {dec_max}
    """
    print(f"Querying SDSS DR{dr}...")
    try:
        result = SDSS.query_sql(query, data_release=dr)
    except Exception as e:
        raise RuntimeError(f"SDSS query failed: {e}")

    if result is None or len(result) == 0:
        raise RuntimeError("No galaxies returned from SDSS query.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the results to a FITS file
    result.write(output_path, overwrite=True)
    print(f"Saved {len(result)} galaxies to {output_path}")

    return result

if __name__ == "__main__":
    # Fetch the SDSS galaxies and print basic info
    table = fetch_sdss_galaxies()
    print(f"Extracted table with {len(table)} galaxies")
    print(table[:5])  # Display the first 5 rows for inspection 