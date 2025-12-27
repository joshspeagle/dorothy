"""
Cross-match astronomical coordinates with Gaia DR3.

This module provides utilities for batch cross-matching RA/Dec coordinates
with the Gaia DR3 catalog using online services (ESA Gaia Archive or CDS XMatch).

No local Gaia file is required - queries are performed via TAP/ADQL.

Example:
    >>> from dorothy.data.gaia_crossmatch import crossmatch_to_gaia_dr3
    >>> ra = np.array([180.0, 181.0, 182.0])
    >>> dec = np.array([45.0, 45.5, 46.0])
    >>> gaia_ids, separations, matched = crossmatch_to_gaia_dr3(ra, dec)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from astropy.table import Table, vstack


logger = logging.getLogger(__name__)


def crossmatch_to_gaia_dr3(
    ra: np.ndarray,
    dec: np.ndarray,
    radius_arcsec: float = 1.0,
    method: Literal["gaia_tap", "cds_xmatch"] = "cds_xmatch",
    chunk_size: int = 50000,
    return_best_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-match RA/Dec coordinates with Gaia DR3.

    This function queries online services to find Gaia DR3 matches for
    input coordinates. No local Gaia file is required.

    Args:
        ra: Right ascension in degrees (ICRS).
        dec: Declination in degrees (ICRS).
        radius_arcsec: Cross-match radius in arcseconds. Default is 1.0.
        method: Which service to use:
            - "cds_xmatch": CDS XMatch service (recommended, simpler)
            - "gaia_tap": ESA Gaia Archive TAP upload
        chunk_size: Number of sources to process per batch. Default is 50000.
        return_best_only: If True, return only the closest match per source.

    Returns:
        Tuple of (gaia_source_ids, separations, matched_mask):
            - gaia_source_ids: Gaia DR3 source_id for each input (-1 if no match)
            - separations: Angular separation in arcseconds (inf if no match)
            - matched_mask: Boolean mask of successfully matched sources

    Example:
        >>> ra = np.array([56.75, 83.82, 101.29])
        >>> dec = np.array([24.12, -5.39, -16.72])
        >>> gaia_ids, seps, matched = crossmatch_to_gaia_dr3(ra, dec)
        >>> print(f"Matched {matched.sum()}/{len(ra)} sources")
    """
    n_sources = len(ra)
    logger.info(f"Cross-matching {n_sources} sources with Gaia DR3 (method={method})")

    if method == "cds_xmatch":
        result = _crossmatch_via_cds(ra, dec, radius_arcsec, chunk_size)
    elif method == "gaia_tap":
        result = _crossmatch_via_gaia_tap(ra, dec, radius_arcsec, chunk_size)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cds_xmatch' or 'gaia_tap'")

    # Convert results to output arrays
    gaia_ids = np.full(n_sources, -1, dtype=np.int64)
    separations = np.full(n_sources, np.inf, dtype=np.float64)
    matched_mask = np.zeros(n_sources, dtype=bool)

    if result is not None and len(result) > 0:
        if return_best_only:
            result = _get_best_matches(result)

        for row in result:
            idx = int(row["input_idx"])
            if 0 <= idx < n_sources:
                gaia_ids[idx] = int(row["source_id"])
                separations[idx] = (
                    float(row["angDist"])
                    if "angDist" in row.colnames
                    else float(row.get("separation_arcsec", 0))
                )
                matched_mask[idx] = True

    logger.info(f"Matched {matched_mask.sum()}/{n_sources} sources")
    return gaia_ids, separations, matched_mask


def _crossmatch_via_cds(
    ra: np.ndarray,
    dec: np.ndarray,
    radius_arcsec: float,
    chunk_size: int,
) -> Table | None:
    """Cross-match using CDS XMatch service."""
    try:
        from astropy import units as u
        from astroquery.xmatch import XMatch
    except ImportError as err:
        raise ImportError(
            "astroquery is required for Gaia cross-matching. "
            "Install with: pip install astroquery"
        ) from err

    n_sources = len(ra)
    n_chunks = (n_sources + chunk_size - 1) // chunk_size
    all_results = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_sources)

        logger.info(
            f"  Processing chunk {i+1}/{n_chunks} (sources {start_idx}-{end_idx-1})"
        )

        # Create input table with indices
        input_table = Table(
            {
                "input_idx": np.arange(start_idx, end_idx),
                "ra": ra[start_idx:end_idx],
                "dec": dec[start_idx:end_idx],
            }
        )

        try:
            # Query CDS XMatch with Gaia DR3 (VizieR catalog I/355/gaiadr3)
            result = XMatch.query(
                cat1=input_table,
                cat2="vizier:I/355/gaiadr3",
                max_distance=radius_arcsec * u.arcsec,
                colRA1="ra",
                colDec1="dec",
            )

            if result is not None and len(result) > 0:
                # Rename columns for consistency
                if "Source" in result.colnames:
                    result.rename_column("Source", "source_id")
                all_results.append(result)
                logger.info(f"    Found {len(result)} matches")
            else:
                logger.info("    No matches found")

        except Exception as e:
            logger.warning(f"    Error in chunk {i+1}: {e}")

    if all_results:
        return vstack(all_results)
    return None


def _crossmatch_via_gaia_tap(
    ra: np.ndarray,
    dec: np.ndarray,
    radius_arcsec: float,
    chunk_size: int,
) -> Table | None:
    """Cross-match using ESA Gaia Archive TAP upload."""
    try:
        from astroquery.gaia import Gaia
    except ImportError as err:
        raise ImportError(
            "astroquery is required for Gaia cross-matching. "
            "Install with: pip install astroquery"
        ) from err

    # Convert radius to degrees for ADQL
    radius_deg = radius_arcsec / 3600.0

    n_sources = len(ra)
    n_chunks = (n_sources + chunk_size - 1) // chunk_size
    all_results = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_sources)

        logger.info(
            f"  Processing chunk {i+1}/{n_chunks} (sources {start_idx}-{end_idx-1})"
        )

        # Create upload table with indices
        chunk_table = Table(
            {
                "input_idx": np.arange(start_idx, end_idx),
                "ra": ra[start_idx:end_idx],
                "dec": dec[start_idx:end_idx],
            }
        )

        # ADQL query for cross-match
        query = f"""
        SELECT
            upload.input_idx,
            gaia.source_id,
            gaia.ra AS gaia_ra,
            gaia.dec AS gaia_dec,
            DISTANCE(
                POINT(upload.ra, upload.dec),
                POINT(gaia.ra, gaia.dec)
            ) * 3600 AS separation_arcsec
        FROM TAP_UPLOAD.crossmatch_input AS upload
        JOIN gaiadr3.gaia_source AS gaia
        ON 1 = CONTAINS(
            POINT(upload.ra, upload.dec),
            CIRCLE(gaia.ra, gaia.dec, {radius_deg})
        )
        ORDER BY upload.input_idx, separation_arcsec
        """

        try:
            job = Gaia.launch_job_async(
                query=query,
                upload_resource=chunk_table,
                upload_table_name="crossmatch_input",
                verbose=False,
            )

            result = job.get_results()

            if result is not None and len(result) > 0:
                all_results.append(result)
                logger.info(f"    Found {len(result)} matches")
            else:
                logger.info("    No matches found")

        except Exception as e:
            logger.warning(f"    Error in chunk {i+1}: {e}")

    if all_results:
        return vstack(all_results)
    return None


def _get_best_matches(table: Table) -> Table:
    """Keep only the best (closest) match per input source."""
    # Sort by input_idx and separation
    sep_col = "angDist" if "angDist" in table.colnames else "separation_arcsec"
    table.sort(["input_idx", sep_col])

    # Get unique input indices (first occurrence = best match)
    _, unique_indices = np.unique(table["input_idx"], return_index=True)

    return table[unique_indices]


def check_gaia_connectivity() -> bool:
    """
    Check if Gaia/CDS services are accessible.

    Returns:
        True if services are reachable, False otherwise.
    """
    try:
        from astropy.table import Table
        from astroquery.xmatch import XMatch

        # Try a minimal query
        test_table = Table({"ra": [180.0], "dec": [45.0]})
        from astropy import units as u

        XMatch.query(
            cat1=test_table,
            cat2="vizier:I/355/gaiadr3",
            max_distance=1 * u.arcsec,
            colRA1="ra",
            colDec1="dec",
        )
        return True
    except Exception as e:
        logger.warning(f"Gaia service check failed: {e}")
        return False
