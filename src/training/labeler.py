"""
Semi-Automatic Labeling Engine for Fire Classification.

Applies a priority-ordered rule hierarchy to assign fire type labels
based on spatial proximity to industrial facilities, land cover class,
and temporal persistence characteristics.

Label Priority:
    1. Gas Flare / Industrial Heat  (persistent + near facility)
    2. Industrial Fire              (near facility, non-persistent)
    3. Forest Fire                  (forest land cover)
    4. Agricultural Burning         (agricultural land cover)
    5. Unknown/Other                (everything else)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_GAS_FLARE = "Gas Flare"
LABEL_INDUSTRIAL_HEAT = "Industrial Heat"
LABEL_INDUSTRIAL_FIRE = "Industrial Fire"
LABEL_FOREST_FIRE = "Forest Fire"
LABEL_AGRICULTURAL_BURNING = "Agricultural Burning"
LABEL_UNKNOWN = "Unknown/Other"

# Land-cover codes from config/settings.py
LC_INDUSTRIAL = 2
LC_AGRICULTURAL = 3
LC_FOREST = 4


class SemiAutoLabeler:
    """
    Rule-based labeling engine for supervised fire classification.

    The labeler consumes columns produced by earlier pipeline stages
    (Part 1 preprocessing + Part 2 analysis) and assigns a categorical
    ``fire_label`` column plus a ``label_confidence`` score (0-1).
    """

    def __init__(self, proximity_threshold_km: float = 1.0):
        self.proximity_threshold_km = proximity_threshold_km
        self._label_distribution: dict = {}
        self._flare_keywords = frozenset([
            "gas_flare", "flare", "gas", "lng_terminal", "petroleum_well",
            "oil_refinery", "petrochemical",
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_fires(self, fires_df: pd.DataFrame, facilities_gdf=None) -> pd.DataFrame:
        """
        Apply semi-automatic labeling rules to fire records.

        Parameters
        ----------
        fires_df : DataFrame
            Enriched fire dataset (output of Part 1 / Part 2.4).
        facilities_gdf : GeoDataFrame, optional
            Industrial facility locations.

        Returns
        -------
        DataFrame
            Copy of fires_df with ``fire_label`` and
            ``label_confidence`` columns appended.
        """
        if fires_df.empty:
            logger.warning("Empty fire DataFrame -- nothing to label.")
            return fires_df

        df = fires_df.copy()
        n = len(df)
        logger.info(f"Labeling {n} fire records ...")

        # Initialize label columns
        df["fire_label"] = LABEL_UNKNOWN
        df["label_confidence"] = 0.0

        # Ensure proximity columns exist
        if "distance_to_nearest_industrial" not in df.columns:
            if facilities_gdf is not None and not facilities_gdf.empty:
                df = self._compute_facility_distances(df, facilities_gdf)
            else:
                logger.warning(
                    "No facility distance data and no facilities GeoDataFrame "
                    "provided. Facility-based labeling rules will be skipped."
                )
                df["distance_to_nearest_industrial"] = np.inf
                df["nearest_facility_type"] = "none"

        # ----------------------------------------------------------
        # Rule 1 -- Gas Flare / Industrial Heat (persistent + near)
        # ----------------------------------------------------------
        near_mask = (
            df["distance_to_nearest_industrial"] <= self.proximity_threshold_km
        )
        persistent_mask = self._classify_persistent_thermal(df)
        gas_keywords = self._flare_keywords
        flare_type_mask = df["nearest_facility_type"].astype(str).str.lower().apply(
            lambda t: any(kw in t for kw in gas_keywords)
        )

        # Gas Flare: persistent + near + flare-type facility
        gas_flare_mask = near_mask & persistent_mask & flare_type_mask
        df.loc[gas_flare_mask, "fire_label"] = LABEL_GAS_FLARE
        df.loc[gas_flare_mask, "label_confidence"] = 0.90

        # Industrial Heat: persistent + near + non-flare facility
        industrial_heat_mask = near_mask & persistent_mask & ~flare_type_mask
        df.loc[industrial_heat_mask, "fire_label"] = LABEL_INDUSTRIAL_HEAT
        df.loc[industrial_heat_mask, "label_confidence"] = 0.85

        # ----------------------------------------------------------
        # Rule 2 -- Industrial Fire (near facility, non-persistent)
        # ----------------------------------------------------------
        unlabeled = df["fire_label"] == LABEL_UNKNOWN
        industrial_fire_mask = unlabeled & near_mask & ~persistent_mask
        df.loc[industrial_fire_mask, "fire_label"] = LABEL_INDUSTRIAL_FIRE
        df.loc[industrial_fire_mask, "label_confidence"] = 0.80

        # ----------------------------------------------------------
        # Rule 3 -- Forest Fire (land_cover == 4)
        # ----------------------------------------------------------
        unlabeled = df["fire_label"] == LABEL_UNKNOWN
        if "land_cover" in df.columns:
            forest_mask = unlabeled & (df["land_cover"] == LC_FOREST)
            df.loc[forest_mask, "fire_label"] = LABEL_FOREST_FIRE
            df.loc[forest_mask, "label_confidence"] = 0.75

        # ----------------------------------------------------------
        # Rule 4 -- Agricultural Burning (land_cover == 3)
        # ----------------------------------------------------------
        unlabeled = df["fire_label"] == LABEL_UNKNOWN
        if "land_cover" in df.columns:
            agri_mask = unlabeled & (df["land_cover"] == LC_AGRICULTURAL)
            df.loc[agri_mask, "fire_label"] = LABEL_AGRICULTURAL_BURNING
            df.loc[agri_mask, "label_confidence"] = 0.70

        # ----------------------------------------------------------
        # Rule 5 -- Unknown / Other  (already the default)
        # ----------------------------------------------------------
        unlabeled = df["fire_label"] == LABEL_UNKNOWN
        df.loc[unlabeled, "label_confidence"] = 0.30

        # Store distribution for reporting
        self._label_distribution = df["fire_label"].value_counts().to_dict()
        logger.info("Label distribution:")
        for label, count in self._label_distribution.items():
            pct = count / n * 100
            logger.info(f"  {label:25s}  {count:>6d}  ({pct:5.1f}%)")

        return df

    def get_label_distribution(self) -> dict:
        """Return the most recent label distribution {label: count}."""
        return dict(self._label_distribution)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_facility_distances(self, fires_df, facilities_gdf):
        """Compute haversine distance to nearest industrial facility via BallTree."""
        logger.info("Computing facility distances via BallTree ...")

        if "latitude" not in facilities_gdf.columns:
            facilities_gdf = facilities_gdf.copy()
            facilities_gdf["latitude"] = facilities_gdf.geometry.y
            facilities_gdf["longitude"] = facilities_gdf.geometry.x

        fire_coords = np.radians(
            fires_df[["latitude", "longitude"]].values.astype(float)
        )
        fac_coords = np.radians(
            facilities_gdf[["latitude", "longitude"]].values.astype(float)
        )

        tree = BallTree(fac_coords, metric="haversine")
        distances, indices = tree.query(fire_coords, k=1)

        R_EARTH_KM = 6371.0
        fires_df["distance_to_nearest_industrial"] = distances.flatten() * R_EARTH_KM

        nearest = facilities_gdf.iloc[indices.flatten()]
        if "facility_type" in nearest.columns:
            fires_df["nearest_facility_type"] = nearest["facility_type"].values
        else:
            fires_df["nearest_facility_type"] = "unknown"
        if "name" in nearest.columns:
            fires_df["nearest_facility_name"] = nearest["name"].values

        return fires_df

    def _classify_persistent_thermal(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean mask identifying persistent thermal sources."""
        n = len(df)

        if "cluster_class" in df.columns:
            mask = df["cluster_class"].astype(str).str.lower() == "persistent"
            logger.info(f"Persistent thermal sources (cluster_class): {mask.sum()} / {n}")
            return mask

        if "duration_days" in df.columns:
            mask = df["duration_days"].fillna(0) > 7
            logger.info(f"Persistent thermal sources (duration_days > 7): {mask.sum()} / {n}")
            return mask

        if "is_hotspot" in df.columns:
            mask = df["is_hotspot"].astype(bool)
            logger.info(f"Persistent thermal sources (is_hotspot fallback): {mask.sum()} / {n}")
            return mask

        logger.warning(
            "No persistence columns found (cluster_class, duration_days, "
            "is_hotspot). Persistent-thermal rules will not fire."
        )
        return pd.Series(np.zeros(n, dtype=bool), index=df.index)
