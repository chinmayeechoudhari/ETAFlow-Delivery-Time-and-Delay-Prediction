#!/usr/bin/env python3
"""
ETAFlow - Synthetic Logistics Dataset Generator
===============================================
A high-performance, professional synthetic dataset generator for logistics delivery
ETA and delay prediction. Generates realistic, domain-grounded Indian logistics
shipment data with non-linear physics, environmental/operational interactions,
controlled missingness, and mathematically coupled targets.

Usage:
    python scripts/generate_dataset.py --samples 100000 --seed 42 --force
"""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ETAFlow.DataGen")

# ==============================================================================
# DOMAIN LOOKUP TABLES & INDIAN LOGISTICS KNOWLEDGE BASE
# ==============================================================================

# Major Indian Logistics Hubs with Coordinates, State, and Tier
# Tier 1 = Primary Metros, Tier 2 = Major Commercial Hubs, Tier 3 = Regional / Emerging
CITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Mumbai": {"state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "tier": "Metro", "zone": "West", "fc_id": "FC-BOM-01"},
    "Pune": {"state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "tier": "Tier 1", "zone": "West", "fc_id": "FC-PNQ-01"},
    "Nagpur": {"state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "tier": "Tier 2", "zone": "Central", "fc_id": "FC-NAG-01"},
    "Nashik": {"state": "Maharashtra", "lat": 19.9975, "lon": 73.7898, "tier": "Tier 2", "zone": "West", "fc_id": "FC-NSK-01"},
    "Delhi": {"state": "Delhi", "lat": 28.6139, "lon": 77.2090, "tier": "Metro", "zone": "North", "fc_id": "FC-DEL-01"},
    "Bengaluru": {"state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "tier": "Metro", "zone": "South", "fc_id": "FC-BLR-01"},
    "Hyderabad": {"state": "Telangana", "lat": 17.3850, "lon": 78.4867, "tier": "Tier 1", "zone": "South", "fc_id": "FC-HYD-01"},
    "Chennai": {"state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "tier": "Metro", "zone": "South", "fc_id": "FC-MAA-01"},
    "Ahmedabad": {"state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "tier": "Tier 1", "zone": "West", "fc_id": "FC-AMD-01"},
    "Surat": {"state": "Gujarat", "lat": 21.1702, "lon": 72.8311, "tier": "Tier 2", "zone": "West", "fc_id": "FC-STV-01"},
    "Vadodara": {"state": "Gujarat", "lat": 22.3072, "lon": 73.1812, "tier": "Tier 2", "zone": "West", "fc_id": "FC-BDQ-01"},
    "Kolkata": {"state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "tier": "Metro", "zone": "East", "fc_id": "FC-CCU-01"},
    "Jaipur": {"state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "tier": "Tier 1", "zone": "North", "fc_id": "FC-JAI-01"},
    "Lucknow": {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "tier": "Tier 1", "zone": "North", "fc_id": "FC-LKO-01"},
    "Kanpur": {"state": "Uttar Pradesh", "lat": 26.4499, "lon": 80.3319, "tier": "Tier 2", "zone": "North", "fc_id": "FC-KNP-01"},
    "Agra": {"state": "Uttar Pradesh", "lat": 27.1767, "lon": 78.0081, "tier": "Tier 2", "zone": "North", "fc_id": "FC-AGR-01"},
    "Varanasi": {"state": "Uttar Pradesh", "lat": 25.3176, "lon": 82.9739, "tier": "Tier 2", "zone": "North", "fc_id": "FC-VNS-01"},
    "Chandigarh": {"state": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "tier": "Tier 1", "zone": "North", "fc_id": "FC-IXC-01"},
    "Ludhiana": {"state": "Punjab", "lat": 30.9010, "lon": 75.8573, "tier": "Tier 2", "zone": "North", "fc_id": "FC-LUH-01"},
    "Bhopal": {"state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126, "tier": "Tier 2", "zone": "Central", "fc_id": "FC-BHO-01"},
    "Indore": {"state": "Madhya Pradesh", "lat": 22.7196, "lon": 75.8577, "tier": "Tier 1", "zone": "Central", "fc_id": "FC-IDR-01"},
    "Kochi": {"state": "Kerala", "lat": 9.9312, "lon": 76.2673, "tier": "Tier 1", "zone": "South", "fc_id": "FC-COK-01"},
    "Coimbatore": {"state": "Tamil Nadu", "lat": 11.0168, "lon": 76.9558, "tier": "Tier 2", "zone": "South", "fc_id": "FC-CJB-01"},
    "Visakhapatnam": {"state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185, "tier": "Tier 2", "zone": "South", "fc_id": "FC-VTZ-01"},
    "Patna": {"state": "Bihar", "lat": 25.5941, "lon": 85.1376, "tier": "Tier 2", "zone": "East", "fc_id": "FC-PAT-01"},
    "Bhubaneswar": {"state": "Odisha", "lat": 20.2961, "lon": 85.8245, "tier": "Tier 2", "zone": "East", "fc_id": "FC-BBI-01"},
    "Guwahati": {"state": "Assam", "lat": 26.1445, "lon": 91.7362, "tier": "Tier 2", "zone": "East", "fc_id": "FC-GAU-01"},
    "Panaji": {"state": "Goa", "lat": 15.4909, "lon": 73.8278, "tier": "Tier 3", "zone": "West", "fc_id": "FC-GOI-01"},
}

CITIES: List[str] = list(CITY_REGISTRY.keys())

# Hub selection weights (Metros generate higher origin/destination traffic)
CITY_WEIGHTS: np.ndarray = np.array([
    0.12,  # Mumbai
    0.06,  # Pune
    0.03,  # Nagpur
    0.02,  # Nashik
    0.13,  # Delhi
    0.11,  # Bengaluru
    0.08,  # Hyderabad
    0.07,  # Chennai
    0.05,  # Ahmedabad
    0.03,  # Surat
    0.02,  # Vadodara
    0.06,  # Kolkata
    0.04,  # Jaipur
    0.03,  # Lucknow
    0.02,  # Kanpur
    0.015, # Agra
    0.015, # Varanasi
    0.02,  # Chandigarh
    0.02,  # Ludhiana
    0.015, # Bhopal
    0.025, # Indore
    0.02,  # Kochi
    0.015, # Coimbatore
    0.015, # Visakhapatnam
    0.015, # Patna
    0.015, # Bhubaneswar
    0.01,  # Guwahati
    0.005, # Panaji
])
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

# Major Indian Logistics Carriers and their Baseline Reliability Profiles
CARRIER_PROFILES: Dict[str, Dict[str, Any]] = {
    "BlueDart": {"weight": 0.22, "base_delay_rate": 0.14, "speed_factor": 1.15, "tier": "Premium"},
    "Delhivery": {"weight": 0.28, "base_delay_rate": 0.22, "speed_factor": 1.02, "tier": "Standard"},
    "Ekart Logistics": {"weight": 0.18, "base_delay_rate": 0.18, "speed_factor": 1.05, "tier": "Standard"},
    "DTDC": {"weight": 0.12, "base_delay_rate": 0.25, "speed_factor": 0.95, "tier": "Economy"},
    "Xpressbees": {"weight": 0.08, "base_delay_rate": 0.23, "speed_factor": 0.98, "tier": "Standard"},
    "Shadowfax": {"weight": 0.05, "base_delay_rate": 0.26, "speed_factor": 0.94, "tier": "Hyperlocal/Surface"},
    "TCI Express": {"weight": 0.04, "base_delay_rate": 0.28, "speed_factor": 0.90, "tier": "Heavy Freight"},
    "SafeXpress": {"weight": 0.03, "base_delay_rate": 0.27, "speed_factor": 0.91, "tier": "B2B Freight"},
}

# Transport Mode Probabilities (Road dominates domestic logistics)
TRANSPORT_MODES: List[str] = ["Road", "Rail", "Air", "Sea"]
TRANSPORT_MODE_WEIGHTS: List[float] = [0.73, 0.14, 0.12, 0.01]

# Vehicle Types compatible with Transport Mode
VEHICLE_MODE_MAP: Dict[str, List[Tuple[str, float]]] = {
    "Road": [
        ("2-Wheeler (Hyperlocal)", 0.08),
        ("Small Commercial Van (Tata Ace)", 0.22),
        ("Light Commercial Vehicle (LCV)", 0.32),
        ("Heavy Commercial Vehicle (HCV)", 0.26),
        ("Multi-Axle Container Trailer", 0.12),
    ],
    "Rail": [
        ("Freight Express Parcel Van", 0.65),
        ("Dedicated Freight Container (CONCOR)", 0.35),
    ],
    "Air": [
        ("Commercial Belly Cargo", 0.70),
        ("Dedicated Cargo Aircraft (B737/B757)", 0.30),
    ],
    "Sea": [
        ("Coastal Feeder Vessel", 0.85),
        ("Ro-Ro Barge", 0.15),
    ],
}

# Product Categories and typical logistics traits
PRODUCT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Electronics": {"weight": 0.26, "density_factor": 1.4, "val_med": 14000, "val_sigma": 1.1, "fragile": True},
    "Apparel": {"weight": 0.24, "density_factor": 0.6, "val_med": 2200, "val_sigma": 0.8, "fragile": False},
    "Home & Kitchen": {"weight": 0.16, "density_factor": 0.8, "val_med": 3500, "val_sigma": 0.9, "fragile": False},
    "Healthcare": {"weight": 0.10, "density_factor": 1.2, "val_med": 2800, "val_sigma": 1.0, "fragile": True},
    "Automotive Parts": {"weight": 0.08, "density_factor": 2.2, "val_med": 7500, "val_sigma": 1.2, "fragile": False},
    "Books & Media": {"weight": 0.06, "density_factor": 1.8, "val_med": 850, "val_sigma": 0.6, "fragile": False},
    "Groceries & FMCG": {"weight": 0.06, "density_factor": 1.0, "val_med": 1200, "val_sigma": 0.7, "fragile": False},
    "Industrial Equipment": {"weight": 0.04, "density_factor": 3.0, "val_med": 48000, "val_sigma": 1.4, "fragile": False},
}

# Service Levels
SERVICE_LEVELS: List[str] = ["Same Day", "Next Day", "Express", "Standard", "Economy Surface"]
SERVICE_LEVEL_WEIGHTS: List[float] = [0.04, 0.16, 0.35, 0.35, 0.10]

# Priority Levels
PRIORITY_LEVELS: List[str] = ["Low", "Medium", "High", "Critical"]
PRIORITY_WEIGHTS: List[float] = [0.20, 0.50, 0.24, 0.06]

# Warehouse Types
WAREHOUSE_TYPES: List[str] = [
    "Local Fulfillment Center (FC)",
    "Regional Distribution Center (RDC)",
    "Mother Hub",
    "Sort Center (SC)",
]
WAREHOUSE_TYPE_WEIGHTS: List[float] = [0.35, 0.35, 0.18, 0.12]

# Route Types
ROUTE_TYPES: List[str] = [
    "Urban Last-Mile",
    "Inter-City Highway",
    "National Trunk Corridor",
    "Semi-Urban / Feeder",
    "Hilly / Remote Terrain",
]
ROUTE_TYPE_WEIGHTS: List[float] = [0.20, 0.40, 0.25, 0.11, 0.04]

# Customer Types
CUSTOMER_TYPES: List[str] = ["B2C", "D2C", "B2B", "Enterprise"]
CUSTOMER_TYPE_WEIGHTS: List[float] = [0.55, 0.25, 0.15, 0.05]

# Indian Holiday Calendar across 2023 - 2025 (Peak shopping and logistics surge dates)
INDIAN_HOLIDAYS: List[Tuple[str, str, str]] = [
    # (start_date, end_date, name)
    ("2023-01-26", "2023-01-26", "Republic Day"),
    ("2023-03-07", "2023-03-08", "Holi"),
    ("2023-04-22", "2023-04-22", "Eid-ul-Fitr"),
    ("2023-08-15", "2023-08-15", "Independence Day"),
    ("2023-08-30", "2023-08-30", "Raksha Bandhan"),
    ("2023-09-19", "2023-09-20", "Ganesh Chaturthi"),
    ("2023-10-23", "2023-10-24", "Dussehra"),
    ("2023-11-08", "2023-11-15", "Diwali Festival Week"),
    ("2023-12-24", "2023-12-26", "Christmas"),
    ("2024-01-26", "2024-01-26", "Republic Day"),
    ("2024-03-24", "2024-03-25", "Holi"),
    ("2024-04-11", "2024-04-11", "Eid-ul-Fitr"),
    ("2024-08-15", "2024-08-15", "Independence Day"),
    ("2024-08-19", "2024-08-19", "Raksha Bandhan"),
    ("2024-09-07", "2024-09-08", "Ganesh Chaturthi"),
    ("2024-10-11", "2024-10-12", "Dussehra"),
    ("2024-10-29", "2024-11-04", "Diwali Festival Week"),
    ("2024-12-24", "2024-12-26", "Christmas"),
    ("2025-01-26", "2025-01-26", "Republic Day"),
    ("2025-03-14", "2025-03-15", "Holi"),
    ("2025-03-31", "2025-03-31", "Eid-ul-Fitr"),
    ("2025-08-15", "2025-08-15", "Independence Day"),
    ("2025-09-02", "2025-09-03", "Ganesh Chaturthi"),
    ("2025-10-02", "2025-10-02", "Gandhi Jayanti / Dussehra"),
    ("2025-10-19", "2025-10-25", "Diwali Festival Week"),
    ("2025-12-24", "2025-12-26", "Christmas"),
]


# ==============================================================================
# GEOGRAPHICAL DISTANCE & NETWORK ROUTING CALCULATIONS
# ==============================================================================

def vectorized_haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Computes great-circle distances in kilometers between origin and destination
    coordinate arrays using the vectorized Haversine formula.
    """
    R = 6371.0  # Earth's mean radius in kilometers

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    # Clip to guard against floating-point inaccuracies
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# ==============================================================================
# SYNTHETIC DATASET GENERATOR ENGINE
# ==============================================================================

class LogisticsDatasetGenerator:
    """
    Production-grade synthetic dataset generator for ETAFlow.
    Leverages NumPy vectorized operations for scalable generation of 100k - 1M+ rows.
    """

    def __init__(
        self,
        n_samples: int = 100_000,
        seed: int = 42,
        start_date: str = "2023-01-01",
        end_date: str = "2025-12-31",
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        """Executes the full vectorized generation pipeline."""
        logger.info("Initializing generation of %d synthetic shipment records (seed=%d)...", self.n_samples, self.seed)

        # 1. Unique Identifiers
        shipment_ids = [f"SHP{i+1:08d}" for i in range(self.n_samples)]

        # 2. Temporal Scaffold
        df_temporal = self._generate_temporal_scaffold()

        # 3. Geography & Network Routing
        df_geo = self._generate_geography()

        # 4. Logistics Specifications & Modes
        df_logistics = self._generate_logistics_info(df_geo["distance_km"].values)

        # 5. Shipment Physical Characteristics
        df_shipment = self._generate_shipment_characteristics()

        # 6. Operational & Environmental Variables
        df_ops_env = self._generate_operational_and_environment(
            df_geo, df_temporal, df_logistics, df_shipment
        )

        # Combine intermediate features
        df = pd.concat([
            pd.Series(shipment_ids, name="shipment_id"),
            df_temporal,
            df_shipment,
            df_geo,
            df_logistics,
            df_ops_env,
        ], axis=1)

        # 7. Non-linear Target Generation (actual_delivery_days, promised_delivery_days, delay)
        df = self._compute_targets_and_timestamps(df)

        # 8. Inject Controlled Missing Values & Outliers
        df = self._inject_controlled_missingness(df)

        # 9. Enforce Column Order & Clean Formatting
        final_df = self._finalize_schema_order(df)

        logger.info("Generation complete. Total shape: %s", final_df.shape)
        return final_df

    def _generate_temporal_scaffold(self) -> pd.DataFrame:
        """Generates order dates and calendar features with Indian holiday matching."""
        start_sec = int(self.start_date.timestamp())
        end_sec = int(self.end_date.timestamp())

        # Sample uniform order timestamps
        random_timestamps = self.rng.integers(start_sec, end_sec, size=self.n_samples)
        order_dates = pd.to_datetime(random_timestamps, unit="s")

        # Temporal components
        day_of_week = order_dates.day_name()
        month = order_dates.month
        quarter = order_dates.quarter
        is_weekend = (order_dates.dayofweek >= 5).astype(int)

        # Vectorized holiday mapping
        is_holiday = np.zeros(self.n_samples, dtype=int)
        holiday_names = np.full(self.n_samples, None, dtype=object)

        order_date_str = order_dates.strftime("%Y-%m-%d")
        for h_start, h_end, h_name in INDIAN_HOLIDAYS:
            mask = (order_date_str >= h_start) & (order_date_str <= h_end)
            is_holiday[mask] = 1
            holiday_names[mask] = h_name

        return pd.DataFrame({
            "order_date": order_dates.strftime("%Y-%m-%d %H:%M:%S"),
            "day_of_week": day_of_week,
            "month": month,
            "quarter": quarter,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "holiday_name": holiday_names,
        })

    def _generate_geography(self) -> pd.DataFrame:
        """Generates realistic origin/destination hubs, coordinates, and road-circuity distance."""
        # Sample origin city
        orig_indices = self.rng.choice(len(CITIES), size=self.n_samples, p=CITY_WEIGHTS)
        orig_cities = [CITIES[i] for i in orig_indices]

        # Sample destination city (avoiding pure identity where possible, but allow ~5% intra-city)
        dest_indices = self.rng.choice(len(CITIES), size=self.n_samples, p=CITY_WEIGHTS)
        same_city_mask = (orig_indices == dest_indices) & (self.rng.random(self.n_samples) > 0.05)
        # Re-sample same cities to different cities
        dest_indices[same_city_mask] = (dest_indices[same_city_mask] + self.rng.integers(1, len(CITIES), size=same_city_mask.sum())) % len(CITIES)
        dest_cities = [CITIES[i] for i in dest_indices]

        orig_lats = np.array([CITY_REGISTRY[c]["lat"] for c in orig_cities])
        orig_lons = np.array([CITY_REGISTRY[c]["lon"] for c in orig_cities])
        orig_states = [CITY_REGISTRY[c]["state"] for c in orig_cities]
        fcs = [CITY_REGISTRY[c]["fc_id"] for c in orig_cities]

        dest_lats = np.array([CITY_REGISTRY[c]["lat"] for c in dest_cities])
        dest_lons = np.array([CITY_REGISTRY[c]["lon"] for c in dest_cities])
        dest_states = [CITY_REGISTRY[c]["state"] for c in dest_cities]
        dest_tiers = [CITY_REGISTRY[c]["tier"] for c in dest_cities]

        # Great-circle distance
        straight_distance = vectorized_haversine(orig_lats, orig_lons, dest_lats, dest_lons)

        # Indian highway network circuity factor: roads are ~1.22x to 1.38x straight line distance
        # Terrain adjustment: hilly northern and north-eastern corridors (e.g. Guwahati) have circuity up to 1.45x
        hilly_mask = np.isin(dest_cities, ["Guwahati", "Chandigarh"]) | np.isin(orig_cities, ["Guwahati", "Chandigarh"])
        circuity = self.rng.uniform(1.22, 1.36, size=self.n_samples)
        circuity[hilly_mask] += self.rng.uniform(0.08, 0.16, size=hilly_mask.sum())

        # For intra-city deliveries (straight distance 0), road distance is realistic local courier distance (15 - 45 km)
        is_intra_city = (straight_distance < 1.0)
        final_distance = np.where(
            is_intra_city,
            self.rng.uniform(15.0, 45.0, size=self.n_samples),
            straight_distance * circuity + self.rng.uniform(10.0, 30.0, size=self.n_samples)
        )
        final_distance = np.round(final_distance, 1)

        # Delivery zone based on destination city tier
        delivery_zones = []
        for tier in dest_tiers:
            if tier == "Metro":
                delivery_zones.append(self.rng.choice(["Metro", "Tier 1"], p=[0.85, 0.15]))
            elif tier == "Tier 1":
                delivery_zones.append(self.rng.choice(["Tier 1", "Tier 2"], p=[0.80, 0.20]))
            elif tier == "Tier 2":
                delivery_zones.append(self.rng.choice(["Tier 2", "Tier 3"], p=[0.75, 0.25]))
            else:
                delivery_zones.append(self.rng.choice(["Tier 3", "Rural / ODA"], p=[0.70, 0.30]))

        return pd.DataFrame({
            "origin_city": orig_cities,
            "origin_state": orig_states,
            "origin_latitude": np.round(orig_lats, 4),
            "origin_longitude": np.round(orig_lons, 4),
            "destination_city": dest_cities,
            "destination_state": dest_states,
            "destination_latitude": np.round(dest_lats, 4),
            "destination_longitude": np.round(dest_lons, 4),
            "distance_km": final_distance,
            "fulfillment_center": fcs,
            "delivery_zone": delivery_zones,
        })

    def _generate_logistics_info(self, distances: np.ndarray) -> pd.DataFrame:
        """Assigns transport mode, compatible vehicles, carriers, warehouse, and service levels."""
        # Transport mode assignment conditioned on distance:
        # Long distance (> 800 km) can use Rail or Air; Short distance (< 250 km) is 99% Road
        transport_modes = []
        vehicle_types = []

        # Vectorized choice for base modes
        carrier_names = list(CARRIER_PROFILES.keys())
        carrier_weights = [CARRIER_PROFILES[c]["weight"] for c in carrier_names]
        carriers = self.rng.choice(carrier_names, size=self.n_samples, p=carrier_weights)

        # Sample modes conditioned on distance
        rand_vals = self.rng.random(self.n_samples)
        for i in range(self.n_samples):
            dist = distances[i]
            if dist < 200:
                mode = "Road"
            elif dist < 600:
                mode = "Road" if rand_vals[i] < 0.88 else "Rail"
            else:
                if rand_vals[i] < 0.58:
                    mode = "Road"
                elif rand_vals[i] < 0.82:
                    mode = "Rail"
                elif rand_vals[i] < 0.985:
                    mode = "Air"
                else:
                    mode = "Sea"
            transport_modes.append(mode)

            # Assign compatible vehicle
            veh_choices, veh_weights = zip(*VEHICLE_MODE_MAP[mode])
            # If road and ultra short distance, prioritize smaller vans or 2-wheelers
            if mode == "Road" and dist < 50:
                v = self.rng.choice(veh_choices[:2], p=[0.40, 0.60])
            elif mode == "Road" and dist > 800:
                v = self.rng.choice(veh_choices[2:], p=[0.20, 0.50, 0.30])
            else:
                v = self.rng.choice(veh_choices, p=veh_weights)
            vehicle_types.append(v)

        service_levels = self.rng.choice(SERVICE_LEVELS, size=self.n_samples, p=SERVICE_LEVEL_WEIGHTS)
        route_types = self.rng.choice(ROUTE_TYPES, size=self.n_samples, p=ROUTE_TYPE_WEIGHTS)
        warehouse_types = self.rng.choice(WAREHOUSE_TYPES, size=self.n_samples, p=WAREHOUSE_TYPE_WEIGHTS)
        customer_types = self.rng.choice(CUSTOMER_TYPES, size=self.n_samples, p=CUSTOMER_TYPE_WEIGHTS)

        return pd.DataFrame({
            "transport_mode": transport_modes,
            "carrier": carriers,
            "vehicle_type": vehicle_types,
            "route_type": route_types,
            "warehouse_type": warehouse_types,
            "service_level": service_levels,
            "customer_type": customer_types,
        })

    def _generate_shipment_characteristics(self) -> pd.DataFrame:
        """Generates package weights, volumes, counts, categories, and financial values."""
        categories = list(PRODUCT_CATEGORIES.keys())
        cat_weights = [PRODUCT_CATEGORIES[c]["weight"] for c in categories]
        product_categories = self.rng.choice(categories, size=self.n_samples, p=cat_weights)

        # Log-normal package weight (median ~2.4 kg, 95% between 0.3 kg and 25 kg, heavy tail up to 120 kg)
        base_weights = self.rng.lognormal(mean=0.85, sigma=0.85, size=self.n_samples)
        # Adjust weight by category
        weight_adjusters = np.array([PRODUCT_CATEGORIES[c]["density_factor"] for c in product_categories])
        package_weights = np.clip(np.round(base_weights * weight_adjusters, 2), 0.15, 120.0)

        # Package count: mostly 1, occasionally multiple
        counts = self.rng.choice([1, 2, 3, 4, 5, 8], size=self.n_samples, p=[0.78, 0.13, 0.05, 0.025, 0.01, 0.005])

        # Volume: packing density in cm3 (1 kg ~ 2500 - 5000 cm3 for consumer goods)
        volume_cm3 = package_weights * self.rng.uniform(2800, 4500, size=self.n_samples) * (1.0 + 0.3 * (counts - 1))
        # Add slight non-linear shape noise
        volume_cm3 = np.clip(np.round(volume_cm3, 0), 500.0, 850000.0)

        # Priority level
        priority_levels = self.rng.choice(PRIORITY_LEVELS, size=self.n_samples, p=PRIORITY_WEIGHTS)

        # Shipment value INR: log-normal conditioned on category
        values = []
        for i in range(self.n_samples):
            cat = product_categories[i]
            med = PRODUCT_CATEGORIES[cat]["val_med"]
            sigma = PRODUCT_CATEGORIES[cat]["val_sigma"]
            val = self.rng.lognormal(mean=np.log(med), sigma=sigma)
            # Clip between 150 INR and 350,000 INR
            values.append(round(float(np.clip(val, 150.0, 350000.0)), 2))

        return pd.DataFrame({
            "package_weight_kg": package_weights,
            "package_volume_cm3": volume_cm3,
            "package_count": counts,
            "product_category": product_categories,
            "priority_level": priority_levels,
            "shipment_value_inr": values,
        })

    def _generate_operational_and_environment(
        self,
        df_geo: pd.DataFrame,
        df_temporal: pd.DataFrame,
        df_logistics: pd.DataFrame,
        df_shipment: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generates operational hours, traffic levels, weather risk, road quality, and historical rates."""
        months = df_temporal["month"].values
        is_holiday = df_temporal["is_holiday"].values
        is_weekend = df_temporal["is_weekend"].values
        distances = df_geo["distance_km"].values
        dest_zones = df_geo["delivery_zone"].values
        route_types = df_logistics["route_type"].values
        carriers = df_logistics["carrier"].values

        # 1. Traffic Level & Congestion Index
        # Traffic is elevated during holidays, weekends, metro zones, and peak months (Oct-Dec)
        traffic_scores = self.rng.uniform(0.15, 0.65, size=self.n_samples)
        traffic_scores += np.where(df_geo["origin_city"].isin(["Mumbai", "Delhi", "Bengaluru"]), 0.15, 0.0)
        traffic_scores += np.where(is_holiday == 1, 0.20, 0.0)
        traffic_scores += np.where(np.isin(months, [10, 11, 12]), 0.10, 0.0)
        traffic_scores += np.where(route_types == "Urban Last-Mile", 0.12, 0.0)
        congestion_index = np.clip(np.round(traffic_scores, 3), 0.05, 0.98)

        traffic_levels = []
        for score in congestion_index:
            if score < 0.35:
                traffic_levels.append("Low")
            elif score < 0.60:
                traffic_levels.append("Moderate")
            elif score < 0.82:
                traffic_levels.append("High")
            else:
                traffic_levels.append("Severe")

        # 2. Weather Condition & Weather Risk Score
        # Indian monsoon peaks June to September; winter fog in North India in Dec-Jan
        is_monsoon = np.isin(months, [6, 7, 8, 9])
        is_winter = np.isin(months, [12, 1])
        is_north = df_geo["destination_state"].isin(["Delhi", "Uttar Pradesh", "Punjab", "Chandigarh", "Rajasthan"])

        weather_conditions = []
        weather_risks = []
        for i in range(self.n_samples):
            r = self.rng.random()
            if is_monsoon[i]:
                if r < 0.38:
                    cond, risk = "Rainy", self.rng.uniform(0.35, 0.65)
                elif r < 0.62:
                    cond, risk = "Heavy Rain / Storm", self.rng.uniform(0.65, 0.92)
                elif r < 0.66:
                    cond, risk = "Cyclonic / Gale", self.rng.uniform(0.85, 0.99)
                else:
                    cond, risk = "Clear", self.rng.uniform(0.05, 0.25)
            elif is_winter[i] and is_north[i]:
                if r < 0.45:
                    cond, risk = "Fog / Low Visibility", self.rng.uniform(0.50, 0.85)
                elif r < 0.55:
                    cond, risk = "Rainy", self.rng.uniform(0.30, 0.55)
                else:
                    cond, risk = "Clear", self.rng.uniform(0.05, 0.20)
            elif months[i] in [4, 5]:  # Summer heatwave
                if r < 0.30:
                    cond, risk = "Heatwave", self.rng.uniform(0.25, 0.50)
                else:
                    cond, risk = "Clear", self.rng.uniform(0.02, 0.18)
            else:
                if r < 0.12:
                    cond, risk = "Rainy", self.rng.uniform(0.20, 0.45)
                else:
                    cond, risk = "Clear", self.rng.uniform(0.01, 0.15)
            weather_conditions.append(cond)
            weather_risks.append(round(risk, 3))

        # 3. Road Condition
        # Poor during monsoons or in rural/hilly terrain
        road_conditions = []
        for i in range(self.n_samples):
            rt = route_types[i]
            mon = is_monsoon[i]
            r = self.rng.random()
            if rt == "National Trunk Corridor":
                road = "Good" if r < 0.78 else ("Average" if r < 0.94 else "Under Construction")
            elif rt == "Hilly / Remote Terrain":
                road = "Poor" if mon or r < 0.50 else "Average"
            elif rt == "Rural Access":
                road = "Poor" if mon or r < 0.40 else "Average"
            else:
                road = "Good" if r < 0.60 else ("Average" if r < 0.88 else "Poor")
            road_conditions.append(road)

        # 4. Warehouse Processing, Loading, Handling, Stops
        # Warehouse processing hours: base 3 to 14 hrs + volume/item count + holiday load spike (up to 1.5x)
        base_wh = self.rng.lognormal(mean=1.8, sigma=0.45, size=self.n_samples)
        # Holiday bottleneck multiplier
        holiday_wh_mult = np.where(is_holiday == 1, self.rng.uniform(1.3, 1.8, size=self.n_samples), 1.0)
        # Volume/Count effect
        item_wh_mult = 1.0 + 0.08 * (df_shipment["package_count"].values - 1)
        wh_hours = np.clip(np.round(base_wh * holiday_wh_mult * item_wh_mult, 1), 1.5, 48.0)

        # Loading time (hours): depends on vehicle type and package weight
        base_loading = self.rng.uniform(0.4, 2.2, size=self.n_samples)
        is_heavy_veh = np.isin(df_logistics["vehicle_type"].values, [
            "Heavy Commercial Vehicle (HCV)", "Multi-Axle Container Trailer",
            "Dedicated Freight Container (CONCOR)", "Dedicated Cargo Aircraft (B737/B757)"
        ])
        loading_hours = np.where(is_heavy_veh, base_loading + self.rng.uniform(1.0, 2.8, size=self.n_samples), base_loading)
        loading_hours = np.clip(np.round(loading_hours, 1), 0.2, 8.0)

        # Handling time (hours): fragile items or high priority take careful handling
        base_handling = self.rng.uniform(0.2, 1.5, size=self.n_samples)
        is_fragile = df_shipment["product_category"].isin(["Electronics", "Healthcare"])
        handling_hours = np.where(is_fragile, base_handling + self.rng.uniform(0.4, 1.2, size=self.n_samples), base_handling)
        handling_hours = np.clip(np.round(handling_hours, 1), 0.1, 5.0)

        # Customs Clearance Hours: 0.0 for domestic (~98.5%), > 0 for international / SEZ freight (~1.5%)
        is_customs_lane = (self.rng.random(self.n_samples) < 0.015)
        customs_hours = np.where(is_customs_lane, np.round(self.rng.uniform(12.0, 48.0, size=self.n_samples), 1), 0.0)

        # Number of intermediate stops: 0 for direct point-to-point / air; 1 - 5 for hub-and-spoke
        base_stops = np.floor(distances / 350.0).astype(int)
        stops = np.clip(base_stops + self.rng.integers(0, 3, size=self.n_samples), 0, 7)
        # Air and Same Day usually direct (0 stops)
        is_air_or_sameday = (df_logistics["transport_mode"] == "Air") | (df_logistics["service_level"] == "Same Day")
        stops = np.where(is_air_or_sameday, np.clip(stops, 0, 1), stops)

        # Route Complexity Score: 1.0 to 5.0 continuous
        complexity = 1.2 + (distances / 1200.0) * 0.8 + (stops * 0.35)
        complexity += np.where(route_types == "Hilly / Remote Terrain", 1.4, 0.0)
        complexity += np.where(route_types == "Urban Last-Mile", 0.5, 0.0)
        complexity += self.rng.normal(0.0, 0.25, size=self.n_samples)
        complexity_score = np.clip(np.round(complexity, 2), 1.0, 5.0)

        # 5. Historical / Prior Baseline Delay Rates (Cold Features known prior to delivery)
        carrier_hist_rate = np.array([CARRIER_PROFILES[c]["base_delay_rate"] for c in carriers])
        carrier_hist_rate = np.clip(carrier_hist_rate + self.rng.normal(0.0, 0.025, size=self.n_samples), 0.08, 0.40)

        route_hist_rate = 0.12 + (distances / 2500.0) * 0.15 + (complexity_score / 5.0) * 0.10
        route_hist_rate = np.clip(route_hist_rate + self.rng.normal(0.0, 0.03, size=self.n_samples), 0.06, 0.45)

        wh_hist_rate = 0.14 + (wh_hours / 48.0) * 0.12 + self.rng.normal(0.0, 0.02, size=self.n_samples)
        wh_hist_rate = np.clip(wh_hist_rate, 0.05, 0.38)

        dest_hist_rate = np.where(
            dest_zones == "Metro", 0.16,
            np.where(dest_zones == "Tier 1", 0.20,
            np.where(dest_zones == "Tier 2", 0.24, 0.32))
        ) + self.rng.normal(0.0, 0.025, size=self.n_samples)
        dest_hist_rate = np.clip(dest_hist_rate, 0.07, 0.45)

        return pd.DataFrame({
            "warehouse_processing_hours": wh_hours,
            "loading_time_hours": loading_hours,
            "handling_time_hours": handling_hours,
            "customs_clearance_hours": customs_hours,
            "number_of_stops": stops,
            "route_complexity_score": complexity_score,
            "traffic_level": traffic_levels,
            "weather_condition": weather_conditions,
            "weather_risk_score": weather_risks,
            "road_condition": road_conditions,
            "congestion_index": congestion_index,
            "carrier_historical_delay_rate": np.round(carrier_hist_rate, 3),
            "route_historical_delay_rate": np.round(route_hist_rate, 3),
            "warehouse_historical_delay_rate": np.round(wh_hist_rate, 3),
            "destination_historical_delay_rate": np.round(dest_hist_rate, 3),
        })

    def _compute_targets_and_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes realistic non-linear transit times, promised SLA days, actual delivery days,
        and derives mathematically consistent delay targets and timestamps.
        """
        dist = df["distance_km"].values
        mode = df["transport_mode"].values
        carrier = df["carrier"].values
        service = df["service_level"].values
        priority = df["priority_level"].values
        traffic = df["traffic_level"].values
        congestion = df["congestion_index"].values
        weather = df["weather_condition"].values
        weather_risk = df["weather_risk_score"].values
        road = df["road_condition"].values
        stops = df["number_of_stops"].values
        complexity = df["route_complexity_score"].values
        is_holiday = df["is_holiday"].values

        # 1. Base Transit Speed & Hours by Mode
        # Average effective operational transit speed (including driver rest / tolls / rail shunting / air ground turnaround)
        base_speed = np.where(
            mode == "Air", 420.0,
            np.where(mode == "Rail", 42.0,
            np.where(mode == "Sea", 22.0, 48.0))  # Road base = 48 km/h on mixed Indian highways
        )

        # Carrier performance multiplier (BlueDart is faster, budget freight is slower)
        carrier_speed_mult = np.array([CARRIER_PROFILES[c]["speed_factor"] for c in carrier])

        # Terminal ground turnaround / sorting buffer by mode
        terminal_hours = np.where(
            mode == "Air", 5.0,
            np.where(mode == "Rail", 8.0,
            np.where(mode == "Sea", 18.0, 1.5))
        )

        base_transit_hours = (dist / (base_speed * carrier_speed_mult)) + terminal_hours

        # 2. Non-linear Multipliers
        # Traffic Multiplier
        traffic_mult = 1.0 + np.where(
            traffic == "Severe", 0.70 + 0.35 * (congestion ** 2),
            np.where(traffic == "High", 0.35 + 0.15 * congestion,
            np.where(traffic == "Moderate", 0.12, 0.0))
        )
        # Air is mostly immune to road highway traffic (only affects origin/dest ground transfers)
        traffic_mult = np.where(mode == "Air", 1.0 + 0.15 * (traffic_mult - 1.0), traffic_mult)

        # Weather Multiplier with non-linear severe storm threshold
        weather_mult = 1.0 + np.where(
            weather == "Cyclonic / Gale", 1.20 + 0.50 * weather_risk,
            np.where(weather == "Heavy Rain / Storm", 0.55 + 0.30 * weather_risk,
            np.where(weather == "Fog / Low Visibility", 0.40 + 0.20 * weather_risk,
            np.where(weather == "Rainy", 0.15, 0.0)))
        )
        # Fog and storms heavily affect Air flight schedules
        weather_mult = np.where((mode == "Air") & (weather == "Fog / Low Visibility"), 1.65, weather_mult)

        # Road Condition Multiplier
        road_mult = 1.0 + np.where(
            road == "Under Construction", 0.35,
            np.where(road == "Poor", 0.22,
            np.where(road == "Average", 0.08, 0.0))
        )
        road_mult = np.where(np.isin(mode, ["Air", "Rail", "Sea"]), 1.0, road_mult)

        # Non-linear Interaction Effects:
        # A) Traffic * Weather interaction (Severe rain during peak traffic causes disproportionate gridlock)
        traffic_weather_interaction = 1.0 + 0.30 * (congestion * weather_risk)

        # B) Route Complexity * Road Condition interaction
        complexity_road_interaction = 1.0 + 0.05 * (complexity * (road_mult - 1.0))

        # C) Stops delay buffer (each stop adds 1.2 to 2.5 hours of unloading/sorting)
        stops_delay_hours = stops * self.rng.uniform(1.2, 2.2, size=self.n_samples)

        # D) Stochastic noise and operational shock (heavy-tailed log-normal for rare mechanical/toll breakdowns)
        # 94% normal operations, 6% operational shock
        is_shock = (self.rng.random(self.n_samples) < 0.065)
        shock_hours = np.where(is_shock, self.rng.lognormal(mean=2.4, sigma=0.6, size=self.n_samples), 0.0)
        regular_noise = self.rng.normal(0.0, 1.2, size=self.n_samples)

        # Effective transit hours
        effective_transit_hours = (
            base_transit_hours
            * traffic_mult
            * weather_mult
            * road_mult
            * traffic_weather_interaction
            * complexity_road_interaction
            + stops_delay_hours
            + shock_hours
            + regular_noise
        )
        effective_transit_hours = np.clip(effective_transit_hours, 1.5, 360.0)

        # Total elapsed duration from order placement to delivery
        wh_hours = df["warehouse_processing_hours"].values
        loading_hours = df["loading_time_hours"].values
        handling_hours = df["handling_time_hours"].values
        customs_hours = df["customs_clearance_hours"].values

        total_delivery_hours = wh_hours + loading_hours + handling_hours + customs_hours + effective_transit_hours
        actual_delivery_days = np.clip(np.round(total_delivery_hours / 24.0, 2), 0.25, 30.0)

        # 3. Promised Delivery Days (Customer SLA Policy agreed at booking)
        # Commercial SLA matrix based on service_level and distance tier
        # Same Day: 0.5 to 1.0 days
        # Next Day: 1.0 to 1.5 days
        # Express: 2.0 to 3.5 days
        # Standard: 3.5 to 6.0 days
        # Economy: 5.0 to 9.0 days
        promised_sla_days = np.zeros(self.n_samples, dtype=float)
        for i in range(self.n_samples):
            srv = service[i]
            d = dist[i]
            prio = priority[i]

            if srv == "Same Day":
                base_sla = 0.75
            elif srv == "Next Day":
                base_sla = 1.0 if d < 500 else 1.5
            elif srv == "Express":
                base_sla = 1.5 + (d / 800.0) * 0.8
            elif srv == "Standard":
                base_sla = 2.5 + (d / 500.0) * 0.9
            else:  # Economy Surface
                base_sla = 4.0 + (d / 400.0) * 1.1

            # Priority SLA tightness: Critical orders are promised tighter deadlines
            if prio == "Critical":
                base_sla = max(0.5, base_sla * 0.85)
            elif prio == "Low":
                base_sla = base_sla * 1.15

            promised_sla_days[i] = round(base_sla, 1)

        # Ensure promised days is at least 0.5
        promised_delivery_days = np.clip(promised_sla_days, 0.5, 25.0)

        # 4. Strict Mathematical Target Coupling
        # delivery_delay_days = max(actual_delivery_days - promised_delivery_days, 0)
        # is_delayed = 1 if delivery_delay_days > 0 else 0
        raw_diff = actual_delivery_days - promised_delivery_days
        # If diff is tiny (e.g. within 0.02 days / ~30 mins), treat as on-time to avoid micro-second noise
        delivery_delay_days = np.where(raw_diff > 0.01, np.round(raw_diff, 2), 0.0)
        is_delayed = (delivery_delay_days > 0.0).astype(int)

        # 5. Chronologically Consistent Datetime Timestamps
        # order_date <= pickup_datetime <= actual_dispatch_datetime <= delivery_datetime
        order_dates = pd.to_datetime(df["order_date"])

        # Pickup occurs 1 to 12 hours after order
        pickup_lags_min = self.rng.integers(30, 480, size=self.n_samples)
        pickup_datetimes = order_dates + pd.to_timedelta(pickup_lags_min, unit="m")

        # Estimated dispatch: promised warehouse SLA (order + standard 6-12 hours)
        est_dispatch_datetimes = pickup_datetimes + pd.to_timedelta(self.rng.integers(4, 14, size=self.n_samples), unit="h")

        # Actual dispatch: pickup + warehouse_processing + loading
        dispatch_duration_h = wh_hours + loading_hours
        actual_dispatch_datetimes = pickup_datetimes + pd.to_timedelta(dispatch_duration_h, unit="h")

        # Delivery: actual dispatch + transit + handling + customs
        delivery_datetimes = actual_dispatch_datetimes + pd.to_timedelta(
            effective_transit_hours + handling_hours + customs_hours, unit="h"
        )

        df["pickup_datetime"] = pickup_datetimes.dt.strftime("%Y-%m-%d %H:%M:%S")
        df["estimated_dispatch_datetime"] = est_dispatch_datetimes.dt.strftime("%Y-%m-%d %H:%M:%S")
        df["actual_dispatch_datetime"] = actual_dispatch_datetimes.dt.strftime("%Y-%m-%d %H:%M:%S")
        df["delivery_datetime"] = delivery_datetimes.dt.strftime("%Y-%m-%d %H:%M:%S")

        df["promised_delivery_days"] = promised_delivery_days
        df["actual_delivery_days"] = actual_delivery_days
        df["delivery_delay_days"] = delivery_delay_days
        df["is_delayed"] = is_delayed

        return df

    def _inject_controlled_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Injects realistic, controlled missing data into non-target fields.
        Targets, shipment_ids, and primary routing attributes remain non-null.
        """
        logger.info("Injecting controlled sensor & telematics missingness...")
        n = len(df)

        # Weather telematics dropout (~2.2% in remote / tier-3 areas)
        weather_null_mask = self.rng.random(n) < 0.024
        df.loc[weather_null_mask, "weather_condition"] = np.nan
        df.loc[weather_null_mask, "weather_risk_score"] = np.nan

        # Traffic GPS telematics dropout (~1.8%)
        traffic_null_mask = self.rng.random(n) < 0.018
        df.loc[traffic_null_mask, "traffic_level"] = np.nan
        df.loc[traffic_null_mask, "congestion_index"] = np.nan

        # Cold-start / new carrier rate dropout (~1.5%)
        carrier_hist_null_mask = self.rng.random(n) < 0.015
        df.loc[carrier_hist_null_mask, "carrier_historical_delay_rate"] = np.nan

        # New route corridor rate dropout (~1.8%)
        route_hist_null_mask = self.rng.random(n) < 0.018
        df.loc[route_hist_null_mask, "route_historical_delay_rate"] = np.nan

        # Road condition reporting dropout in rural zones (~1.2%)
        road_null_mask = self.rng.random(n) < 0.012
        df.loc[road_null_mask, "road_condition"] = np.nan

        return df

    def _finalize_schema_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforces clean, logical column ordering as defined in ETAFlow specification."""
        column_order = [
            # 1. Identifiers
            "shipment_id",
            # 2. Temporal Features
            "order_date",
            "pickup_datetime",
            "estimated_dispatch_datetime",
            "actual_dispatch_datetime",
            "delivery_datetime",
            "day_of_week",
            "month",
            "quarter",
            "is_weekend",
            "is_holiday",
            "holiday_name",
            # 3. Shipment Characteristics
            "package_weight_kg",
            "package_volume_cm3",
            "package_count",
            "product_category",
            "priority_level",
            "shipment_value_inr",
            # 4. Geography & Routing
            "origin_city",
            "origin_state",
            "origin_latitude",
            "origin_longitude",
            "destination_city",
            "destination_state",
            "destination_latitude",
            "destination_longitude",
            "distance_km",
            "delivery_zone",
            "fulfillment_center",
            # 5. Logistics & Fleet
            "transport_mode",
            "carrier",
            "vehicle_type",
            "route_type",
            "warehouse_type",
            "service_level",
            "customer_type",
            # 6. Operational Variables
            "warehouse_processing_hours",
            "loading_time_hours",
            "handling_time_hours",
            "customs_clearance_hours",
            "number_of_stops",
            "route_complexity_score",
            # 7. Traffic & Environment
            "traffic_level",
            "congestion_index",
            "weather_condition",
            "weather_risk_score",
            "road_condition",
            # 8. Historical Baseline Priors
            "carrier_historical_delay_rate",
            "route_historical_delay_rate",
            "warehouse_historical_delay_rate",
            "destination_historical_delay_rate",
            # 9. Targets & Outcomes
            "promised_delivery_days",
            "actual_delivery_days",
            "delivery_delay_days",
            "is_delayed",
        ]
        return df[column_order]


# ==============================================================================
# DATA QUALITY VALIDATION SUITE
# ==============================================================================

def validate_dataset(df: pd.DataFrame, expected_samples: int) -> Dict[str, Any]:
    """
    Performs rigorous data validation against business rules and integrity constraints.
    Raises ValueError if any critical constraint is violated.
    """
    logger.info("Executing automated dataset validation checks...")

    # 1. Row count validation
    if len(df) != expected_samples:
        raise ValueError(f"Validation Error: Expected {expected_samples} rows, found {len(df)}.")

    # 2. Primary key uniqueness
    if df["shipment_id"].duplicated().any():
        dups = df["shipment_id"].duplicated().sum()
        raise ValueError(f"Validation Error: Found {dups} duplicate shipment_id values.")

    # 3. Critical non-null targets & identifiers
    critical_cols = [
        "shipment_id", "order_date", "pickup_datetime", "actual_dispatch_datetime",
        "delivery_datetime", "distance_km", "transport_mode", "carrier",
        "promised_delivery_days", "actual_delivery_days", "delivery_delay_days", "is_delayed"
    ]
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Validation Error: Column '{col}' contains {null_count} unexpected null values.")

    # 4. Strict target consistency
    # delivery_delay_days must equal max(actual_delivery_days - promised_delivery_days, 0)
    expected_delay = np.maximum(np.round(df["actual_delivery_days"] - df["promised_delivery_days"], 2), 0.0)
    delay_diff = np.abs(df["delivery_delay_days"] - expected_delay)
    if (delay_diff > 0.02).any():
        inconsistent = (delay_diff > 0.02).sum()
        raise ValueError(f"Validation Error: {inconsistent} rows have inconsistent delivery_delay_days targets.")

    # is_delayed must equal (delivery_delay_days > 0).astype(int)
    expected_is_delayed = (df["delivery_delay_days"] > 0.0).astype(int)
    if not (df["is_delayed"] == expected_is_delayed).all():
        mismatch = (df["is_delayed"] != expected_is_delayed).sum()
        raise ValueError(f"Validation Error: {mismatch} rows have mismatch between delivery_delay_days and is_delayed.")

    # 5. Non-negativity and positive bounds
    if (df["package_weight_kg"] <= 0).any():
        raise ValueError("Validation Error: package_weight_kg contains non-positive values.")
    if (df["package_volume_cm3"] <= 0).any():
        raise ValueError("Validation Error: package_volume_cm3 contains non-positive values.")
    if (df["distance_km"] <= 0).any():
        raise ValueError("Validation Error: distance_km contains non-positive values.")
    if (df["actual_delivery_days"] <= 0).any():
        raise ValueError("Validation Error: actual_delivery_days contains non-positive values.")
    if (df["promised_delivery_days"] <= 0).any():
        raise ValueError("Validation Error: promised_delivery_days contains non-positive values.")
    if (df["delivery_delay_days"] < 0).any():
        raise ValueError("Validation Error: delivery_delay_days contains negative values.")

    # 6. Valid binary classification domain
    unique_delayed = set(df["is_delayed"].unique())
    if not unique_delayed.issubset({0, 1}):
        raise ValueError(f"Validation Error: is_delayed contains non-binary values: {unique_delayed}")

    # 7. Realistic Class Balance Check (Expect 20% to 35% delayed)
    delay_rate = float(df["is_delayed"].mean()) * 100.0
    if not (18.0 <= delay_rate <= 38.0):
        logger.warning("Caution: Delay rate is %.2f%%, outside typical 20-35%% window.", delay_rate)

    # 8. Chronological integrity checks
    order_dt = pd.to_datetime(df["order_date"])
    pickup_dt = pd.to_datetime(df["pickup_datetime"])
    dispatch_dt = pd.to_datetime(df["actual_dispatch_datetime"])
    delivery_dt = pd.to_datetime(df["delivery_datetime"])

    if (pickup_dt < order_dt).any():
        raise ValueError("Validation Error: pickup_datetime precedes order_date in some records.")
    if (dispatch_dt < pickup_dt).any():
        raise ValueError("Validation Error: actual_dispatch_datetime precedes pickup_datetime in some records.")
    if (delivery_dt < dispatch_dt).any():
        raise ValueError("Validation Error: delivery_datetime precedes actual_dispatch_datetime in some records.")

    # 9. Geographical Coordinate Bounding Box (India: Lat 8 - 37 N, Lon 68 - 98 E)
    if not df["origin_latitude"].between(8.0, 37.0).all() or not df["destination_latitude"].between(8.0, 37.0).all():
        raise ValueError("Validation Error: Latitude values outside valid Indian subcontinent range.")
    if not df["origin_longitude"].between(68.0, 98.0).all() or not df["destination_longitude"].between(68.0, 98.0).all():
        raise ValueError("Validation Error: Longitude values outside valid Indian subcontinent range.")

    logger.info("All validation assertions passed successfully.")
    return {
        "total_rows": len(df),
        "unique_shipments": int(df["shipment_id"].nunique()),
        "delayed_shipments": int(df["is_delayed"].sum()),
        "delay_percentage": round(delay_rate, 2),
        "mean_delivery_days": round(float(df["actual_delivery_days"].mean()), 2),
        "median_delivery_days": round(float(df["actual_delivery_days"].median()), 2),
        "mean_delay_days": round(float(df["delivery_delay_days"].mean()), 2),
        "max_delay_days": round(float(df["delivery_delay_days"].max()), 2),
    }


# ==============================================================================
# DATA DICTIONARY GENERATION
# ==============================================================================

def create_data_dictionary() -> pd.DataFrame:
    """
    Constructs the comprehensive data dictionary matching ETAFlow standards.
    """
    dict_records = [
        {
            "column_name": "shipment_id",
            "data_type": "string",
            "description": "Unique alphanumeric tracking identifier for each shipment",
            "unit": "None",
            "role": "identifier",
            "allowed_values": "SHP00000001 to SHP99999999",
            "nullable": "false",
            "generation_logic": "Zero-padded sequential identifier",
        },
        {
            "column_name": "order_date",
            "data_type": "datetime",
            "description": "Timestamp when customer placed the order",
            "unit": "YYYY-MM-DD HH:MM:SS",
            "role": "feature",
            "allowed_values": "Configurable date range (2023-01-01 to 2025-12-31)",
            "nullable": "false",
            "generation_logic": "Uniform stochastic sample across configured date interval",
        },
        {
            "column_name": "pickup_datetime",
            "data_type": "datetime",
            "description": "Timestamp when carrier picked up the package from origin facility",
            "unit": "YYYY-MM-DD HH:MM:SS",
            "role": "feature",
            "allowed_values": "order_date + [30 mins, 8 hours]",
            "nullable": "false",
            "generation_logic": "order_date plus warehouse intake lag",
        },
        {
            "column_name": "estimated_dispatch_datetime",
            "data_type": "datetime",
            "description": "Target dispatch milestone scheduled by facility SLA",
            "unit": "YYYY-MM-DD HH:MM:SS",
            "role": "feature",
            "allowed_values": "pickup_datetime + [4, 14] hours",
            "nullable": "false",
            "generation_logic": "SLA dispatch commitment",
        },
        {
            "column_name": "actual_dispatch_datetime",
            "data_type": "datetime",
            "description": "Timestamp when package physically departed origin warehouse",
            "unit": "YYYY-MM-DD HH:MM:SS",
            "role": "feature",
            "allowed_values": "pickup_datetime + warehouse_processing + loading_time",
            "nullable": "false",
            "generation_logic": "pickup_datetime plus processing and loading duration",
        },
        {
            "column_name": "delivery_datetime",
            "data_type": "datetime",
            "description": "Timestamp when package was successfully delivered to consignee",
            "unit": "YYYY-MM-DD HH:MM:SS",
            "role": "feature / metadata",
            "allowed_values": "actual_dispatch_datetime + transit_hours + handling",
            "nullable": "false",
            "generation_logic": "Physical transit arrival milestone",
        },
        {
            "column_name": "day_of_week",
            "data_type": "string",
            "description": "Day of week when order was initiated",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday",
            "nullable": "false",
            "generation_logic": "Extracted from order_date",
        },
        {
            "column_name": "month",
            "data_type": "integer",
            "description": "Calendar month of order",
            "unit": "1 to 12",
            "role": "feature",
            "allowed_values": "1 to 12",
            "nullable": "false",
            "generation_logic": "Extracted from order_date",
        },
        {
            "column_name": "quarter",
            "data_type": "integer",
            "description": "Financial / calendar quarter of order",
            "unit": "1 to 4",
            "role": "feature",
            "allowed_values": "1 to 4",
            "nullable": "false",
            "generation_logic": "Extracted from order_date",
        },
        {
            "column_name": "is_weekend",
            "data_type": "integer",
            "description": "Binary indicator if order was placed on Saturday or Sunday",
            "unit": "Binary [0, 1]",
            "role": "feature",
            "allowed_values": "0, 1",
            "nullable": "false",
            "generation_logic": "Derived from order_date day of week",
        },
        {
            "column_name": "is_holiday",
            "data_type": "integer",
            "description": "Binary indicator if order date falls within a major Indian holiday window",
            "unit": "Binary [0, 1]",
            "role": "feature",
            "allowed_values": "0, 1",
            "nullable": "false",
            "generation_logic": "Matched against Indian holiday calendar (Diwali, Holi, Dussehra, etc.)",
        },
        {
            "column_name": "holiday_name",
            "data_type": "string",
            "description": "Name of the national/cultural holiday if applicable",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Diwali Festival Week, Holi, Dussehra, Republic Day, Independence Day, etc.",
            "nullable": "true",
            "generation_logic": "Lookup from Indian holiday schedule; null for standard non-holiday days",
        },
        {
            "column_name": "package_weight_kg",
            "data_type": "float",
            "description": "Gross chargeable weight of shipment in kilograms",
            "unit": "kg",
            "role": "feature",
            "allowed_values": "0.15 to 120.0",
            "nullable": "false",
            "generation_logic": "Log-normal distribution adjusted by product category density factor",
        },
        {
            "column_name": "package_volume_cm3",
            "data_type": "float",
            "description": "External cubic dimension volume of package in cubic centimeters",
            "unit": "cm³",
            "role": "feature",
            "allowed_values": "500.0 to 850000.0",
            "nullable": "false",
            "generation_logic": "Correlated with weight, item count, and category bulkiness",
        },
        {
            "column_name": "package_count",
            "data_type": "integer",
            "description": "Total number of individual items/cartons in consignment",
            "unit": "count",
            "role": "feature",
            "allowed_values": "1 to 8",
            "nullable": "false",
            "generation_logic": "Categorical discrete distribution (mostly 1, long tail up to 8)",
        },
        {
            "column_name": "product_category",
            "data_type": "string",
            "description": "Merchandise classification",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Electronics, Apparel, Home & Kitchen, Healthcare, Automotive Parts, Books & Media, Groceries & FMCG, Industrial Equipment",
            "nullable": "false",
            "generation_logic": "Market-calibrated categorical distribution",
        },
        {
            "column_name": "priority_level",
            "data_type": "string",
            "description": "Consignment priority tier specified by customer or shipper",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Low, Medium, High, Critical",
            "nullable": "false",
            "generation_logic": "Categorical distribution calibrated to e-commerce/freight tiers",
        },
        {
            "column_name": "shipment_value_inr",
            "data_type": "float",
            "description": "Declared commercial invoice value in Indian Rupees (INR)",
            "unit": "INR (₹)",
            "role": "feature",
            "allowed_values": "150.0 to 350000.0",
            "nullable": "false",
            "generation_logic": "Log-normal distribution conditioned on product category median",
        },
        {
            "column_name": "origin_city",
            "data_type": "string",
            "description": "Dispatch logistics hub city name",
            "unit": "None",
            "role": "feature",
            "allowed_values": "28 major Indian commercial hubs (Mumbai, Delhi, Bengaluru, Pune, etc.)",
            "nullable": "false",
            "generation_logic": "Weighted sampling reflecting metro freight outbound generation",
        },
        {
            "column_name": "origin_state",
            "data_type": "string",
            "description": "State or Union Territory of dispatch hub",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Maharashtra, Delhi, Karnataka, Telangana, Tamil Nadu, etc.",
            "nullable": "false",
            "generation_logic": "Deterministic lookup from origin_city",
        },
        {
            "column_name": "origin_latitude",
            "data_type": "float",
            "description": "Geographic latitude coordinate of dispatch hub",
            "unit": "Degrees North",
            "role": "feature",
            "allowed_values": "8.0 to 37.0",
            "nullable": "false",
            "generation_logic": "Deterministic lookup from city registry",
        },
        {
            "column_name": "origin_longitude",
            "data_type": "float",
            "description": "Geographic longitude coordinate of dispatch hub",
            "unit": "Degrees East",
            "role": "feature",
            "allowed_values": "68.0 to 98.0",
            "nullable": "false",
            "generation_logic": "Deterministic lookup from city registry",
        },
        {
            "column_name": "destination_city",
            "data_type": "string",
            "description": "Consignee delivery destination hub city name",
            "unit": "None",
            "role": "feature",
            "allowed_values": "28 major Indian commercial hubs",
            "nullable": "false",
            "generation_logic": "Weighted sampling with realistic inter-city freight demand patterns",
        },
        {
            "column_name": "destination_state",
            "data_type": "string",
            "description": "State or Union Territory of delivery destination",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Maharashtra, Delhi, Karnataka, Telangana, Tamil Nadu, etc.",
            "nullable": "false",
            "generation_logic": "Deterministic lookup from destination_city",
        },
        {
            "column_name": "destination_latitude",
            "data_type": "float",
            "description": "Geographic latitude coordinate of delivery destination",
            "unit": "Degrees North",
            "role": "feature",
            "allowed_values": "8.0 to 37.0",
            "nullable": "false",
            "generation_logic": "Deterministic lookup from city registry",
        },
        {
            "column_name": "destination_longitude",
            "data_type": "float",
            "description": "Geographic longitude coordinate of delivery destination",
            "unit": "Degrees East",
            "role": "feature",
            "allowed_values": "68.0 to 98.0",
            "nullable": "false",
            "generation_logic": "Deterministic lookup from city registry",
        },
        {
            "column_name": "distance_km",
            "data_type": "float",
            "description": "Estimated road/transit route distance in kilometers",
            "unit": "km",
            "role": "feature",
            "allowed_values": "15.0 to 3500.0",
            "nullable": "false",
            "generation_logic": "Vectorized Haversine distance multiplied by road network circuity factor",
        },
        {
            "column_name": "delivery_zone",
            "data_type": "string",
            "description": "Urban density and logistical accessibility tier of destination",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Metro, Tier 1, Tier 2, Tier 3, Rural / ODA",
            "nullable": "false",
            "generation_logic": "Mapped from destination city tier with realistic neighborhood variance",
        },
        {
            "column_name": "fulfillment_center",
            "data_type": "string",
            "description": "Origin fulfillment node code",
            "unit": "None",
            "role": "feature",
            "allowed_values": "FC-BOM-01, FC-DEL-01, FC-BLR-01, FC-HYD-01, etc.",
            "nullable": "false",
            "generation_logic": "Deterministic facility lookup by origin city",
        },
        {
            "column_name": "transport_mode",
            "data_type": "string",
            "description": "Primary mode of long-haul / line-haul transportation",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Road, Rail, Air, Sea",
            "nullable": "false",
            "generation_logic": "Distance-conditioned mode selection (Road dominates domestic freight)",
        },
        {
            "column_name": "carrier",
            "data_type": "string",
            "description": "3PL or dedicated logistics provider handling the consignment",
            "unit": "None",
            "role": "feature",
            "allowed_values": "BlueDart, Delhivery, Ekart Logistics, DTDC, Xpressbees, Shadowfax, TCI Express, SafeXpress",
            "nullable": "false",
            "generation_logic": "Market share weighted categorical distribution",
        },
        {
            "column_name": "vehicle_type",
            "data_type": "string",
            "description": "Fleet equipment / vehicle assigned for transit",
            "unit": "None",
            "role": "feature",
            "allowed_values": "2-Wheeler, Small Commercial Van, LCV, HCV, Multi-Axle Container, Belly Cargo, Cargo Aircraft, Freight Train, Feeder Vessel",
            "nullable": "false",
            "generation_logic": "Strictly constrained to be compatible with transport_mode and package weight",
        },
        {
            "column_name": "route_type",
            "data_type": "string",
            "description": "Topographical and logistical classification of transit route",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Urban Last-Mile, Inter-City Highway, National Trunk Corridor, Semi-Urban / Feeder, Hilly / Remote Terrain",
            "nullable": "false",
            "generation_logic": "Categorical sampling based on route distance and hub geography",
        },
        {
            "column_name": "warehouse_type",
            "data_type": "string",
            "description": "Classification of origin dispatch facility",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Local Fulfillment Center (FC), Regional Distribution Center (RDC), Mother Hub, Sort Center (SC)",
            "nullable": "false",
            "generation_logic": "Categorical distribution across facility tiers",
        },
        {
            "column_name": "service_level",
            "data_type": "string",
            "description": "Commercial delivery SLA agreement chosen by customer",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Same Day, Next Day, Express, Standard, Economy Surface",
            "nullable": "false",
            "generation_logic": "Categorical distribution calibrated to e-commerce delivery options",
        },
        {
            "column_name": "customer_type",
            "data_type": "string",
            "description": "Customer commercial segment",
            "unit": "None",
            "role": "feature",
            "allowed_values": "B2C, D2C, B2B, Enterprise",
            "nullable": "false",
            "generation_logic": "Categorical distribution across commercial customer personas",
        },
        {
            "column_name": "warehouse_processing_hours",
            "data_type": "float",
            "description": "Duration spent in origin warehouse for picking, packing, sorting, and staging",
            "unit": "hours",
            "role": "feature",
            "allowed_values": "1.5 to 48.0",
            "nullable": "false",
            "generation_logic": "Log-normal processing time with holiday demand multipliers and item count scaling",
        },
        {
            "column_name": "loading_time_hours",
            "data_type": "float",
            "description": "Duration required to cross-dock and load consignment onto transit vehicle",
            "unit": "hours",
            "role": "feature",
            "allowed_values": "0.2 to 8.0",
            "nullable": "false",
            "generation_logic": "Conditioned on vehicle type and package weight",
        },
        {
            "column_name": "handling_time_hours",
            "data_type": "float",
            "description": "Specialized handling and inspection duration",
            "unit": "hours",
            "role": "feature",
            "allowed_values": "0.1 to 5.0",
            "nullable": "false",
            "generation_logic": "Conditioned on item fragility, dangerous goods check, and priority level",
        },
        {
            "column_name": "customs_clearance_hours",
            "data_type": "float",
            "description": "Regulatory inspection and customs processing hours (0.0 for domestic)",
            "unit": "hours",
            "role": "feature",
            "allowed_values": "0.0 (domestic) to 48.0 (SEZ / bonded courier)",
            "nullable": "false",
            "generation_logic": "0.0 for 98.5% domestic; 12-48 hours for 1.5% cross-border / SEZ freight",
        },
        {
            "column_name": "number_of_stops",
            "data_type": "integer",
            "description": "Number of intermediate cross-dock hubs or sort centers along line-haul route",
            "unit": "count",
            "role": "feature",
            "allowed_values": "0 to 7",
            "nullable": "false",
            "generation_logic": "Proportional to distance and hub-and-spoke consolidation (0 for direct/air)",
        },
        {
            "column_name": "route_complexity_score",
            "data_type": "float",
            "description": "Engineered route difficulty index capturing terrain, tolls, and elevation",
            "unit": "Score (1.0 to 5.0)",
            "role": "feature",
            "allowed_values": "1.0 to 5.0",
            "nullable": "false",
            "generation_logic": "Derived from distance, number of stops, and hilly terrain factors",
        },
        {
            "column_name": "traffic_level",
            "data_type": "string",
            "description": "Qualitative traffic congestion severity along corridor",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Low, Moderate, High, Severe",
            "nullable": "true",
            "generation_logic": "Categorized from congestion_index; ~1.8% missing due to telematics dropout",
        },
        {
            "column_name": "congestion_index",
            "data_type": "float",
            "description": "Continuous normalized traffic congestion index (0.0 = clear, 1.0 = gridlock)",
            "unit": "Index [0.0, 1.0]",
            "role": "feature",
            "allowed_values": "0.05 to 0.98",
            "nullable": "true",
            "generation_logic": "Simulated from city density, peak shopping months, holidays, and urban routes",
        },
        {
            "column_name": "weather_condition",
            "data_type": "string",
            "description": "Meteorological weather condition encountered during line-haul transit",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Clear, Rainy, Heavy Rain / Storm, Fog / Low Visibility, Heatwave, Cyclonic / Gale",
            "nullable": "true",
            "generation_logic": "Seasonally correlated (Monsoon Jun-Sep, North India Winter Fog Dec-Jan); ~2.4% missing",
        },
        {
            "column_name": "weather_risk_score",
            "data_type": "float",
            "description": "Meteorological disruption probability index (0.0 = benign, 1.0 = severe catastrophe)",
            "unit": "Score [0.0, 1.0]",
            "role": "feature",
            "allowed_values": "0.01 to 0.99",
            "nullable": "true",
            "generation_logic": "Continuous disruption probability paired with weather condition",
        },
        {
            "column_name": "road_condition",
            "data_type": "string",
            "description": "Highway / road infrastructure surface state",
            "unit": "None",
            "role": "feature",
            "allowed_values": "Good, Average, Poor, Under Construction",
            "nullable": "true",
            "generation_logic": "Conditioned on route type and monsoon rainfall; ~1.2% missing",
        },
        {
            "column_name": "carrier_historical_delay_rate",
            "data_type": "float",
            "description": "Prior rolling 90-day delay rate for assigned carrier (cold feature, zero target leakage)",
            "unit": "Probability [0.0, 1.0]",
            "role": "feature",
            "allowed_values": "0.08 to 0.40",
            "nullable": "true",
            "generation_logic": "Historical carrier performance benchmark + noise; ~1.5% missing (cold start)",
        },
        {
            "column_name": "route_historical_delay_rate",
            "data_type": "float",
            "description": "Prior historical delay probability on this origin-destination corridor",
            "unit": "Probability [0.0, 1.0]",
            "role": "feature",
            "allowed_values": "0.06 to 0.45",
            "nullable": "true",
            "generation_logic": "Historical lane delay benchmark + noise; ~1.8% missing (new lanes)",
        },
        {
            "column_name": "warehouse_historical_delay_rate",
            "data_type": "float",
            "description": "Prior historical dispatch delay rate for origin fulfillment center",
            "unit": "Probability [0.0, 1.0]",
            "role": "feature",
            "allowed_values": "0.05 to 0.38",
            "nullable": "false",
            "generation_logic": "Facility baseline prior rate",
        },
        {
            "column_name": "destination_historical_delay_rate",
            "data_type": "float",
            "description": "Prior delivery exception rate in destination delivery zone",
            "unit": "Probability [0.0, 1.0]",
            "role": "feature",
            "allowed_values": "0.07 to 0.45",
            "nullable": "false",
            "generation_logic": "Destination delivery zone baseline prior rate",
        },
        {
            "column_name": "promised_delivery_days",
            "data_type": "float",
            "description": "Target SLA delivery duration promised to customer at order placement",
            "unit": "days",
            "role": "feature / SLA baseline",
            "allowed_values": "0.5 to 25.0",
            "nullable": "false",
            "generation_logic": "Customer SLA policy based on service level, distance tier, and priority level",
        },
        {
            "column_name": "actual_delivery_days",
            "data_type": "float",
            "description": "Actual elapsed time from order placement to customer delivery in fractional days",
            "unit": "days",
            "role": "target (regression)",
            "allowed_values": "0.25 to 30.0",
            "nullable": "false",
            "generation_logic": "Total elapsed physical hours / 24.0 (warehouse + loading + transit + customs + delays)",
        },
        {
            "column_name": "delivery_delay_days",
            "data_type": "float",
            "description": "Days overdue beyond promised delivery SLA: max(actual_delivery_days - promised_delivery_days, 0)",
            "unit": "days",
            "role": "target (regression / risk)",
            "allowed_values": "0.0 to 20.0",
            "nullable": "false",
            "generation_logic": "Mathematically enforced as max(actual_delivery_days - promised_delivery_days, 0)",
        },
        {
            "column_name": "is_delayed",
            "data_type": "integer",
            "description": "Binary classification target indicating whether shipment breached promised SLA (1) or was on-time (0)",
            "unit": "Binary [0, 1]",
            "role": "target (classification)",
            "allowed_values": "0, 1",
            "nullable": "false",
            "generation_logic": "Strictly defined as 1 if delivery_delay_days > 0 else 0 (~20-35% realistic positive class balance)",
        },
    ]
    return pd.DataFrame(dict_records)


# ==============================================================================
# DATASET METADATA & SERIALIZATION
# ==============================================================================

def create_dataset_metadata(
    df: pd.DataFrame,
    seed: int,
    start_date: str,
    end_date: str,
    validation_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Generates detailed JSON metadata documenting generation run and statistical distributions."""
    # Compute missingness percentages
    missing_rates = {col: round(float(df[col].isnull().mean()) * 100.0, 2) for col in df.columns if df[col].isnull().any()}

    # Compute key categorical frequencies (top 5 values for key features)
    categorical_cols = ["transport_mode", "carrier", "product_category", "priority_level", "service_level", "delivery_zone", "weather_condition"]
    cat_summary = {}
    for c in categorical_cols:
        counts = df[c].value_counts(dropna=False).head(5).to_dict()
        cat_summary[c] = {str(k): int(v) for k, v in counts.items()}

    # Compute numerical summaries
    num_cols = ["distance_km", "package_weight_kg", "actual_delivery_days", "promised_delivery_days", "delivery_delay_days"]
    num_summary = {}
    for c in num_cols:
        num_summary[c] = {
            "mean": round(float(df[c].mean()), 2),
            "std": round(float(df[c].std()), 2),
            "min": round(float(df[c].min()), 2),
            "p25": round(float(df[c].quantile(0.25)), 2),
            "median": round(float(df[c].median()), 2),
            "p75": round(float(df[c].quantile(0.75)), 2),
            "p95": round(float(df[c].quantile(0.95)), 2),
            "max": round(float(df[c].max()), 2),
        }

    return {
        "dataset_name": "ETAFlow Synthetic Logistics Delivery Dataset",
        "version": "1.0.0",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "generator_script": "scripts/generate_dataset.py",
        "random_seed": seed,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "configured_date_range": {"start_date": start_date, "end_date": end_date},
        "target_definitions": {
            "regression_target": "actual_delivery_days (Total duration in days from order placement to delivery)",
            "classification_target": "is_delayed (Binary: 1 if delivery_delay_days > 0 else 0)",
            "delay_severity_target": "delivery_delay_days (max(actual_delivery_days - promised_delivery_days, 0))"
        },
        "delay_class_balance": {
            "total_shipments": len(df),
            "delayed_shipments": int(df["is_delayed"].sum()),
            "on_time_shipments": int((df["is_delayed"] == 0).sum()),
            "delay_rate_percentage": round(float(df["is_delayed"].mean()) * 100.0, 2),
        },
        "missing_value_rates_pct": missing_rates,
        "numerical_distributions": num_summary,
        "categorical_distributions": cat_summary,
        "validation_summary": validation_stats,
    }


# ==============================================================================
# MAIN CLI ENTRYPOINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETAFlow Synthetic Logistics Dataset Generator. Generates realistic, reproducible delivery data with domain physics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of shipment records to generate (e.g. 10000, 50000, 100000, 250000, 500000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for reproducibility.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date for order generation (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-31",
        help="End date for order generation (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-data",
        type=str,
        default="data/raw/shipments.csv",
        help="Target output CSV file path for shipments data.",
    )
    parser.add_argument(
        "--output-dict",
        type=str,
        default="data/reference/data_dictionary.csv",
        help="Target output CSV file path for data dictionary.",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default="data/reference/dataset_metadata.json",
        help="Target output JSON file path for dataset metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing output files without prompting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.output_data)
    dict_path = Path(args.output_dict)
    meta_path = Path(args.output_meta)

    # 1. Overwrite protection
    existing_files = [str(p) for p in [data_path, dict_path, meta_path] if p.exists()]
    if existing_files and not args.force:
        logger.error(
            "Output file(s) already exist:\n  %s\nUse --force to explicitly overwrite existing files.",
            "\n  ".join(existing_files),
        )
        sys.exit(1)

    # 2. Safe directory creation
    for p in [data_path, dict_path, meta_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    logger.info("=================================================================")
    logger.info("ETAFLOW SYNTHETIC LOGISTICS DATASET GENERATOR")
    logger.info("=================================================================")
    logger.info("Configured samples:     %d", args.samples)
    logger.info("Random seed:            %d", args.seed)
    logger.info("Temporal interval:      %s to %s", args.start_date, args.end_date)
    logger.info("Output dataset path:    %s", data_path)
    logger.info("Data dictionary path:   %s", dict_path)
    logger.info("Metadata JSON path:     %s", meta_path)
    logger.info("Force overwrite:        %s", args.force)
    logger.info("-----------------------------------------------------------------")

    # 3. Generate dataset
    generator = LogisticsDatasetGenerator(
        n_samples=args.samples,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    df = generator.generate()

    # 4. Perform rigorous validation
    validation_stats = validate_dataset(df, expected_samples=args.samples)

    # 5. Create Data Dictionary & Metadata
    logger.info("Building data dictionary and metadata artifacts...")
    df_dict = create_data_dictionary()
    metadata = create_dataset_metadata(
        df=df,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
        validation_stats=validation_stats,
    )

    # 6. Save Artifacts to Disk
    logger.info("Writing dataset to '%s' (this may take a few moments for large samples)...", data_path)
    df.to_csv(data_path, index=False)

    logger.info("Writing data dictionary to '%s'...", dict_path)
    df_dict.to_csv(dict_path, index=False)

    logger.info("Writing dataset metadata to '%s'...", meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = (datetime.now() - start_time).total_seconds()
    file_size_mb = data_path.stat().st_size / (1024 * 1024)

    # 7. Print Professional Dataset Summary
    print("\n" + "=" * 70)
    print("                     DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"Status:                      SUCCESSFUL")
    print(f"Elapsed Time:                {elapsed:.2f} seconds")
    print(f"Dataset File Size:           {file_size_mb:.2f} MB")
    print(f"Dataset Shape:               {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Unique Shipment IDs:         {validation_stats['unique_shipments']:,}")
    print(f"Configured Date Range:       {args.start_date} -> {args.end_date}")
    print("-" * 70)
    print("TARGET VARIABLE DISTRIBUTION:")
    print(f"  Total Shipments:           {validation_stats['total_rows']:,}")
    print(f"  Delayed Shipments:         {validation_stats['delayed_shipments']:,}")
    print(f"  Delay Percentage:          {validation_stats['delay_percentage']:.2f}%")
    print(f"  Average Delivery Time:     {validation_stats['mean_delivery_days']:.2f} days")
    print(f"  Median Delivery Time:      {validation_stats['median_delivery_days']:.2f} days")
    print(f"  Average Delay (Overall):   {validation_stats['mean_delay_days']:.2f} days")
    print(f"  Maximum Delay:             {validation_stats['max_delay_days']:.2f} days")
    print("-" * 70)
    print("MISSING VALUE SUMMARY:")
    for col, rate in metadata["missing_value_rates_pct"].items():
        print(f"  {col:<35} : {rate:.2f}% missing")
    print("-" * 70)
    print("ARTIFACTS CREATED:")
    print(f"  1. Dataset:         {data_path.resolve()}")
    print(f"  2. Data Dictionary: {dict_path.resolve()}")
    print(f"  3. Metadata:        {meta_path.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
