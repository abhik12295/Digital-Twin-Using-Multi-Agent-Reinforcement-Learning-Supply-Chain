from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class LiveDataConfig:
    fred_series: tuple[str, ...] = (
        "PCU484484",          # Truck Transportation PPI
        "PCU484121484121",    # General Freight Trucking, Long-Distance Truckload
        "PCU484110484110P",   # General Freight Trucking, Local: Primary Services
    )
    fred_start_date: str = "2020-01-01"
    fred_end_date: str = "9999-12-31"

    # Example NWS points; later you can make these dynamic by route/state
    nws_points: tuple[tuple[float, float], ...] = (
        (41.8781, -87.6298),   # Chicago
        (32.7767, -96.7970),   # Dallas
        (33.4484, -112.0740),  # Phoenix
        (33.7490, -84.3880),   # Atlanta
    )