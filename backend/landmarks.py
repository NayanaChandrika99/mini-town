"""
Named landmarks for the Mini-Town demo.

Coordinates are expressed in simulation units (0..map_width, 0..map_height).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Landmark:
    id: str
    name: str
    description: str
    x: float
    y: float


LANDMARKS: List[Landmark] = [
    Landmark(
        id="plaza_fountain",
        name="Town Plaza Fountain",
        description="Central plaza with the stone fountain and cafe tables.",
        x=200.0,
        y=155.0,
    ),
    Landmark(
        id="market_square",
        name="Market Square",
        description="Busy market stalls near the cobblestone crossroads.",
        x=115.0,
        y=190.0,
    ),
    Landmark(
        id="riverside_park",
        name="Riverside Park",
        description="Grassy parkland by the river bridge.",
        x=320.0,
        y=125.0,
    ),
    Landmark(
        id="observatory_hill",
        name="Observatory Hill",
        description="Quiet overlook on the hilltop observatory path.",
        x=345.0,
        y=65.0,
    ),
    Landmark(
        id="garden_terrace",
        name="Garden Terrace",
        description="Flower garden and benches near the town homes.",
        x=185.0,
        y=255.0,
    ),
]

_LANDMARK_LOOKUP: Dict[str, Landmark] = {landmark.id: landmark for landmark in LANDMARKS}


def get_landmark(landmark_id: str) -> Optional[Landmark]:
    """Return a landmark by id."""
    return _LANDMARK_LOOKUP.get(landmark_id)


def list_landmarks() -> List[Landmark]:
    """Return all defined landmarks."""
    return LANDMARKS.copy()
