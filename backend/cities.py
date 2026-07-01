"""Fixed set of cities the dashboard tracks.

Kept as plain data (not user input) so the demo has no attack surface for
arbitrary geocoding lookups and stays reproducible.
"""

CITIES = [
    {"name": "São Paulo", "latitude": -23.5505, "longitude": -46.6333},
    {"name": "New York", "latitude": 40.7128, "longitude": -74.0060},
    {"name": "London", "latitude": 51.5074, "longitude": -0.1278},
    {"name": "Tokyo", "latitude": 35.6762, "longitude": 139.6503},
    {"name": "Sydney", "latitude": -33.8688, "longitude": 151.2093},
    {"name": "Cairo", "latitude": 30.0444, "longitude": 31.2357},
]
