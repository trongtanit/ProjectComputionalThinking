# classes.py
from typing import List, Set, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field, asdict
import math

# -----------------------
# Helper utilities
# -----------------------

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two points (kilometers).
    """
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def estimate_travel_time_minutes_km(distance_km: float, avg_speed_kmh: float = 25.0) -> float:
    """
    Simple travel time estimator: minutes = distance_km / speed_kmh * 60
    Default speed ~25 km/h (urban mixed traffic).
    """
    if distance_km is None:
        return float('inf')
    return (distance_km / avg_speed_kmh) * 60.0

# -----------------------
# POI class - Enhanced
# -----------------------

@dataclass
class PointOfInterest:
    poi_id: int
    name: str
    latitude: float
    longitude: float
    poi_type: str                       # coarse type (e.g., Food_Dining, Culture_History)
    category_detail: Optional[str] = None   # fine-grained (e.g., cafe, rooftop, museum)
    vibe: Optional[str] = None             # one of 7 vibes (Chill, Party, Romantic, Local, Culture, Family, Luxury)
    avg_cost: float = 0.0                  # estimated average cost in VND
    simulated_rating: float = 0.0          # simulated or scraped rating (0-5)
    dwell_time_minutes: int = 60           # expected dwell time in minutes
    popularity_score: float = 0.0          # 0..1 popularity normalized
    opening_time: str = "08:00"            # "HH:MM"
    closing_time: str = "21:00"            # "HH:MM"
    time_block_suitability: Set[str] = field(default_factory=set)  # e.g., {"morning","afternoon","evening","night"}
    is_central: bool = False
    recommendation_score: float = 0.0      # dynamic score from ranker (0..1)
    travel_time_cache: Dict[int, float] = field(default_factory=dict)  # poi_id -> minutes

    def __str__(self) -> str:
        return (f"POI {self.poi_id}: {self.name} [{self.category_detail or self.poi_type}] "
                f"Vibe={self.vibe} Cost={int(self.avg_cost):,} VNĐ Rating={self.simulated_rating:.1f}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PointOfInterest":
        return cls(
            poi_id=data.get("poi_id", -1),
            name=data.get("name", ""),
            latitude=float(data.get("latitude", 0.0)),
            longitude=float(data.get("longitude", 0.0)),
            poi_type=data.get("poi_type", ""),
            category_detail=data.get("category_detail"),
            vibe=data.get("vibe"),
            avg_cost=float(data.get("avg_cost", 0.0)),
            simulated_rating=float(data.get("simulated_rating", 0.0)),
            dwell_time_minutes=int(data.get("dwell_time_minutes", 60)),
            popularity_score=float(data.get("popularity_score", 0.0)),
            opening_time=data.get("opening_time", "08:00"),
            closing_time=data.get("closing_time", "21:00"),
            time_block_suitability=set(data.get("time_block_suitability", [])),
            is_central=bool(data.get("is_central", False)),
            recommendation_score=float(data.get("recommendation_score", 0.0)),
            travel_time_cache=dict(data.get("travel_time_cache", {}))
        )

    # --------------
    # Travel helpers
    # --------------
    def distance_to(self, other: "PointOfInterest") -> float:
        """Return distance in km."""
        return haversine_distance_km(self.latitude, self.longitude, other.latitude, other.longitude)

    def estimate_travel_time_to(self, other: "PointOfInterest", avg_speed_kmh: float = 25.0) -> float:
        """Return estimated travel minutes between this POI and other (uses cache if available)."""
        if other.poi_id in self.travel_time_cache:
            return self.travel_time_cache[other.poi_id]
        dist = self.distance_to(other)
        minutes = estimate_travel_time_minutes_km(dist, avg_speed_kmh)
        # cache both directions if desired
        self.travel_time_cache[other.poi_id] = minutes
        return minutes

    def is_open_during(self, time_hhmm: str) -> bool:
        """
        time_hhmm: "HH:MM"
        Returns True if the POI is open at that time (naive check, does not handle overnight ranges crossing midnight).
        """
        def to_minutes(t: str) -> int:
            h, m = map(int, t.split(":"))
            return h * 60 + m
        t = to_minutes(time_hhmm)
        open_t = to_minutes(self.opening_time)
        close_t = to_minutes(self.closing_time)
        # handle overnight (e.g., close at 02:00)
        if close_t <= open_t:
            # closing next day
            return t >= open_t or t <= close_t
        return open_t <= t <= close_t

# -----------------------
# Itinerary class - Enhanced
# -----------------------

@dataclass
class Itinerary:
    poi_sequence: List[PointOfInterest] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration_minutes: float = 0.0  # includes travel + dwell
    total_satisfaction_score: float = 0.0  # aggregated experience score
    details: List[Dict[str, Any]] = field(default_factory=list)  # step-by-step details

    def compute_summary(self, avg_speed_kmh: float = 25.0, return_to_start: bool = False) -> None:
        """
        Compute totals for cost, duration and satisfaction based on current poi_sequence.
        This function fills total_cost, total_duration_minutes, total_satisfaction_score and details.
        """
        self.total_cost = 0.0
        self.total_duration_minutes = 0.0
        self.total_satisfaction_score = 0.0
        self.details = []

        if not self.poi_sequence:
            return

        # Sum dwell times and costs
        for poi in self.poi_sequence:
            self.total_cost += poi.avg_cost
            self.total_duration_minutes += poi.dwell_time_minutes
            # satisfaction contribution: weighted by recommendation_score and popularity
            self.total_satisfaction_score += (poi.recommendation_score * 0.7 + poi.popularity_score * 0.3) * 10.0

        # Sum travel times between consecutive POIs
        for i in range(len(self.poi_sequence) - 1):
            a = self.poi_sequence[i]
            b = self.poi_sequence[i + 1]
            travel_min = a.estimate_travel_time_to(b, avg_speed_kmh)
            self.total_duration_minutes += travel_min
            self.details.append({
                "from": a.name,
                "to": b.name,
                "distance_km": round(a.distance_to(b), 3),
                "travel_minutes": round(travel_min, 1)
            })

        # option to return to start (useful if round-trip required)
        if return_to_start and len(self.poi_sequence) > 1:
            last = self.poi_sequence[-1]
            first = self.poi_sequence[0]
            travel_min = last.estimate_travel_time_to(first, avg_speed_kmh)
            self.total_duration_minutes += travel_min
            self.details.append({
                "from": last.name,
                "to": first.name,
                "distance_km": round(last.distance_to(first), 3),
                "travel_minutes": round(travel_min, 1)
            })

        # Normalize satisfaction (average)
        self.total_satisfaction_score = (self.total_satisfaction_score / max(1, len(self.poi_sequence)))

    def print_summary(self) -> None:
        print("--- LỘ TRÌNH TỐI ƯU ---")
        for i, poi in enumerate(self.poi_sequence):
            print(f"{i+1}. {poi.name} ({poi.category_detail or poi.poi_type}) | Vibe={poi.vibe} "
                  f"| Cost={int(poi.avg_cost):,} VNĐ | Dwell={poi.dwell_time_minutes}min | RecScore={poi.recommendation_score:.2f}")
        hours = int(self.total_duration_minutes // 60)
        minutes = int(self.total_duration_minutes % 60)
        print(f"\nTổng thời gian: {hours} giờ {minutes} phút ({int(self.total_duration_minutes)} phút)")
        print(f"Tổng chi phí ước tính: {int(self.total_cost):,} VNĐ")
        print(f"Chỉ số trải nghiệm (satisfaction): {self.total_satisfaction_score:.2f}")
        if self.details:
            print("\n--- Chi tiết di chuyển ---")
            for seg in self.details:
                print(f"{seg['from']} -> {seg['to']}: {seg['distance_km']} km, {seg['travel_minutes']} phút")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "poi_ids": [p.poi_id for p in self.poi_sequence],
            "total_cost": self.total_cost,
            "total_duration_minutes": self.total_duration_minutes,
            "total_satisfaction_score": self.total_satisfaction_score,
            "details": self.details
        }

    @classmethod
    def from_poi_list(cls, poi_list: List[PointOfInterest]) -> "Itinerary":
        it = cls(poi_sequence=poi_list.copy())
        it.compute_summary()
        return it
