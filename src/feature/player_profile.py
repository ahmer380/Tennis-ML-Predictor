from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

BASE_ELO = 1500.0


@dataclass
class _Match:
    """Class to hold match information for a player, including date, outcome, and game statistics."""

    date: pd.Timestamp
    won: bool
    p_ace: float
    p_df: float
    p_1st_in: float
    p_1st_won: float
    p_2nd_won: Optional[float]
    p_bp_saved: Optional[float]
    p_rp_won: float
    p_bp_won: Optional[float]


@dataclass
class PlayerProfile:
    """Class to hold player profile information, including Elo ratings, match history, and game stats."""

    # Bio and physical
    id: int
    name: str
    age: int = None
    ht: int = None

    # Ranking
    rank: int = None
    rank_points: int = None

    # Elo
    elos: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: BASE_ELO))

    # Head-to-head
    h2h_records: Dict[int, List[bool]] = field(
        default_factory=lambda: defaultdict(list)
    )  # key=opponent_id, value=win/loss record

    # Fatigue
    last_tournament_played_date: pd.Timestamp = None
    current_tournament_minutes_played: int = 0

    # Experience, form, and game stats
    match_history: Dict[str, List[_Match]] = field(default_factory=lambda: defaultdict(list))

    def to_dict(self) -> Dict:
        """Convert the PlayerProfile to a lightweight dictionary representation."""
        return {
            "name": self.name,
            "rank": self.rank,
            "rank_points": self.rank_points,
            "age": round(self.age),
            "height": self.ht,
            "global_elo": round(self.elos["global"], 2),
            "hard_elo": round(self.elos["Hard"], 2),
            "clay_elo": round(self.elos["Clay"], 2),
            "grass_elo": round(self.elos["Grass"], 2),
        }

    def get_matches_played(self, surface: str = "global", from_date: pd.Timestamp = None) -> int:
        """Return the number of matches played on a given surface from a specific date."""
        if not from_date:
            return len(self.match_history[surface])

        matches_played = 0
        for match in range(len(self.match_history[surface]) - 1, -1, -1):
            if self.match_history[surface][match].date >= from_date:
                matches_played += 1
            else:
                break

        return matches_played

    def get_h2h_wins(self, opponent_id: int) -> int:
        """Return the number of head-to-head wins against a specific opponent."""
        return self.h2h_records[opponent_id].count(True)

    def get_recent_win_percentage(self, num_matches: int, surface: str = "global") -> float:
        """Return the win percentage over the last `num_matches` on a given surface."""
        recent_matches = self.match_history[surface][-min(num_matches, len(self.match_history[surface])) :]
        return sum(match.won for match in recent_matches) / len(recent_matches) if recent_matches else 0.5

    def get_recent_game_stat_average(
        self, game_stat: str, num_matches: int = 100, default: float = 0.5, surface: str = "global"
    ) -> float:
        """Return the average of a game stat over the last `num_matches`."""
        recent_matches = self.match_history[surface][-min(num_matches, len(self.match_history[surface])) :]
        recent_stats = [getattr(match, game_stat) for match in recent_matches if getattr(match, game_stat) is not None]
        return sum(recent_stats) / len(recent_stats) if recent_stats else default

    def pre_match_update(self, row: pd.Series) -> None:
        """Update the player's profile before a match"""
        if self.last_tournament_played_date != pd.to_datetime(row["tourney_date"], format="%Y%m%d"):
            self.current_tournament_minutes_played = 0

    def post_match_update(self, row: pd.Series, prefix: str) -> None:
        """Update the player's profile after a match (elo update is handled externally)."""
        assert prefix in ["player_A", "player_B"], "Prefix must be either 'player_A' or 'player_B'"
        opponent_prefix = "player_B" if prefix == "player_A" else "player_A"

        won = row[f"player_A_win"] == (1 if prefix == "player_A" else 0)

        self.age = row[f"{prefix}_age"]
        self.ht = row[f"{prefix}_ht"]

        self.rank = row[f"{prefix}_rank"]
        self.rank_points = row[f"{prefix}_rank_points"]

        self.h2h_records[row[f"{opponent_prefix}_id"]].append(won)

        self.last_tournament_played_date = pd.to_datetime(row["tourney_date"], format="%Y%m%d")
        self.current_tournament_minutes_played += row["minutes"]

        p_ace = row[f"{prefix}_ace"] / row[f"{prefix}_svpt"]
        p_df = row[f"{prefix}_df"] / row[f"{prefix}_svpt"]
        p_1st_in = row[f"{prefix}_1stIn"] / row[f"{prefix}_svpt"]
        p_1st_won = row[f"{prefix}_1stWon"] / row[f"{prefix}_1stIn"]
        p_2nd_won = (
            row[f"{prefix}_2ndWon"] / (row[f"{prefix}_svpt"] - row[f"{prefix}_1stIn"])
            if row[f"{prefix}_svpt"] - row[f"{prefix}_1stIn"] > 0
            else None
        )
        p_bp_saved = row[f"{prefix}_bpSaved"] / row[f"{prefix}_bpFaced"] if row[f"{prefix}_bpFaced"] > 0 else None
        p_rp_won = (
            row[f"{opponent_prefix}_svpt"] - row[f"{opponent_prefix}_1stWon"] - row[f"{opponent_prefix}_2ndWon"]
        ) / row[f"{opponent_prefix}_svpt"]
        p_bp_won = (
            (row[f"{opponent_prefix}_bpFaced"] - row[f"{opponent_prefix}_bpSaved"]) / row[f"{opponent_prefix}_bpFaced"]
            if row[f"{opponent_prefix}_bpFaced"] > 0
            else None
        )

        match = _Match(
            date=pd.to_datetime(row["tourney_date"], format="%Y%m%d"),
            won=won,
            p_ace=p_ace,
            p_df=p_df,
            p_1st_in=p_1st_in,
            p_1st_won=p_1st_won,
            p_2nd_won=p_2nd_won,
            p_bp_saved=p_bp_saved,
            p_rp_won=p_rp_won,
            p_bp_won=p_bp_won,
        )
        self.match_history["global"].append(match)
        self.match_history[row["surface"]].append(match)
