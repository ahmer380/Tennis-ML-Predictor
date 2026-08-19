class EloRatingEngine:
    """A class to calculate and update Elo ratings for tennis players."""

    def expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def k_factor(self, matches_played: int, tourney_level: str) -> float:
        """Calculate K-factor based on player experience and tournament level."""
        MIN_K_FACTOR = 18.0
        MAX_K_FACTOR = 40.0

        k_factor = max(MIN_K_FACTOR, min(MAX_K_FACTOR, 400.0 / (matches_played + 1)))

        tier_multipliers = {"G": 1.1, "M": 1.0, "F": 1.0, "A": 0.9, "C": 0.6}

        return k_factor * tier_multipliers[tourney_level]

    def update_ratings(
        self,
        elo_a: float,
        elo_b: float,
        matches_played_a: int,
        matches_played_b: int,
        tourney_level: str,
        score_a: float,
    ) -> tuple[float, float]:
        """Update player elos according to this formula: https://martiningram.github.io/elo-dynamic"""
        expected_a = self.expected_score(elo_a, elo_b)
        expected_b = self.expected_score(elo_b, elo_a)

        k_a = self.k_factor(matches_played_a, tourney_level)
        k_b = self.k_factor(matches_played_b, tourney_level)

        new_elo_a = elo_a + k_a * (score_a - expected_a)
        new_elo_b = elo_b + k_b * ((1 - score_a) - expected_b)

        return new_elo_a, new_elo_b
