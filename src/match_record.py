from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pandas as pd

TourneyLevel = Literal["G", "M", "A", "C", "S", "F", "D"]
Hand = Literal["R", "L", "U"]
PlayerEntry = Literal["WC", "Q", "LL", "PR", "ITF"]
BestOf = Literal[3, 5]

TOURNEY_LEVELS: set[str] = {"G", "M", "A", "C", "S", "F", "D"}
HANDS: set[str] = {"R", "L", "U"}
KNOWN_ENTRIES: set[str] = {"WC", "Q", "LL", "PR", "ITF"}
BEST_OF_VALUES: set[int] = {3, 5}
HEIGHT_MIN_CM = 140
HEIGHT_MAX_CM = 220

JEFF_SACKMANN_COLUMNS: list[str] = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
]

WINNER_STAT_COLUMNS = [
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
]

LOSER_STAT_COLUMNS = [
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]


@dataclass(frozen=True)
class RowAuditResult:
    total_rows: int
    valid_rows: int
    invalid_rows: int
    sample_errors: tuple[str, ...]


def _is_blank(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip() == ""


def _row_context(row: pd.Series) -> str:
    return f"tourney_id={row.get('tourney_id', '?')}, match_num={row.get('match_num', '?')}"


def _require_str(row: pd.Series, col: str) -> str:
    value = row[col]
    if _is_blank(value):
        raise ValueError(f"Missing required field '{col}' ({_row_context(row)})")
    return str(value).strip()


def _require_int(row: pd.Series, col: str, *, min_val: int = 1) -> int:
    value = row[col]
    if _is_blank(value):
        raise ValueError(f"Missing required field '{col}' ({_row_context(row)})")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid int for '{col}': {value!r} ({_row_context(row)})"
        ) from exc
    if parsed < min_val:
        raise ValueError(f"Invalid int for '{col}': {parsed} ({_row_context(row)})")
    return parsed


def _require_float(row: pd.Series, col: str, *, min_val: float = 0) -> float:
    value = row[col]
    if _is_blank(value):
        raise ValueError(f"Missing required field '{col}' ({_row_context(row)})")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid float for '{col}': {value!r} ({_row_context(row)})"
        ) from exc
    if parsed <= min_val:
        raise ValueError(f"Invalid float for '{col}': {parsed} ({_row_context(row)})")
    return parsed


def _require_enum(row: pd.Series, col: str, allowed: set[str]) -> str:
    value = _require_str(row, col)
    if value not in allowed:
        raise ValueError(f"Invalid value for '{col}': {value!r} ({_row_context(row)})")
    return value


def _optional_int(row: pd.Series, col: str, *, min_val: int = 1) -> int | None:
    value = row[col]
    if _is_blank(value):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid int for '{col}': {value!r} ({_row_context(row)})"
        ) from exc
    if parsed < min_val:
        raise ValueError(f"Invalid int for '{col}': {parsed} ({_row_context(row)})")
    return parsed


def _optional_enum(row: pd.Series, col: str, allowed: set[str]) -> str | None:
    value = row[col]
    if _is_blank(value):
        return None
    text = str(value).strip()
    if text not in allowed:
        raise ValueError(f"Invalid value for '{col}': {text!r} ({_row_context(row)})")
    return text


def _require_date_yyyymmdd(row: pd.Series, col: str) -> int:
    value = _require_int(row, col, min_val=10000101)
    if len(str(value)) != 8:
        raise ValueError(f"Invalid date for '{col}': {value} ({_row_context(row)})")
    return value


def _optional_ioc(row: pd.Series, col: str) -> str | None:
    value = row[col]
    if _is_blank(value):
        return None
    text = str(value).strip().upper()
    if len(text) != 3 or not text.isalpha():
        raise ValueError(f"Invalid IOC for '{col}': {value!r} ({_row_context(row)})")
    return text


def _optional_height_cm(row: pd.Series, col: str) -> int | None:
    height = _optional_int(row, col, min_val=HEIGHT_MIN_CM)
    if height is None:
        return None
    if height > HEIGHT_MAX_CM:
        raise ValueError(f"Invalid height for '{col}': {height} ({_row_context(row)})")
    return height


def _parse_stats_block(row: pd.Series, columns: list[str]) -> PlayerMatchStats | None:
    values: list[int | None] = []
    for col in columns:
        raw = row[col]
        if _is_blank(raw):
            values.append(None)
            continue
        parsed = int(float(raw))
        if parsed < 0:
            raise ValueError(f"Invalid stat '{col}': {parsed} ({_row_context(row)})")
        values.append(parsed)

    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError(f"Partial stat block in {columns[0]} ({_row_context(row)})")

    return PlayerMatchStats(
        ace=values[0],  # type: ignore[arg-type]
        df=values[1],  # type: ignore[arg-type]
        svpt=values[2],  # type: ignore[arg-type]
        first_in=values[3],  # type: ignore[arg-type]
        first_won=values[4],  # type: ignore[arg-type]
        second_won=values[5],  # type: ignore[arg-type]
        sv_gms=values[6],  # type: ignore[arg-type]
        bp_saved=values[7],  # type: ignore[arg-type]
        bp_faced=values[8],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class PlayerInfo:
    id: int
    name: str
    hand: Hand
    height_cm: int
    ioc: str | None
    age: float
    seed: int | None
    entry: PlayerEntry | None
    rank: int | None
    rank_points: int | None


@dataclass(frozen=True)
class PlayerMatchStats:
    ace: int
    df: int
    svpt: int
    first_in: int
    first_won: int
    second_won: int
    sv_gms: int
    bp_saved: int
    bp_faced: int


@dataclass(frozen=True)
class MatchRecord:
    tourney_id: str
    tourney_name: str
    surface: str
    draw_size: int
    tourney_level: TourneyLevel
    tourney_date: int
    match_num: int
    score: str
    best_of: BestOf
    round: str
    minutes: int | None
    player1: PlayerInfo
    player2: PlayerInfo
    player1_stats: PlayerMatchStats | None
    player2_stats: PlayerMatchStats | None
    player1_won: bool
    source: str = "jeff_sackmann"


def _player_from_prefix(row: pd.Series, prefix: str) -> PlayerInfo:
    entry = _optional_enum(row, f"{prefix}_entry", KNOWN_ENTRIES)
    return PlayerInfo(
        id=_require_int(row, f"{prefix}_id"),
        name=_require_str(row, f"{prefix}_name"),
        hand=cast(Hand, _require_enum(row, f"{prefix}_hand", HANDS)),
        height_cm=_optional_height_cm(row, f"{prefix}_ht"),
        ioc=_optional_ioc(row, f"{prefix}_ioc"),
        age=_require_float(row, f"{prefix}_age"),
        seed=_optional_int(row, f"{prefix}_seed", min_val=1),
        entry=cast(PlayerEntry | None, entry),
        rank=_optional_int(row, f"{prefix}_rank", min_val=1),
        rank_points=_optional_int(row, f"{prefix}_rank_points", min_val=1),
    )


def from_jeff_sackmann_row(row: pd.Series) -> MatchRecord:
    winner = _player_from_prefix(row, "winner")
    loser = _player_from_prefix(row, "loser")
    w_stats = _parse_stats_block(row, WINNER_STAT_COLUMNS)
    l_stats = _parse_stats_block(row, LOSER_STAT_COLUMNS)

    best_of_raw = _require_int(row, "best_of", min_val=3)
    if best_of_raw not in BEST_OF_VALUES:
        raise ValueError(f"Invalid best_of: {best_of_raw} ({_row_context(row)})")

    match = MatchRecord(
        tourney_id=_require_str(row, "tourney_id"),
        tourney_name=_require_str(row, "tourney_name"),
        surface=_require_str(row, "surface"),
        draw_size=_require_int(row, "draw_size"),
        tourney_level=cast(
            TourneyLevel, _require_enum(row, "tourney_level", TOURNEY_LEVELS)
        ),
        tourney_date=_require_date_yyyymmdd(row, "tourney_date"),
        match_num=_require_int(row, "match_num"),
        score=_require_str(row, "score"),
        best_of=cast(BestOf, best_of_raw),
        round=_require_str(row, "round"),
        minutes=_optional_int(row, "minutes", min_val=1),
        player1=winner,
        player2=loser,
        player1_stats=w_stats,
        player2_stats=l_stats,
        player1_won=True,
    )

    if random.random() < 0.5:
        return match

    return MatchRecord(
        tourney_id=match.tourney_id,
        tourney_name=match.tourney_name,
        surface=match.surface,
        draw_size=match.draw_size,
        tourney_level=match.tourney_level,
        tourney_date=match.tourney_date,
        match_num=match.match_num,
        score=match.score,
        best_of=match.best_of,
        round=match.round,
        minutes=match.minutes,
        player1=match.player2,
        player2=match.player1,
        player1_stats=match.player2_stats,
        player2_stats=match.player1_stats,
        player1_won=False,
        source=match.source,
    )


def audit_jeff_sackmann_rows(
    data_dir: Path, *, max_samples: int = 15
) -> RowAuditResult:
    total_rows = 0
    invalid_rows = 0
    sample_errors: list[str] = []

    for filepath in sorted(data_dir.glob("atp_matches_*.csv")):
        df = pd.read_csv(filepath)
        if list(df.columns) != JEFF_SACKMANN_COLUMNS:
            raise ValueError(f"Unexpected columns in {filepath.name}")

        for _, row in df.iterrows():
            total_rows += 1
            try:
                from_jeff_sackmann_row(row)
            except ValueError as exc:
                invalid_rows += 1
                if len(sample_errors) < max_samples:
                    sample_errors.append(str(exc))

    return RowAuditResult(
        total_rows=total_rows,
        valid_rows=total_rows - invalid_rows,
        invalid_rows=invalid_rows,
        sample_errors=tuple(sample_errors),
    )
