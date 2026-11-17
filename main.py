"""Forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.

For each of hours 2..24 of the release day (UTC), emit five quantiles
(5/25/50/75/95%) of concurrent viewers in the game's Twitch category,
given everything recorded up to the end of hour 1.

Approach: a log-space comparable-launch forecaster, fitted at run time
from the case file's own comparables (past big launches). For each hour
h, it fits

    log viewers(h) ~ intercept + slope * log(hour-1 print)

by weighted least squares over the comparables. The weights are a
two-dimensional Gaussian kernel measuring how much each comparable
resembles the target: one axis is a = log(hour-1 print / pre-launch
baseline), which separates rolling-midnight launches (large a: the
category was quiet before, then the regions unlocked one by one) from
simultaneous-global launches (small a: the whole world unlocked at
once); the other axis is log(hour-1 print) itself, the launch's scale.
Early-access-contaminated comparables are down-weighted (their release
day was already partly spent), and when the target is a console-only
launch its console-only comparables are up-weighted (console launches
share the rolling-midnight shape: an overnight dip, a jump at the US
unlock, and a big US prime-time evening).

The forecast spread is the weighted residual standard deviation of the
fit, floored, turned into quantiles with the normal curve. Everything
is deterministic: no randomness, no network, standard library only.

If anything about the model fails for a case, the program falls back
to a persistence forecast (every hour draws hour 1's print, with a
spread that widens like a random walk in log space), so it always
writes a valid output file.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Kernel bandwidths (in natural-log units) for the similarity weights.
BW_A = 3.0       # on a = log(hour-1 print / pre-launch baseline)
BW_LH1 = 1.5     # on log(hour-1 print)
# Comparables whose unlock notes say early-access-contaminated count less.
EA_WEIGHT = 0.25
# When the target is console-only, console-only comparables count more.
CONSOLE_UP = 3.0
# The forecast spread never goes below this (log units).
SD_FLOOR = 0.20
# The fitted slope of log viewers(h) on log(hour-1 print) is clamped to
# this range. GTA VI's hour-1 print will likely exceed every training
# print, so the model extrapolates linearly in log space there; a
# degenerate fit (a negative or very steep slope) would explode that
# extrapolation. Observed slopes run 0.41..1.11 across the practice
# cases and a final-like case, so the clamp only fires on a fit that
# is already implausible.
SLOPE_MIN = 0.0
SLOPE_MAX = 1.25
# The persistence fallback's spread growth, per square-root hour.
PERSIST_SIGMA = 0.15


def parse_ts(text):
    """Parse an ISO-8601 timestamp (with a Z suffix) to a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(values):
    """The median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None when the matrix is (near-)singular.
    """
    n = len(matrix)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        rest = sum(rows[i][j] * out[j] for j in range(i + 1, n))
        out[i] = (rows[i][n] - rest) / rows[i][i]
    return out


def weighted_least_squares(design, target, weights):
    """Weighted least squares for design rows that already hold a 1.0
    intercept column. Returns the coefficient list, or None if singular."""
    k = len(design[0])
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for row, y, w in zip(design, target, weights):
        for i in range(k):
            xty[i] += w * row[i] * y
            for j in range(k):
                xtx[i][j] += w * row[i] * row[j]
    return solve_linear(xtx, xty)


def round2(value):
    """Round to 2 decimals, half up (as JavaScript's Math.round does).

    Whole numbers come back as ints, so the JSON writes 1240, not 1240.0.
    """
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100.0
    return int(rounded) if rounded.is_integer() else rounded


def target_is_console_only(notes):
    """True when the target's notes describe a console-only launch.

    The notes name the platforms before the words "real unlock"; a
    console-only launch either says so outright or names no PC release
    there. (GTA VI's notes say PS5 + Xbox Series X|S only.)
    """
    segment = (notes or "").lower().split("real unlock")[0][:250]
    return "console only" in segment or "pc" not in segment


def baseline_level(points, day_start):
    """A launch's quiet pre-launch level: the median hourly viewers
    over days -14..-3 before the release day.

    The window widens when the history is short (the live recorder
    only started a couple of days before the final's release day):
    first days -14..-1, then every point before the release day.
    Returns 1.0 when there is no usable history at all.
    """
    windows = [
        (day_start - timedelta(days=14), day_start - timedelta(days=3)),
        (day_start - timedelta(days=14), day_start - timedelta(days=1)),
        (None, day_start),
    ]
    for start, end in windows:
        values = [v for t, v in points.items()
                  if (start is None or t >= start) and t < end]
        if len(values) >= 12:
            return max(1.0, median(values))
    values = [v for t, v in points.items() if t < day_start]
    if values:
        return max(1.0, median(values))
    return 1.0


def hourly_points(rows):
    """Collapse hourly rows into {timestamp: viewers}, dropping empties."""
    points = {}
    for row in rows or []:
        viewers = row.get("viewers")
        if viewers:
            points[parse_ts(row["from"])] = viewers
    return points


def comparable_features(comp):
    """The forecasting features of one past launch, or None when it has
    no usable release day (no hour-1 print)."""
    day_start = parse_ts(comp["day_start"])
    points = hourly_points(comp.get("hourly"))
    day = {}
    for hour in range(1, 25):
        value = points.get(day_start + timedelta(hours=hour - 1))
        if value:
            day[hour] = value
    if 1 not in day:
        return None
    hour1 = day[1]
    base = baseline_level(points, day_start)
    unlock = (comp.get("unlock") or "").lower()
    platforms = (comp.get("platforms") or "").lower()
    return {
        "hour1": hour1,
        "a": math.log(hour1 / base),
        "early_access": "early-access" in unlock,
        "console_only": "pc" not in platforms,
        "day": day,
    }


def target_features(case):
    """The same features for the launch being forecast."""
    day_start = parse_ts(case["day_start"])
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    hour1 = max(1, first["viewers"])
    points = hourly_points(case.get("history_hourly"))
    base = baseline_level(points, day_start)
    return {
        "hour1": hour1,
        "a": math.log(hour1 / base),
        "console_only": target_is_console_only(case.get("notes")),
    }


def similarity_weight(comp, target):
    """How much one comparable should count for this target: a Gaussian
    kernel on (a, log hour-1 print), with tier and platform adjustments."""
    da = (comp["a"] - target["a"]) / BW_A
    dl = (math.log(comp["hour1"]) - math.log(target["hour1"])) / BW_LH1
    weight = math.exp(-0.5 * (da * da + dl * dl))
    if comp["early_access"]:
        weight *= EA_WEIGHT
    if target["console_only"] and comp["console_only"]:
        weight *= CONSOLE_UP
    return weight


def model_forecast(case):
    """The comparable-launch forecast: {hour: [q05, q25, q50, q75, q95]}."""
    target = target_features(case)
    comps = []
    for comp in case.get("comparables") or []:
        features = comparable_features(comp)
        if features is not None:
            comps.append(features)
    if len(comps) < 5:
        raise ValueError("too few usable comparables")
    p1 = float(target["hour1"])
    log_p1 = math.log(p1)
    out = {}
    for hour in range(2, 25):
        design, values, weights = [], [], []
        for comp in comps:
            if hour not in comp["day"]:
                continue
            design.append([1.0, math.log(comp["hour1"])])
            values.append(math.log(comp["day"][hour]))
            weights.append(similarity_weight(comp, target))
        if len(values) < 5 or sum(weights) <= 0.0:
            raise ValueError(f"too few comparables cover hour {hour}")
        beta = weighted_least_squares(design, values, weights)
        if beta is None:
            # Singular fit: fall back to the weighted mean log-ratio.
            ratios = [math.log(comp["day"][hour] / comp["hour1"])
                      for comp in comps if hour in comp["day"]]
            pred = log_p1 + (sum(w * r for w, r in zip(weights, ratios))
                             / sum(weights))
            spread = 0.5
        else:
            slope = min(max(beta[1], SLOPE_MIN), SLOPE_MAX)
            if slope != beta[1]:
                # Clamped slope: re-center the intercept so the fit
                # still passes through the weighted mean of the data.
                total = sum(weights)
                mean_x = sum(w * row[1]
                             for row, w in zip(design, weights)) / total
                mean_y = sum(w * y
                             for y, w in zip(values, weights)) / total
                beta = [mean_y - slope * mean_x, slope]
            pred = beta[0] + beta[1] * log_p1
            residuals = [y - (beta[0] + beta[1] * row[1])
                         for row, y in zip(design, values)]
            total = sum(weights)
            mean_resid = sum(w * r for w, r in zip(weights, residuals)) / total
            var = sum(w * (r - mean_resid) ** 2
                      for w, r in zip(weights, residuals)) / total
            spread = math.sqrt(max(var, 0.0))
        spread = max(SD_FLOOR, spread)
        out[hour] = [math.exp(pred + z * spread) for z in Z]
    return out


def persistence_forecast(p1):
    """Fallback: every hour draws hour 1's print, spread widening like
    a random walk in log space. Always valid, never crashes."""
    p1 = max(1.0, float(p1))
    out = {}
    for hour in range(2, 25):
        spread = PERSIST_SIGMA * math.sqrt(hour - 1)
        out[hour] = [p1 * math.exp(z * spread) for z in Z]
    return out


def clean_quantiles(raw):
    """Clamp to (0, 1e9], force non-decreasing, round to 2 decimals."""
    cleaned = []
    floor = 0.01
    for value in raw:
        value = min(max(value, floor), 1e9)
        if not math.isfinite(value):
            value = floor
        cleaned.append(value)
        floor = value
    return [round2(v) for v in cleaned]


def forecast_case(case):
    """One case's output object. Never raises: any model failure falls
    back to the persistence forecast."""
    try:
        first = next((k for k in case["known"] if k["hour"] == 1), None)
        p1 = max(1, first["viewers"]) if first else 1
    except Exception:
        p1 = 1
    try:
        forecast = model_forecast(case)
    except Exception:
        forecast = persistence_forecast(p1)
    hours = []
    for hour in range(2, 25):
        hours.append({"hour": hour, "q": clean_quantiles(forecast[hour])})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            result = forecast_case(case)
        except Exception:
            # Even an unreadable case file gets a valid-shaped output.
            result = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]}
                                for h in range(2, 25)]}
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
