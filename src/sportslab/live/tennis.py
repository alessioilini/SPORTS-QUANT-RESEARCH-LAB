"""Exact point-level Markov model for tennis (iid points on serve).

s_h = P(home player wins a point on own serve), s_a likewise for away.
Assumptions (stated): points iid given server; tiebreak at 6-6 in every set
(first to 7, win by 2); future sets start with either server w.p. 1/2.
"""
from __future__ import annotations

from functools import lru_cache

from scipy.optimize import brentq


def p_game(p: float, a: int = 0, b: int = 0) -> float:
    """P(server wins game) from server points a, receiver points b."""
    @lru_cache(None)
    def g(a, b):
        if a >= 4 and a - b >= 2:
            return 1.0
        if b >= 4 and b - a >= 2:
            return 0.0
        if a >= 3 and b >= 3:
            d = p * p / (p * p + (1 - p) * (1 - p))
            if a == b:
                return d
            return p + (1 - p) * d if a > b else p * d
        return p * g(a + 1, b) + (1 - p) * g(a, b + 1)
    return g(a, b)


def p_tiebreak(s_h: float, s_a: float, h: int = 0, a: int = 0, first_server_home: bool = True) -> float:
    """P(home wins tiebreak) from (h, a) points; server of point k: first server if ((k+1)//2) even."""
    def win_point(k):
        home_serving = (((k + 1) // 2) % 2 == 0) == first_server_home
        return s_h if home_serving else 1 - s_a

    @lru_cache(None)
    def t(h, a):
        if h >= 7 and h - a >= 2:
            return 1.0
        if a >= 7 and a - h >= 2:
            return 0.0
        if h >= 6 and a >= 6 and h == a:
            q1, q2 = win_point(h + a), win_point(h + a + 1)
            w, l = q1 * q2, (1 - q1) * (1 - q2)
            return w / (w + l)
        q = win_point(h + a)
        return q * t(h + 1, a) + (1 - q) * t(h, a + 1)
    return t(h, a)


def p_set(s_h: float, s_a: float, gh: int = 0, ga: int = 0, home_serving: bool = True,
          ph: int = 0, pa: int = 0, in_tiebreak: bool = False) -> float:
    """P(home wins current set) from games (gh, ga), current game points (ph, pa)."""
    hold_h, hold_a = p_game(s_h), p_game(s_a)

    @lru_cache(None)
    def st(gh, ga, home_srv):
        if (gh >= 6 and gh - ga >= 2) or gh == 7:
            return 1.0
        if (ga >= 6 and ga - gh >= 2) or ga == 7:
            return 0.0
        if gh == 6 and ga == 6:
            return p_tiebreak(s_h, s_a, 0, 0, home_srv)
        q = hold_h if home_srv else 1 - hold_a
        return q * st(gh + 1, ga, not home_srv) + (1 - q) * st(gh, ga + 1, not home_srv)

    if in_tiebreak or (gh == 6 and ga == 6):
        k = ph + pa   # index of the next point; derive who served point 0 of the tiebreak
        first_home = home_serving if ((k + 1) // 2) % 2 == 0 else not home_serving
        return p_tiebreak(s_h, s_a, ph, pa, first_home)
    if ph == 0 and pa == 0:
        return st(gh, ga, home_serving)
    # mid-game: points counted from the server's perspective
    if home_serving:
        q = p_game(s_h, ph, pa)
    else:
        q = 1 - p_game(s_a, pa, ph)
    return q * st(gh + 1, ga, not home_serving) + (1 - q) * st(gh, ga + 1, not home_serving)


def p_match(s_h: float, s_a: float, best_of: int = 3, sh: int = 0, sa: int = 0, gh: int = 0, ga: int = 0,
            home_serving: bool = True, ph: int = 0, pa: int = 0, in_tiebreak: bool = False) -> float:
    need = best_of // 2 + 1
    if sh >= need:
        return 1.0
    if sa >= need:
        return 0.0
    cur = p_set(s_h, s_a, gh, ga, home_serving, ph, pa, in_tiebreak)
    fut = 0.5 * (p_set(s_h, s_a, 0, 0, True) + p_set(s_h, s_a, 0, 0, False))

    @lru_cache(None)
    def m(sh, sa):
        if sh >= need:
            return 1.0
        if sa >= need:
            return 0.0
        return fut * m(sh + 1, sa) + (1 - fut) * m(sh, sa + 1)
    return cur * m(sh + 1, sa) + (1 - cur) * m(sh, sa + 1)


def serve_probs_from_prematch(p_home: float, best_of: int = 3, base: float = 0.64) -> tuple[float, float]:
    """Invert a pre-match win probability into (s_h, s_a) = (base+δ, base−δ)."""
    p_home = min(max(p_home, 0.005), 0.995)
    f = lambda d: p_match(base + d, base - d, best_of) - p_home
    d = brentq(f, -0.3, 0.3)
    return base + d, base - d
