import numpy as np
import pandas as pd

from statarb.data import synthetic_prices
from statarb.pairs import benjamini_hochberg, half_life, hedge_ratio, select_pairs


def test_half_life_recovers_known_ou():
    rng = np.random.default_rng(1)
    theta, n = 0.1, 20000
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] * (1 - theta) + rng.normal(0, 0.01)
    expected = -np.log(2) / np.log(1 - theta)   # ~6.6 days
    assert abs(half_life(pd.Series(s)) - expected) / expected < 0.1


def test_half_life_random_walk_is_long_or_inf():
    rw = pd.Series(np.cumsum(np.random.default_rng(2).normal(size=3000)))
    assert half_life(rw) > 100


def test_hedge_ratio_exact():
    x = pd.Series(np.linspace(0, 1, 100))
    a, b = hedge_ratio(0.3 + 1.7 * x, x)
    assert np.isclose(a, 0.3) and np.isclose(b, 1.7)


def test_bh_matches_hand_example():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    # thresholds alpha*k/n with alpha=0.05, n=8: 0.00625, 0.0125, 0.01875, ...
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True] + [False] * 6


def test_selection_finds_true_pairs_and_rejects_noise():
    # Engle-Granger has low power on short samples with slow reversion, so we
    # give it ~4 years here. (With 400 days it finds only 1 of 3: try it.)
    prices = synthetic_prices(n_days=1000, n_pairs=3, n_noise=6, seed=7)
    pairs = select_pairs(prices, min_corr=0.3, max_pairs=10)
    found = {frozenset((p.y, p.x)) for p in pairs}
    truth = {frozenset((f"P{i}A", f"P{i}B")) for i in range(3)}
    assert len(found & truth) >= 2
    noise_only = [f for f in found if all(t.startswith("N") for t in f)]
    assert not noise_only


def test_each_ticker_used_once():
    prices = synthetic_prices(n_days=500, n_pairs=4, n_noise=4, seed=3)
    pairs = select_pairs(prices, min_corr=0.0, correction="none")
    legs = [t for p in pairs for t in (p.y, p.x)]
    assert len(legs) == len(set(legs))
