import numpy as np

from htmek_processing.analysis.exchange import fit_exchange_joint


def _curve(t, s, a0, ainf, kx, p0=5.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    A = ainf * t + (a0 - ainf) * (1 - np.exp(-kx * t)) / kx if kx else ainf * t
    return p0 + s * (1 - np.exp(-A)) + rng.normal(0, noise, len(t))


def test_exchange_found_only_where_it_is():
    t = np.arange(20) * 63.0 + 30
    S = 300.0
    curves = {"K": (t, _curve(t, S, 1e-4, 1e-4, 0, seed=1)),                 # single rate (the control)
              "NaK": (t, _curve(t, S, 2e-4, 0.8e-4, np.log(2) / 180, seed=2)),  # falls 2.5x, half-life 3 min
              "Na": (t, _curve(t, S, 0.6e-4, 0.6e-4, 0, seed=3))}
    rows = {r["run"]: r for r in fit_exchange_joint(curves, ["K", "NaK", "Na"])}
    assert rows["NaK"]["d_aic"] > 10
    assert abs(rows["NaK"]["a_inf"] / rows["NaK"]["a0"] - 0.4) < 0.1
    assert abs(np.log(2) / rows["NaK"]["k_x"] / 180 - 1) < 0.35
    assert rows["K"]["d_aic"] < 4 and rows["Na"]["d_aic"] < 4


def test_mm_turnover_recovers_exchange():
    from htmek_processing.analysis.exchange import substrate_left
    t = np.arange(20) * 63.0 + 30
    sigma = 300 / 1400   # [S] / K_m
    def curve(a0, ainf, kx, seed):
        A = ainf * t + ((a0 - ainf) * (1 - np.exp(-kx * t)) / kx if kx else 0)
        return 3 + 350 * (1 - substrate_left(A, sigma)) + np.random.default_rng(seed).normal(0, 0.3, len(t))
    # fast chambers (deplete within the run): a falls 2.5x in NaK only
    curves = {"K": (t, curve(2e-3, 2e-3, 0, 1)), "NaK": (t, curve(3e-3, 1.2e-3, np.log(2) / 150, 2)), "Na": (t, curve(1e-3, 1e-3, 0, 3))}
    rows = {r["run"]: r for r in fit_exchange_joint(curves, ["K", "NaK", "Na"], sigma=sigma)}
    assert rows["NaK"]["d_aic"] > 10 and rows["K"]["d_aic"] < 4 and rows["Na"]["d_aic"] < 4
    assert abs(rows["NaK"]["a_inf"] / rows["NaK"]["a0"] - 0.4) < 0.1
    assert abs(rows["K"]["S_hat"] / 350 - 1) < 0.05
    # first order is the K_m >> S limit
    A = np.linspace(0, 5, 11)
    assert np.allclose(substrate_left(A, 1e-6), np.exp(-A), rtol=1e-4)
