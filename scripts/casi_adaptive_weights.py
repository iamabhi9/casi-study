"""
CASI — Adaptive Weighting Engine

Bayesian recalibration of component weights per project.
Starts from Delphi priors and updates based on observed failure patterns.

Usage:
    from casi_adaptive_weights import AdaptiveWeightEngine
    engine = AdaptiveWeightEngine()
    new_weights = engine.update(component_values, n_failures)
"""

import numpy as np

DELPHI_WEIGHTS = np.array([0.22, 0.18, 0.17, 0.16, 0.14, 0.13])
COMPONENT_NAMES = ['A_broken_index', 'B_avg_fix_time', 'C_downtime',
                   'D_failed_tc_ratio', 'E_failed_suite_ratio', 'F_variances_taken']


class AdaptiveWeightEngine:
    def __init__(self, damping=0.6, bounds=(0.05, 0.40), warm_up=5):
        self.damping = damping; self.bounds = bounds; self.warm_up = warm_up
        self.weights = DELPHI_WEIGHTS.copy() / DELPHI_WEIGHTS.sum()
        self.release_count = 0; self.history = []

    def update(self, component_values, n_failures):
        self.release_count += 1
        v = np.array(component_values, dtype=float)
        degradation = 100 - v
        if n_failures > 0 and degradation.sum() > 0:
            raw_signal = degradation * n_failures
            signal_weights = raw_signal / raw_signal.sum()
        else:
            signal_weights = self.weights
        if self.release_count <= self.warm_up:
            blend = self.damping + (1 - self.damping) * (self.release_count / self.warm_up)
            new_w = blend * DELPHI_WEIGHTS / DELPHI_WEIGHTS.sum() + (1 - blend) * signal_weights
        else:
            new_w = self.damping * self.weights + (1 - self.damping) * signal_weights
        new_w = np.clip(new_w, self.bounds[0], self.bounds[1])
        new_w = new_w / new_w.sum()
        self.history.append({'release': self.release_count, 'weights': new_w.copy(), 'n_failures': n_failures})
        self.weights = new_w; return self.weights

    def weight_shift(self):
        delphi = DELPHI_WEIGHTS / DELPHI_WEIGHTS.sum()
        return dict(zip(COMPONENT_NAMES, self.weights - delphi))

    def summary(self):
        delphi = DELPHI_WEIGHTS / DELPHI_WEIGHTS.sum()
        print("Component         Delphi    Adapted   Shift")
        for name, dw, aw in zip(COMPONENT_NAMES, delphi, self.weights):
            print(f"  {name:<20} {dw:.3f}   {aw:.3f}   {'\u25b2' if aw-dw>0.02 else ('\u25bc' if aw-dw<-0.02 else '\u2248')} {aw-dw:+.3f}")
