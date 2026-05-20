import unittest

import numpy as np

from cuneate.cn_experiments import ExperimentConfig, run_cn_trial_set
from cuneate.cn_simulation import AssignmentConfig, CNLIFConfig, assign_afferents_to_cn, simulate_cn_lif


def _mock_resp(n_aff=24, t_stop=0.5, seed=0):
    rng = np.random.default_rng(seed)
    loc = rng.uniform(-5, 5, size=(n_aff, 2))
    classes = np.array(["PC", "RA", "SA1"] * (n_aff // 3) + ["PC"] * (n_aff % 3), dtype=object)
    patch_id = np.zeros(n_aff, dtype=int)
    patch_id[: n_aff // 2] = 1
    patch_id[n_aff // 2 :] = 2
    spikes = []
    for _ in range(n_aff):
        n = int(rng.integers(2, 10))
        spikes.append(np.sort(rng.uniform(0.0, t_stop, size=n)))
    return {
        "spikes": spikes,
        "location": loc,
        "class_str": classes,
        "patch_ID": patch_id,
    }


class TestCNPipeline(unittest.TestCase):
    def test_assignment_respects_input_range(self):
        resp = _mock_resp()
        cfg = AssignmentConfig(min_inputs=5, max_inputs=9, strategy="biased")
        assn = assign_afferents_to_cn(resp, n_cn_per_patch=2, assignment_cfg=cfg, seed=1)
        self.assertTrue(len(assn) > 0)
        for row in assn:
            n = len(row["afferent_indices"])
            self.assertGreaterEqual(n, 5)
            self.assertLessEqual(n, 9)

    def test_lif_produces_spikes(self):
        resp = _mock_resp()
        assn = assign_afferents_to_cn(resp, n_cn_per_patch=1, assignment_cfg=AssignmentConfig(), seed=2)
        lif = CNLIFConfig(v_threshold=0.5, current_gain=1.5)
        out = simulate_cn_lif(resp, assn, lif_cfg=lif, t_stop=0.5)
        self.assertEqual(len(out["cn_spikes"]), len(assn))
        self.assertTrue(all(isinstance(x, np.ndarray) for x in out["cn_spikes"]))

    def test_dual_mi_computes(self):
        base = _mock_resp(seed=10)
        trials = []
        for i in range(8):
            r = dict(base)
            r["spikes"] = [s + (i * 0.001) for s in base["spikes"]]
            trials.append(r)
        stim = np.column_stack(
            [
                np.linspace(0.1, 1.0, len(trials)),
                np.linspace(10, 80, len(trials)),
                np.linspace(0.2, 1.2, len(trials)),
                np.linspace(-2, 2, len(trials)),
                np.linspace(-1, 1, len(trials)),
            ]
        )
        out = run_cn_trial_set(
            trials,
            stimulus_features=stim,
            assignment_cfg=AssignmentConfig(strategy="biased"),
            exp_cfg=ExperimentConfig(),
        )
        self.assertIn("mi", out)
        self.assertGreaterEqual(out["mi"]["mi_stimulus_cn"], 0.0)
        self.assertGreaterEqual(out["mi"]["mi_afferent_cn"], 0.0)


if __name__ == "__main__":
    unittest.main()
