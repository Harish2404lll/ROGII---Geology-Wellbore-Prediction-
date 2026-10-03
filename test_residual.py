import unittest
import numpy as np
import pandas as pd
from train_residual import features, split_wells


def sample():
    frames = []
    for w in range(10):
        md = np.arange(100, dtype=float)
        target = 1000+w*10+.1*md
        frames.append(pd.DataFrame({'WELL_ID': str(w), 'MD': md, 'X': md,
            'Y': md*0, 'Z': md*.1, 'GR': np.sin(md/10)*20+50,
            'TVT_input': np.where(md<50, target, np.nan), 'TVT': target,
            'ANCC': target+3}))
    return pd.concat(frames, ignore_index=True)


class SafetyTests(unittest.TestCase):
    def test_target_and_formation_never_affect_features(self):
        raw = sample()
        x, m = features(raw)
        raw['TVT'] = -999999
        raw['ANCC'] = np.nan
        raw['DTW_TVT'] = 999999
        x2, m2 = features(raw)
        pd.testing.assert_frame_equal(x, x2)
        pd.testing.assert_frame_equal(m, m2)
        self.assertEqual(len(x), 500)
        self.assertTrue((m.MD >= 50).all())

    def test_inference_without_target(self):
        raw = sample()
        a, _ = features(raw)
        b, _ = features(raw.drop(columns=['TVT', 'ANCC']))
        pd.testing.assert_frame_equal(a, b)

    def test_future_observations_do_not_change_earlier_features(self):
        raw = sample()
        a, _ = features(raw)
        raw.loc[raw.MD>80, ['GR', 'X', 'Y', 'Z']] = 777
        b, meta = features(raw)
        pd.testing.assert_frame_equal(a.loc[meta.MD<=80], b.loc[meta.MD<=80])

    def test_disjoint_wells(self):
        _, meta = features(sample())
        splits = split_wells(meta, 42)
        sets = [set(meta.WELL_ID.iloc[i]) for i in splits]
        for i in range(3):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])
        self.assertEqual(sum(len(i) for i in splits), len(meta))

    def test_missing_gr_preserves_prediction_rows(self):
        raw = sample()
        raw.loc[raw.MD==75, 'GR'] = np.nan
        x, _ = features(raw)
        self.assertEqual(len(x), 500)
        self.assertEqual(x.GR_missing.sum(), 10)

    def test_ambiguous_prefix_is_rejected(self):
        raw = sample()
        raw.loc[0, 'TVT_input'] = np.nan
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            features(raw)


if __name__ == '__main__':
    unittest.main()
