"""Prediction-zone TVT baseline. No target or train-only geology enters features."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit


def features(raw):
    required = ['WELL_ID', 'MD', 'X', 'Y', 'Z', 'GR', 'TVT_input']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    # Explicit allowlist: TVT, formation labels and legacy DTW never enter X.
    d = raw[required].copy()
    if d.WELL_ID.isna().any():
        raise ValueError('Missing WELL_ID')
    d['WELL_ID'] = d.WELL_ID.astype(str)
    for col in required[1:]:
        d[col] = pd.to_numeric(d[col], errors='raise')
        if np.isinf(d[col]).any():
            raise ValueError(f'Infinite values in {col}')
    if d.MD.isna().any():
        raise ValueError('MD must be finite')
    d['source_row'] = np.arange(len(d))
    blocks, metadata = [], []
    for well, g in d.groupby('WELL_ID', sort=True):
        g = g.sort_values('MD').reset_index(drop=True)
        if g.MD.duplicated().any():
            raise ValueError(f'{well}: duplicate MD requires explicit resolution')
        known = g.TVT_input.notna().to_numpy()
        if not known.any():
            raise ValueError(f'{well}: no known TVT_input anchor')
        a = np.flatnonzero(known)[-1]
        if not known[:a+1].all():
            raise ValueError(f'{well}: TVT_input must be a contiguous known prefix')
        if a == len(g)-1:
            raise ValueError(f'{well}: no prediction zone; provide an explicit PS mask')
        anchor = float(g.TVT_input.iloc[a])
        recent = g.iloc[max(0, a-49):a+1]
        x = recent.MD.to_numpy() - recent.MD.iloc[-1]
        y = recent.TVT_input.to_numpy()
        slope = float(np.dot(x-x.mean(), y-y.mean()) / np.dot(x-x.mean(), x-x.mean())) if len(x)>1 else 0.0
        f = pd.DataFrame(index=g.index)
        f['anchor_tvt'] = anchor
        f['prefix_slope'] = slope
        f['distance_from_ps'] = g.MD - g.MD.iloc[a]
        f['linear_tvt'] = anchor + slope * f.distance_from_ps
        for c in ['X', 'Y', 'Z']:
            f[f'{c}_relative'] = g[c] - g[c].iloc[a]
            f[f'{c}_gradient'] = g[c].diff() / g.MD.diff()
        f['GR'] = g.GR
        f['GR_missing'] = g.GR.isna().astype(float)
        for lag in [1, 5, 20]:
            f[f'GR_lag_{lag}'] = g.GR.shift(lag)
        for window in [5, 20, 50]:
            f[f'GR_mean_{window}'] = g.GR.rolling(window, min_periods=1).mean()
            f[f'GR_std_{window}'] = g.GR.rolling(window, min_periods=2).std()
        zone = np.arange(len(g)) > a
        blocks.append(f.loc[zone])
        m = g.loc[zone, ['WELL_ID', 'MD', 'source_row']].copy()
        m['anchor_tvt'] = anchor
        m['linear_tvt'] = f.loc[zone, 'linear_tvt']
        metadata.append(m)
    if not blocks:
        raise ValueError('No wells found')
    return (pd.concat(blocks, ignore_index=True).replace([np.inf, -np.inf], np.nan),
            pd.concat(metadata, ignore_index=True))


def split_wells(meta, seed):
    if meta.WELL_ID.nunique() < 5:
        raise ValueError('At least five wells required for train/validation/test split')
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    dev, test = next(splitter.split(meta, groups=meta.WELL_ID))
    train_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=.25,
                         random_state=seed+1).split(meta.iloc[dev], groups=meta.WELL_ID.iloc[dev]))
    return dev[train_i], dev[val_i], test


def score(y, prediction, wells):
    err = pd.DataFrame({'well': np.asarray(wells), 'sq': (y-prediction)**2,
                        'abs': np.abs(y-prediction)})
    per_well = np.sqrt(err.groupby('well').sq.mean())
    return {'rmse': float(np.sqrt(err.sq.mean())), 'mae': float(err['abs'].mean()),
            'mean_well_rmse': float(per_well.mean()),
            'worst_well_rmse': float(per_well.max()), 'rows': len(y),
            'per_well_rmse': per_well.to_dict()}


def train(args):
    raw = pd.read_csv(args.data)
    if 'TVT' not in raw:
        raise ValueError('Training requires ground-truth TVT')
    X, meta = features(raw)
    y = pd.to_numeric(raw.TVT, errors='raise').to_numpy()[meta.source_row.to_numpy()]
    if not np.isfinite(y).all():
        raise ValueError('Every training prediction-zone row needs finite TVT')
    tr, va, te = split_wells(meta, args.seed)
    anchor = meta.anchor_tvt.to_numpy()
    residual = y-anchor
    candidates = {'constant_anchor': anchor, 'prefix_linear': meta.linear_tvt.to_numpy()}
    models = {}
    # External validation is grouped by well. Disable sklearn's random-row early stopping.
    for leaves, l2 in [(15, 10.0), (31, 30.0)]:
        name = f'residual_hgb_{leaves}'
        model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.05,
                    max_iter=args.iterations, max_leaf_nodes=leaves, min_samples_leaf=50,
                    l2_regularization=l2, early_stopping=False, random_state=args.seed)
        model.fit(X.iloc[tr], residual[tr])
        models[name] = model
        # Only validation predictions participate in model selection.
        candidates[name] = None
    validation = {}
    for name, pred in candidates.items():
        p = models[name].predict(X.iloc[va])+anchor[va] if pred is None else pred[va]
        validation[name] = score(y[va], p, meta.WELL_ID.iloc[va])
    winner = min(validation, key=lambda k: validation[k]['rmse'])
    if winner in models:
        model = models[winner]
        model.fit(X.iloc[np.r_[tr, va]], residual[np.r_[tr, va]])
        prediction = model.predict(X.iloc[te])+anchor[te]
    else:
        model = None
        prediction = candidates[winner][te]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {'seed': args.seed, 'target': 'TVT', 'zone': 'missing TVT_input after known prefix',
              'selected_on_validation': winner, 'validation': validation,
              'test': score(y[te], prediction, meta.WELL_ID.iloc[te]),
              'test_baselines': {k: score(y[te], candidates[k][te], meta.WELL_ID.iloc[te])
                                 for k in ['constant_anchor', 'prefix_linear']},
              'split_wells': {k: sorted(meta.WELL_ID.iloc[v].unique().tolist())
                              for k, v in [('train', tr), ('validation', va), ('test', te)]}}
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    joblib.dump({'model': model, 'winner': winner, 'features': X.columns.tolist()}, out/'model.joblib')
    p = meta.iloc[te].copy()
    p['TVT_actual'], p['TVT_prediction'] = y[te], prediction
    p.to_csv(out/'test_predictions.csv', index=False)
    print(json.dumps({k: report[k] for k in ['selected_on_validation', 'test']}, indent=2))


def predict(args):
    # Load only model files you trust (joblib uses pickle).
    bundle = joblib.load(args.model)
    X, meta = features(pd.read_csv(args.data))
    if X.columns.tolist() != bundle['features']:
        raise ValueError('Feature schema mismatch')
    if bundle['model'] is not None:
        p = bundle['model'].predict(X)+meta.anchor_tvt.to_numpy()
    else:
        p = meta['anchor_tvt' if bundle['winner']=='constant_anchor' else 'linear_tvt'].to_numpy()
    meta['TVT_prediction'] = p
    meta.sort_values('source_row').to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--data', required=True)
    t.add_argument('--output', default='runs/residual')
    t.add_argument('--iterations', type=int, default=300)
    t.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('predict')
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()
    if args.command == 'train' and args.iterations < 1:
        parser.error('--iterations must be positive')
    (train if args.command == 'train' else predict)(args)
