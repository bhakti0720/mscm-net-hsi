"""
experiments.py - run a configuration over several seeds, save results to Drive, compare to baseline.
Results go to {R}/results/{tag}_seed{k}.npz (finished seeds are skipped, so reruns resume).
"""
import os, numpy as np
import train


def run_config(tag, data_path, gt_path, R, cache_name='indian_pines', seeds=range(10),
               epochs=100, **kw):
    """kw can contain: weight_decay, augment, dropout (model arg), etc."""
    os.makedirs(f'{R}/results', exist_ok=True)
    for s in seeds:
        out = f'{R}/results/{tag}_seed{s}.npz'
        if os.path.exists(out):
            print(f'[{tag}] seed {s} done, skipping'); continue
        res = train.run(data_path, gt_path, name=cache_name, tag=tag, seed=s, epochs=epochs,
                        cache_dir=f'{R}/data/cache', ckpt_dir=f'{R}/checkpoints', **kw)
        np.savez(out, OA=res['OA'], AA=res['AA'], Kappa=res['Kappa'],
                 per_class=res['per_class'], cm=res['cm'])


def _load(R, tag, metric, seeds):
    return np.array([float(np.load(f'{R}/results/{tag}_seed{s}.npz')[metric]) for s in seeds])


def compare(R, tag, baseline='indian_pines', seeds=range(10)):
    """Paired comparison: same seed = same train/test split in both configs."""
    seeds = [s for s in seeds
             if os.path.exists(f'{R}/results/{tag}_seed{s}.npz')
             and os.path.exists(f'{R}/results/{baseline}_seed{s}.npz')]
    for m in ('OA', 'AA', 'Kappa'):
        a, b = _load(R, baseline, m, seeds), _load(R, tag, m, seeds)
        d = b - a
        print(f'{m:6s} {tag}: {b.mean():.2f} ± {b.std():.2f}   baseline: {a.mean():.2f} ± {a.std():.2f}   '
              f'diff {d.mean():+.2f} (sd of diff {d.std():.2f}, better on {(d > 0).sum()}/{len(d)} seeds)')
