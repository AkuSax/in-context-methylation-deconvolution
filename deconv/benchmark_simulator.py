"""Run a small CPU benchmark of simulated tasks. No data files are written."""
import argparse
import numpy as np
from simulator import load_gse35069, split_donors, simulate_task
from solvers import nnls, cp_equality, rpc_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=int, default=30)
    parser.add_argument('--seed', type=int, default=712)
    parser.add_argument('--marker-mode', choices=('random', 'informative'), default='random')
    parser.add_argument('--clean-only', action='store_true')
    parser.add_argument('--rpc', action='store_true', help='include EpiDISH RPC')
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error('--tasks must be positive')
    panel = load_gse35069()
    train, validation = split_donors(panel, seed=0)
    print('source:', panel.source, '|', panel.preprocessing)
    print('train donors:', sorted(train.donors.unique()))
    print('validation donors:', sorted(validation.donors.unique()))
    rng = np.random.default_rng(args.seed)
    regimes = [('clean6', dict(n_types=6)),
               ('mismatch6', dict(n_types=6, donor_mismatch=True)),
               ('missing6', dict(n_types=6, missing=1, donor_mismatch=True)),
               ('synthetic12', dict(n_types=12, augment=True, donor_mismatch=True))]
    if args.clean_only:
        regimes = regimes[:1]
    print('marker mode:', args.marker_mode)
    for name, opts in regimes:
        errors = [[], []]
        tasks = [
            simulate_task(validation, rng, n_markers=300, marker_mode=args.marker_mode, **opts)
            for _ in range(args.tasks)
        ]
        for task in tasks:
            for error, solver in zip(errors, (nnls, cp_equality)):
                estimate = solver(task.reference_mean, task.bulk[:, None])[0]
                error.extend((estimate-task.fractions[:-1])**2)
        methods = ['NNLS', 'CP']
        if args.rpc:
            reference = np.stack([task.reference_mean for task in tasks])
            bulk = np.stack([task.bulk for task in tasks])
            truth = np.stack([task.fractions[:-1] for task in tasks])
            errors.append((rpc_batch(reference, bulk) - truth) ** 2)
            methods.append('RPC')
        scores = ', '.join(f'{method} {100 * np.sqrt(np.mean(error)):.3f}' for method, error in zip(methods, errors))
        print(f'{name} RMSE in percentage points: {scores}')


if __name__ == '__main__':
    main()
