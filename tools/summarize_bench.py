"""Benchmark metrics of one or more episodes.csv files.

Usage: summarize_bench.py results/same.csv results/mild.csv results/large.csv

Rates are shares of all episodes. APE (m), AOE (deg) and APT (s) are means over the
episodes that ended in the target slot.
"""
import argparse
import csv
from pathlib import Path

OUTCOMES = dict(TSR='target_success', TFR='target_fail', NTSR='non_target_success', NTFR='non_target_fail',
                CR='collision', OR='outbound', TR='timeout')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    print('| run | episodes | '+' | '.join(OUTCOMES)+' | APE | AOE | APT |')
    print('|---|---|'+'---|'*(len(OUTCOMES)+3))
    for path in args.files:
        with open(path, newline='') as stream:
            rows = list(csv.DictReader(stream))
        rates = [sum(row['outcome'] == name for row in rows)/len(rows) for name in OUTCOMES.values()]
        done = [row for row in rows if row['outcome'] == 'target_success']
        means = [sum(float(row[key]) for row in done)/max(len(done), 1) for key in ('position_error', 'yaw_error', 'seconds')]
        print(f'| {path.stem} | {len(rows)} | '+' | '.join(f'{rate:.3f}' for rate in rates)
              +f' | {means[0]:.3f} | {means[1]:.2f} | {means[2]:.1f} |')


if __name__ == '__main__':
    main()
