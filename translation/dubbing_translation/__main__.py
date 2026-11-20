"""Command line entry point.

  python -m dubbing_translation run --tasks tasks.json --work-dir /path/lesson/translation \
         [--master master_source_only.json] [--config configs/translation_v1.toml] \
         [--replay NAME=FOLDER ...] [--no-hymt] [--no-llm] [--no-refine] [--label text]
  python -m dubbing_translation runs --work-dir /path/lesson/translation       # compare runs
"""
import argparse
import json
import sys
from pathlib import Path

from .io import DEFAULT_CONFIG, load_config, load_tasks, load_windows


def build_and_run(args, log=print):
    from .generators import HyMT, IsometricLLM, LLMClient, Replay
    from .pipeline import run
    from .scoring import CachedQE, Hunspell, MetricXQE

    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    windows = load_windows(tasks, args.master)
    m = cfg['models']
    gens = [Replay(folder, name) for name, folder in (r.split('=', 1) for r in args.replay)]
    if not args.no_hymt:
        gens.append(HyMT(m['mt']))
    if not args.no_llm:
        gens.append(IsometricLLM(LLMClient(m['llm_provider'], m['llm_model'], min_interval_s=m['llm_min_interval_s'],
                                           timeout_s=m.get('llm_timeout_s', 180))))
    from .colab import refine_client
    client = None if args.no_refine else refine_client(m)
    if args.no_refine:
        cfg['refine']['enabled'] = False
    from .scoring import SegmentedQE
    seg = cfg['gate'].get('qe_segment_min_words', 0)
    metric = MetricXQE(m['qe'])
    qe = CachedQE(SegmentedQE(metric, seg) if seg else metric, Path(args.work_dir) / 'cache' / 'qe_metricx24.jsonl')
    repo = Path(__file__).resolve().parents[2]
    ranker = None
    if cfg.get('rank', {}).get('enabled'):
        ranker = LLMClient(cfg['rank']['provider'], cfg['rank']['model'], min_interval_s=m['llm_min_interval_s'],
                           timeout_s=m.get('llm_timeout_s', 180))
    return run(tasks, windows, cfg, args.work_dir, generators=gens, qe=qe, speller=Hunspell(),
               refine_client=client, rank_client=ranker, repo_dir=repo, run_label=args.label, log=log)


def show_runs(work_dir):
    path = Path(work_dir) / 'registry.jsonl'
    if not path.exists():
        print('no runs yet')
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        mt = r['metrics']
        print(f"{r['run_id']:45} acc {mt['accepted']}/{mt['units']} ({mt['accepted_pct']}%) "
              f"refine {mt['accepted_after_refine']} QE {mt['qe_mean']} cfg {r['config_sha256'][:8]} "
              f"git {str(r['git_commit'])[:12]}")


def main(argv=None):
    p = argparse.ArgumentParser(prog='dubbing_translation')
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--tasks', required=True)
    r.add_argument('--work-dir', required=True)
    r.add_argument('--master')
    r.add_argument('--config', default=str(DEFAULT_CONFIG))
    r.add_argument('--replay', action='append', default=[], help='NAME=FOLDER of saved per-unit outputs')
    r.add_argument('--no-hymt', action='store_true')
    r.add_argument('--no-llm', action='store_true')
    r.add_argument('--no-refine', action='store_true')
    r.add_argument('--label', default='')
    s = sub.add_parser('runs')
    s.add_argument('--work-dir', required=True)
    args = p.parse_args(argv)
    if args.cmd == 'run':
        build_and_run(args)
    else:
        show_runs(args.work_dir)


if __name__ == '__main__':
    sys.exit(main())
