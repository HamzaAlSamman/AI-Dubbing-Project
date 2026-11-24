"""Score every WMT24++ (speech, en->ar_SA) system with the same reference metrics in one evaluation.

Generations are NOT made here: they come from an attached Kaggle dataset of resumable caches
(<generator name>.jsonl, written by benchmark.generate_cached on Kaggle or locally for API models).
Outputs in /kaggle/working/evaluations/<eval_id>/:
  reference.json            per-system summary + per-segment rows (chrF++, BLEU, MetricX-ref, length)
  paired_metricx.json       paired bootstrap on MetricX-ref for every pair of systems
  consistency_vs_kaggle.json  |new - earlier| MetricX for systems already scored in ev_..._wmt24pp_speech
Local use: python score_reference.py <wmt jsonl> <work dir> <repo dir> <gen dir> [...]"""
import glob
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if len(sys.argv) >= 5:
    DATA, WORK, REPO = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    GEN_DIRS = [Path(p) for p in sys.argv[4:]]
else:                                                        # Kaggle kernel
    WORK, REPO = Path('/kaggle/working'), Path('/kaggle/working/repo')
    if not REPO.exists():
        subprocess.run(['git', 'clone', '-q', '-b', os.environ.get('BENCH_BRANCH', 'main'),
                        'https://github.com/HamzaAlSamman/AI-Dubbing-Project.git', str(REPO)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sacrebleu', 'sentencepiece', 'num2words'],
                   check=True)
    DATA = WORK / 'en-ar_SA.jsonl'
    if not DATA.exists():
        urllib.request.urlretrieve('https://huggingface.co/datasets/google/wmt24pp/resolve/main/en-ar_SA.jsonl', DATA)
    GEN_DIRS = sorted({Path(f).parent for f in glob.glob('/kaggle/input/**/*__*.jsonl', recursive=True)})
sys.path.insert(0, str(REPO / 'translation'))
from dubbing_translation.benchmark import load_wmt24pp, run_reference  # noqa: E402
from dubbing_translation.evaluate import Evaluation  # noqa: E402
from dubbing_translation.io import load_config  # noqa: E402
from dubbing_translation.scoring import MetricXRef  # noqa: E402

FILES = {'hy_mt15_7b': 'tencent_HY-MT1.5-7B__plain_v1.jsonl',
         'hy_mt2_7b': 'tencent_Hy-MT2-7B__hy_mt2_g1s0.jsonl',
         'qwen3_4b_baseline': 'Qwen_Qwen3-4B__msa_plain_g1s0.jsonl',
         'gemini35_flash_lite': 'gemini__gemini-3.5-flash-lite__isometric_v1.jsonl',
         'gemini35_flash': 'gemini__gemini-3.5-flash__isometric_v1.jsonl'}

CFG = load_config()
rows = [json.loads(l) for l in DATA.read_text(encoding='utf-8').splitlines() if l.strip()]
items = load_wmt24pp(rows, domains=('speech',))
ids = {i['unit_id'] for i in items}
systems, prov = {}, {}
for name, fname in FILES.items():
    path = next((d / fname for d in GEN_DIRS if (d / fname).exists()), None)
    if path is None:
        print('missing', name)
        continue
    out = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec['items'] and rec['unit_id'] in ids:
                out[rec['unit_id']] = rec['items'][0]['text']     # same pick as generate_cached
    systems[name], prov[name] = out, fname[:-len('.jsonl')]
    print(name, len(out), 'of', len(items), flush=True)

ref_metric = MetricXRef(CFG['models']['qe'])
ev = Evaluation(WORK, items, CFG, label='wmt24pp_speech_gemini', repo_dir=str(REPO), qe_name=ref_metric.name)
t0 = time.time()
summary = run_reference(ev, items, systems, rate=CFG['voice']['letters_per_second'], ref_scorer=ref_metric,
                        provenance=prov, test_set='wmt24pp_en-ar_SA_speech_original_target')
ref_metric.close()
print('scored in', round(time.time() - t0), 's', flush=True)

ref = json.loads((ev.dir / 'reference.json').read_text(encoding='utf-8'))
mx = {}
for r in ref['rows']:
    if r.get('metricx_ref') is not None:
        mx.setdefault(r['system'], {})[r['unit_id']] = r['metricx_ref']


def paired(a, b, n_boot=2000, seed=0):
    common = sorted(set(mx[a]) & set(mx[b]))
    d = [mx[b][u] - mx[a][u] for u in common]            # > 0: a is better (lower MetricX)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {'mean_diff_b_minus_a': round(sum(d) / len(d), 3),
            'ci95': [round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot) - 1], 3)],
            'a_better_by_0.5': sum(x > 0.5 for x in d), 'b_better_by_0.5': sum(x < -0.5 for x in d),
            'within_0.5': sum(abs(x) <= 0.5 for x in d), 'n': len(d)}


names = [n for n in FILES if n in mx]
pairs = {f'{a}_vs_{b}': paired(a, b) for i, a in enumerate(names) for b in names[i + 1:]}
(ev.dir / 'paired_metricx.json').write_text(json.dumps(
    {'metric': ref_metric.name + ' (lower is better)', 'bootstrap': 2000, 'seed': 0, 'pairs': pairs},
    ensure_ascii=False, indent=1), encoding='utf-8')

old_path = REPO / 'translation' / 'results' / 'wmt24pp_speech' / 'reference.json'
check = {'compared': 0}
if old_path.exists():
    old = {(r['system'], r['unit_id']): r['metricx_ref']
           for r in json.loads(old_path.read_text(encoding='utf-8'))['rows'] if r.get('metricx_ref') is not None}
    diffs = [abs(v - old[(s, u)]) for s in mx for u, v in mx[s].items() if (s, u) in old]
    if diffs:
        check = {'compared': len(diffs), 'max_abs_diff': round(max(diffs), 4),
                 'mean_abs_diff': round(sum(diffs) / len(diffs), 5)}
(ev.dir / 'consistency_vs_kaggle.json').write_text(json.dumps(check, indent=1), encoding='utf-8')
print('consistency vs earlier run:', check)
cols = ['units', 'chrf_pp', 'bleu', 'metricx_ref_mean', 'len_ratio_ref_median', 'lc10_pct',
        'fits_or_speedup_pct', 'too_long_pct', 'too_short_pct']
lines = ['| system | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
for n, s in summary.items():
    lines.append(f'| {n} | ' + ' | '.join(str(s.get(c, '')) for c in cols) + ' |')
(ev.dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
print('EVAL_DIR', ev.dir)
