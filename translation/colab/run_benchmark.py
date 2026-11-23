# Evaluation on Google Colab or Kaggle: (21) the baseline on our lecture under the SAME gate as the
# final system, and (22) a reference benchmark on a public test set with human MSA translations.
#
# Run after the cells of run_translation_step.py section 20.1 (repo, RUN_DIR, EXP_DIR, EXP_OUT).
# Secrets (GEMINI_API_KEY, HF_TOKEN) are loaded by the notebook's secrets cell, run by hand.
# Everything is written under RUN_DIR/translation_step/evaluations and registered in
# evaluations/registry.jsonl (one line per system and evaluation).

# %% 21.1 — baseline (run A) and single systems on the lecture, same gate/windows/MetricX as final_v1
import json, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sacrebleu', 'datasets'], check=True)
from dubbing_translation.evaluate import Evaluation, from_run, from_unit_folder   # noqa: E402
from dubbing_translation.io import load_config, load_tasks, load_windows          # noqa: E402
from dubbing_translation.scoring import CachedQE, Hunspell, MetricXQE, SegmentedQE  # noqa: E402

WORK = RUN_DIR / 'translation_step'                                     # noqa: F821
CFG = load_config()
tasks = load_tasks(EXP_DIR / 'tasks.json')                              # noqa: F821
windows = load_windows(tasks, RUN_DIR / 'master_source_only.json')      # noqa: F821
print(sorted(p.name for p in EXP_OUT.iterdir() if p.is_dir()))         # noqa: F821  (check folder names)

final_dir = sorted((WORK / 'runs').glob('*final_v1*'))[-1]
systems = {'final_v1': from_run(final_dir)}
for name in ('run_a', 'qwen3_4b', 'S7_best_of_12_habibi_gate', 'hy_mt15_7b_nocontext', 'gemini35_flash_lite_api'):
    folder = EXP_OUT / name                                             # noqa: F821
    if folder.exists():
        systems[name] = from_unit_folder(folder)
    else:
        print('missing system folder (rename in the list above):', name)

metric = MetricXQE(CFG['models']['qe'])
qe = CachedQE(SegmentedQE(metric, CFG['gate']['qe_segment_min_words']), WORK / 'cache' / 'qe_metricx24.jsonl')
ev = Evaluation(WORK, tasks, CFG, label='baseline_same_gate', repo_dir=TR_REPO, qe_name=qe.name)  # noqa: F821
summary = ev.run_automatic(windows, systems, qe, Hunspell(),
                           provenance={'final_v1': str(final_dir), **{k: f'experiments_v1/outputs/{k}'
                                                                     for k in systems if k != 'final_v1'}})
for name, s in sorted(summary.items(), key=lambda kv: -kv[1]['gate_accept_pct']):
    print(f"{name:32s} accept {s['gate_accept_pct']:5.1f}%  QE {s['qe_mean']}  fits {s['fits_natural_pct']}%  "
          f"too_long {s['too_long_pct']}%  coverage {s['coverage_pct']}%")
metric.close()

# %% 22.1 — public test set with human MSA references: WMT24++ en->ar, speech domain
from dubbing_translation.benchmark import as_tasks, generate_cached, load_wmt24pp_hf, run_reference  # noqa: E402

items = load_wmt24pp_hf('en-ar_SA', domains=('speech',))   # reference = original_target (MSA)
btasks = as_tasks(items)
BENCH = WORK / 'benchmark' / 'wmt24pp_speech'
RATE = CFG['voice']['letters_per_second']
print(len(items), 'segments')

# %% 22.2 — generate (one model on the GPU at a time; each generator resumes from its jsonl cache)
from dubbing_translation.generators import HFChatMT, HyMT, IsometricLLM, LLMClient   # noqa: E402

m = CFG['models']
GENERATORS = {
    'hy_mt15_7b':        lambda: HyMT(m['mt']),
    'hy_mt2_7b':         lambda: HFChatMT('tencent/Hy-MT2-7B', 'hy_mt2', samples=0),
    'qwen3_4b_baseline': lambda: HFChatMT('Qwen/Qwen3-4B', 'msa_plain', samples=0,
                                          template_kwargs={'enable_thinking': False}),
    'jais3_7b':          lambda: HFChatMT('inception42/Jais-3-7B', 'msa_plain', samples=0, trust_remote_code=True),
    'gemini_isometric':  lambda: IsometricLLM(LLMClient(m['llm_provider'], m['llm_model'],
                                                        min_interval_s=m['llm_min_interval_s'],
                                                        timeout_s=m.get('llm_timeout_s', 180))),
}
bench_systems, bench_prov = {}, {}
for name, make in GENERATORS.items():
    gen = make()
    try:
        bench_systems[name] = generate_cached(gen, btasks, BENCH / 'outputs', rate=RATE, budget_cfg=CFG['budget'])
        bench_prov[name] = gen.name
    except Exception as exc:                                   # noqa: BLE001 - keep the other systems
        print('FAILED', name, type(exc).__name__, exc)
    finally:
        if hasattr(gen, 'close'):
            gen.close()

# %% 22.3 — score against the references (chrF++, BLEU, MetricX-24 with reference, length, duration)
from dubbing_translation.scoring import MetricXRef   # noqa: E402

ref_metric = MetricXRef(m['qe'])
bev = Evaluation(WORK, items, CFG, label='wmt24pp_speech', repo_dir=TR_REPO, qe_name=ref_metric.name)  # noqa: F821
bsum = run_reference(bev, items, bench_systems, rate=RATE, ref_scorer=ref_metric, provenance=bench_prov,
                     test_set='wmt24pp_en-ar_SA_speech_original_target')
ref_metric.close()
for name, s in bsum.items():
    print(name, json.dumps(s, ensure_ascii=False))
