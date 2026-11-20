# Translation step on Google Colab — reproduces run `final_v1` and its independent check.
#
# Paste the sections below into notebook cells (one "# %%" block per cell) AFTER the notebook's own
# cells 1 (configuration) and 3 (Drive mount; defines RUN_DIR). Runtime: GPU (T4 is enough).
# Secrets: run the secrets cell yourself from the Colab UI (GEMINI_API_KEY, OPENROUTER_API_KEY, HF_TOKEN);
# secrets cannot be read by cells executed through automation.
#
# Data never leaves Drive: candidates, scores, runs and the registry are written under
#   RUN_DIR/translation_step/{cache,runs,registry.jsonl}

# %% 20.1 — code + dependencies
import json, subprocess, sys, threading, time
from pathlib import Path

REPO_URL = 'https://github.com/HamzaAlSamman/AI-Dubbing-Project.git'
TR_REPO, TR_BRANCH = Path('/content/translation_step_repo'), 'main'
if TR_REPO.exists():
    subprocess.run(['git', '-C', str(TR_REPO), 'fetch', '-q', 'origin', TR_BRANCH], check=True)
    subprocess.run(['git', '-C', str(TR_REPO), 'checkout', '-q', '-B', TR_BRANCH, 'origin/' + TR_BRANCH], check=True)
else:
    subprocess.run(['git', 'clone', '-q', '-b', TR_BRANCH, REPO_URL, str(TR_REPO)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'google-genai', 'num2words', 'sentencepiece',
                'bitsandbytes'], check=True)
sys.path.insert(0, str(TR_REPO / 'translation'))
from dubbing_translation import colab as trc                      # noqa: E402
EXP_DIR = RUN_DIR / 'experiments_v1'                              # noqa: F821  (RUN_DIR from notebook cell 3)
EXP_OUT = EXP_DIR / 'outputs'

# %% 20.2 — run the step (background thread; this cell polls until done)
# Generators: HY-MT1.5-7B (saved outputs of the same model + prompt are reused), Hy-MT2-7B (4-bit),
# Gemini 3.5 Flash Lite with an Arabic-letter budget. Gate -> guarded refine -> blind LLM ranking.
JOB = trc.start(RUN_DIR, replay={'hy_mt15_7b_nocontext': EXP_OUT / 'hy_mt15_7b_nocontext'},  # noqa: F821
                hf_models=[('tencent/Hy-MT2-7B', 'hy_mt2', 1)], use_llm=True, refine=True, rank=True,
                label='final_v1', seed_csv=sorted((EXP_DIR / 'evaluation').glob('units_*.csv'))[-1])
while not JOB['done']:
    print(time.strftime('%H:%M:%S'), '|', JOB['log'][-1] if JOB['log'] else 'starting...', flush=True)
    time.sleep(60)
trc.status(JOB, tail=3)
FINAL_DIR = Path(JOB['result']['run_dir'])          # contains translation_final.json -> hand-off to TTS

# %% 20.3 — independent blind check against a previous system (LLM judge from another model family)
from dubbing_translation.generators import LLMClient   # noqa: E402
from dubbing_translation.io import load_tasks           # noqa: E402
from dubbing_translation.judge import pairwise          # noqa: E402

tasks = load_tasks(EXP_DIR / 'tasks.json')
new = {u['unit_id']: u['natural_ar'] for u in json.loads((FINAL_DIR / 'translation_final.json').read_text())['units']}
old = {p.stem: json.loads(p.read_text())['natural_ar'] for p in (EXP_OUT / 'S7_best_of_12_habibi_gate').glob('*.json')}
judge = LLMClient('openrouter', 'nvidia/nemotron-3-ultra-550b-a55b:free', min_interval_s=3.5, timeout_s=240)
# OpenRouter free tier: ~50 requests/day. The jsonl cache makes a rerun continue where the quota stopped.
print(pairwise(tasks, new, old, judge, FINAL_DIR / 'judge_vs_S7_nemotron_ultra.jsonl'))

# %% 20.4 — compare all runs recorded in the registry
print(subprocess.run([sys.executable, '-m', 'dubbing_translation', 'runs',
                      '--work-dir', str(RUN_DIR / 'translation_step')],              # noqa: F821
                     cwd=str(TR_REPO / 'translation'), capture_output=True, text=True).stdout)
