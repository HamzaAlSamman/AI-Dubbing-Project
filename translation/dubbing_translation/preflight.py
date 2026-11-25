"""Preflight: check the environment BEFORE any model is downloaded or loaded.

A missing package, a missing key, no GPU or a read-only Drive should fail in seconds with a clear
message, not after 20 minutes of model download. The report (with installed versions) goes into the
run manifest, so every result says which environment produced it.

Passing the preflight does not prove the run will succeed (out-of-memory, model download errors,
API quota): it removes the failures that can be known in advance.
"""
import importlib
import importlib.metadata as md
import os
import platform
import shutil
import sys

# import name -> distribution name (for the version)
_DIST = {'google.genai': 'google-genai', 'bitsandbytes': 'bitsandbytes', 'transformers': 'transformers',
         'accelerate': 'accelerate', 'sentencepiece': 'sentencepiece', 'sacrebleu': 'sacrebleu', 'torch': 'torch'}


class PreflightError(RuntimeError):
    pass


def requirements_for(*, use_hymt=False, hf_models=(), use_llm=False, refine=False, rank=False, qe=True,
                     spell=True, language=False):
    """What the planned run needs. `language`: a V2 config (glossary agent uses the LLM)."""
    packages, env, binaries = ['transformers', 'sentencepiece'], [], []
    gpu = False
    if use_hymt or hf_models:
        packages += ['torch', 'accelerate', 'bitsandbytes']
        gpu = True
    if qe:
        packages += ['torch']
        gpu = True
    if use_llm or refine or rank or language:
        packages += ['google.genai']
        env += ['GEMINI_API_KEY']
    if spell:
        binaries += ['hunspell']
    return {'packages': sorted(set(packages)), 'env': sorted(set(env)), 'binaries': binaries, 'gpu': gpu}


def _version(name):
    try:
        return md.version(_DIST.get(name, name))
    except md.PackageNotFoundError:
        return None


def preflight(need, *, write_dir=None, raise_on_error=True):
    """Check packages (importable), environment variables (present, value never read into the report),
    binaries, GPU, and write access. Returns a report dict; raises PreflightError when not ok."""
    errors, versions = [], {}
    for pkg in need.get('packages', []):
        try:
            importlib.import_module(pkg)
            versions[pkg] = _version(pkg)
        except Exception as exc:                      # noqa: BLE001 - any import failure is a preflight error
            errors.append(f'package {pkg!r} cannot be imported: {type(exc).__name__}: {str(exc)[:120]}')
    for var in need.get('env', []):
        if not os.environ.get(var):
            errors.append(f'environment variable {var} is not set (load it from the secrets cell)')
    for b in need.get('binaries', []):
        if shutil.which(b) is None:
            errors.append(f'binary {b!r} not found on PATH')
    gpu = None
    if need.get('gpu'):
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                gpu = {'name': p.name, 'memory_gb': round(p.total_memory / 2 ** 30, 1)}
            else:
                errors.append('no CUDA GPU: Runtime -> Change runtime type -> T4 GPU')
        except Exception as exc:                      # noqa: BLE001
            errors.append(f'GPU check failed: {type(exc).__name__}: {str(exc)[:120]}')
    if write_dir is not None:
        probe = os.path.join(str(write_dir), '.preflight_write_test')
        try:
            os.makedirs(str(write_dir), exist_ok=True)
            with open(probe, 'w', encoding='utf-8') as f:
                f.write('ok')
            os.remove(probe)
        except OSError as exc:
            errors.append(f'cannot write to {write_dir}: {exc}')
    report = {'ok': not errors, 'errors': errors, 'versions': versions, 'gpu': gpu,
              'python': sys.version.split()[0], 'platform': platform.platform(), 'need': need}
    if errors and raise_on_error:
        raise PreflightError('preflight failed:\n- ' + '\n- '.join(errors))
    return report
