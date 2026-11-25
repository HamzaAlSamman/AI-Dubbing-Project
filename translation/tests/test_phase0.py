"""Phase 0 regression tests (audit P0-1 .. P0-5 and the promoted data-integrity items).
Offline: no model, no network. Run from the repo root:
    python -m unittest discover -s translation/tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TR = Path(__file__).resolve().parents[1]


def _notebook_code(name):
    nb = json.loads((TR / 'notebooks' / name).read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


# ---------------------------------------------------------------- P0-1 Colab environment
class ColabEnvironmentTests(unittest.TestCase):
    def test_requirements_file_pins_every_generator_dependency(self):
        req = (TR / 'requirements-colab.txt').read_text(encoding='utf-8')
        names = {line.split('==')[0].split('>=')[0].split('<')[0].strip().lower()
                 for line in req.splitlines() if line.strip() and not line.startswith('#')}
        for pkg in ('transformers', 'accelerate', 'bitsandbytes', 'google-genai', 'sentencepiece', 'sacrebleu'):
            self.assertIn(pkg, names)
        for line in req.splitlines():
            if line.strip() and not line.startswith('#'):
                self.assertTrue(any(op in line for op in ('==', '>=')), f'unpinned: {line}')

    def test_v2_notebook_installs_requirements_and_runs_preflight_before_models(self):
        cells = _notebook_code('02_translation_v2_colab.ipynb')
        setup = next(c for c in cells if 'drive.mount' in c)
        self.assertIn('requirements-colab.txt', setup)
        pre = [i for i, c in enumerate(cells) if 'preflight' in c]
        run = [i for i, c in enumerate(cells) if 'trc.start(' in c]
        self.assertTrue(pre and run and pre[0] < run[0])

    def test_preflight_reports_missing_package_without_loading_models(self):
        from dubbing_translation.preflight import preflight
        rep = preflight(need={'gpu': False, 'packages': ['surely_not_installed_pkg_xyz'], 'env': [], 'binaries': []},
                        raise_on_error=False)
        self.assertFalse(rep['ok'])
        self.assertIn('surely_not_installed_pkg_xyz', ' '.join(rep['errors']))

    def test_preflight_names_needed_packages_from_the_generator_plan(self):
        from dubbing_translation.preflight import requirements_for
        need = requirements_for(use_hymt=True, hf_models=[('tencent/Hy-MT2-7B', 'hy_mt2_terms', 1)], use_llm=True)
        self.assertIn('bitsandbytes', need['packages'])
        self.assertIn('google.genai', need['packages'])
        self.assertIn('GEMINI_API_KEY', need['env'])
        self.assertTrue(need['gpu'])


if __name__ == '__main__':
    unittest.main()
