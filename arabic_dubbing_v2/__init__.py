"""Course-agnostic Arabic educational-dubbing translation workflow.

All modules default to offline, read-only source processing. Nothing calls ASR,
translation or TTS without an explicit caller-provided execution flag.
"""
from .workflow import prepare_project, enrich_master
from .boundaries import build_units_from_whisperx
__version__ = '2.0.0-dev'
__all__ = ['prepare_project', 'enrich_master', 'build_units_from_whisperx']
