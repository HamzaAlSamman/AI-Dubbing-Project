"""Duration-aware EN->AR translation step for the dubbing pipeline.

Recipe (see translation/README.md for the evidence behind each choice):
  1. several candidates per unit from a small, complementary set of generators
     (a dedicated MT model + an LLM prompted with an explicit Arabic-letter budget),
  2. quality and duration gate (reference-free QE, spelling, mechanical QA, validity,
     predicted spoken duration against the unit's real time window),
  3. guarded refine loop for units with no accepted candidate,
  4. lowest-QE accepted candidate wins; everything is written once, resumable,
     and recorded in a run manifest + run registry.

Importing this package never loads a model or calls a network API.
"""

__version__ = '1.0.0'
