"""Cambrian engine.

Deterministic, domain-agnostic diversity engine that owns the anti-convergence
math for the `ideate` skill: embeddings, MAP-Elites archive, geometric novelty,
DPP diverse selection, an anti-collapse monitor, and local preference memory.

The LLM parts (variation operators, the skeptical judge prefilter) are performed
by the agent, not here. This package never judges novelty: geometry owns
diversity, the judge only filters validity/on-brief upstream.
"""

__version__ = "0.6.4"
