"""
MAEHD — Multi-Agent Experimental Hit Discovery

Multi-target active hit discovery with mathematically interpretable
acquisition policies and cross-target resource sharing.

Key idea: querying a single gene reveals outcomes for ALL targets
simultaneously. A multi-agent system allocates shared experimental
budget across K targets more efficiently than K independent searches.

Agents are fully mathematical (no LLM dependency):
  • CorrelationAgent: estimates cross-target structure, adjusts weights
  • AcquisitionAgent: selects genes via principled acquisition functions
  • DesignerAgent: allocates budget across targets and policies

All policies have closed-form expressions or well-understood
Bayesian decision-theoretic justifications.
"""

__version__ = "0.1.0"
