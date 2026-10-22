"""Our custom acquisition algorithms package.

Exports:
	top_k              – diversified Top-K batch selection
	thompson_hit       – Thompson Sampling for Hit Number
	greedy_kg_sum      – greedy knowledge-gradient style cumulative-sum lookahead

Note: Removed stale import of a non-existent 'edug' module which previously
caused ModuleNotFoundError during dynamic loading.
"""

from .tok_k.top_k import TopKAcquisition as top_k
from .thompson_hit.thompson_hit import ThompsonHitAcquisition as thompson_hit
from .greedy_kg_sum.greedy_kg_sum import GreedyKGSumAcquisition as greedy_kg_sum
from .q_thompson_fantasy.q_thompson_fantasy import QThompsonFantasyBatchAcquisition as q_thompson_fantasy
from .tq_thompson_sampling.tq_thompson_sampling import TQThompsonSamplingBatchAcquisition as tq_thompson_sampling

__all__ = ["top_k", "thompson_hit", "greedy_kg_sum", "q_thompson_fantasy", "tq_thompson_sampling"]