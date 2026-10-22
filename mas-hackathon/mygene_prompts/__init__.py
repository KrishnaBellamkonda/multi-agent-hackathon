"""Public re-exports for ``mygene_prompts``."""
from mygene_prompts.client import (
    DEFAULT_FIELDS,
    MG_BASE,
    MyGeneClient,
    batch_fetch,
    fetch_gene,
)
from mygene_prompts.cache import (
    GeneCache,
    build_full_cache,
    default_cache_path,
)
from mygene_prompts.normalize import (
    EVIDENCE_TIERS,
    aliases,
    as_list,
    display_name,
    go_terms,
    interpro_domains,
    pathways,
    summary_text,
)
from mygene_prompts.prompts import (
    PHENOTYPE_KEYWORDS,
    PHENOTYPE_SHORT,
    context_compact,
    context_targeted,
    feature_set_brief,
    lookup_gene_tool_schema,
    phenotype_summary,
)
from mygene_prompts.integration import (
    enriched_feature_recommendation_hint,
    enriched_gene_block,
    enriched_phenotype_context,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # client
    "DEFAULT_FIELDS",
    "MG_BASE",
    "MyGeneClient",
    "batch_fetch",
    "fetch_gene",
    # cache
    "GeneCache",
    "build_full_cache",
    "default_cache_path",
    # normalize
    "EVIDENCE_TIERS",
    "aliases",
    "as_list",
    "display_name",
    "go_terms",
    "interpro_domains",
    "pathways",
    "summary_text",
    # prompts
    "PHENOTYPE_KEYWORDS",
    "PHENOTYPE_SHORT",
    "context_compact",
    "context_targeted",
    "feature_set_brief",
    "lookup_gene_tool_schema",
    "phenotype_summary",
    # integration
    "enriched_feature_recommendation_hint",
    "enriched_gene_block",
    "enriched_phenotype_context",
]
