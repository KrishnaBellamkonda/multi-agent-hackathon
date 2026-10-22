# `mygene_prompts` — MyGene.info prompt-injection module

## 1. What this is

`mygene_prompts` is a small, standalone Python module that turns
[MyGene.info](https://mygene.info/) records into **LLM prompt
fragments** usable inside the `mas-hackathon` active hit-discovery
pipeline.  It fetches per-gene summaries, GO annotations, pathway
memberships, and InterPro domains; filters the noise (LOW-evidence
IEA terms by default); and emits short, well-formed blurbs intended
to drop straight into an existing system prompt or batch-scoring
template.

The signal it adds is **non-redundant**.  Empirically, the cosine
similarity between two genes in the Achilles feature space and the
Jaccard overlap of their GO Biological Process terms correlate at
Pearson r ≈ −0.02 across random pairs from the GeneDisco gene list,
and only ~4 % of random pairs share *any* GO term.  So adding
MyGene-derived context to an LLM's reasoning surface is not a
re-encoding of the existing numerical features — it brings curated
biology that the surrogate model cannot see.

## 2. Quick start

```python
from mygene_prompts import batch_fetch, context_targeted

recs = batch_fetch(["STAT1", "JAK1", "IRGM"])
print(context_targeted(recs["STAT1"], phenotype="schmidt_2021_ifng"))
```

Output (truncated):

```
**STAT1 (signal transducer and activator of transcription 1)**
Summary: The protein encoded by this gene is a member of the STAT
protein family...
Relevant GO BP for schmidt_2021_ifng: cellular response to
interferon-gamma [IDA]; JAK-STAT cascade [IBA]; ...
Relevant pathways: Interferon Signaling (reactome); ...
```

## 3. Building and using the offline cache

The hackathon gene universe is ~17,500 symbols.  Live MyGene fetches
each round are slow and rude; build a one-shot disk cache instead.

```bash
cd mas-hackathon
PYENV_VERSION=weather-climate python -m mygene_prompts build-cache --all
```

* Reads symbols from `ExperimentalDesign/data/achilles.h5` rownames.
* Writes `ExperimentalDesign/data/mygene_cache.pkl.gz`
  (~8–12 MB gzipped, ~30–50 MB raw).
* Expected runtime: ~5–10 minutes (18 chunks × 1000 symbols × ~0.5 s).
* Idempotent: re-running only fetches symbols not already in the cache.

Then in code:

```python
from mygene_prompts import GeneCache
cache = GeneCache.load()  # uses default path
```

Pass `cache` to any of the integration adapters in §6.

The cache path lives under `ExperimentalDesign/data/`, which is
already gitignored — the cache file does **not** end up in the
repo.

## 4. The three prompt patterns

| Pattern | When to use | Approx. token cost |
|---|---|---|
| `context_compact(rec)` | Showing the LLM a batch of genes without telling it which screen we are scoring for. | ~80 / gene |
| `context_targeted(rec, phenotype)` | Per-screen scoring: the caller already knows which phenotype is active. Filters GO/pathway list to entries that match phenotype keywords. | ~50 / gene |
| `lookup_gene_tool_schema()` | Function-calling LLMs.  Returns a JSON-schema describing a `lookup_gene(symbol, phenotype=?)` tool the model can call on its own. | n/a (definition only) |

Plus one phenotype-level builder:

* `phenotype_summary(phenotype, exemplar_genes)` — a 2-3 sentence
  paragraph naming the screen, listing the pathways that recur across
  the supplied exemplar regulators, and listing the regulators
  themselves.  Designed to be appended to the existing
  `BiologyKnowledgeAgent.get_phenotype_context` output.

## 5. General-purpose use case

You have a list of human gene symbols and want a short, well-grounded
text fragment describing each — optionally biased toward a phenotype.

```python
from mygene_prompts import batch_fetch, context_compact, context_targeted

recs = batch_fetch(["TP53", "MYC", "RB1"])
for sym, rec in recs.items():
    print(context_compact(rec))
    print()
```

No `mas-hackathon` dependency is required for this surface; only
`requests` is needed.

## 6. Specific use case: wiring into `mas-hackathon`

All three adapters live in `mygene_prompts.integration` and accept an
optional `cache: GeneCache`.  None of them modify the existing
`maehd/` code — they wrap or extend it from the outside.

### 6.1 Per-screen context

`maehd.agents.biology_agent.BiologyKnowledgeAgent.get_phenotype_context`
already returns a system-prompt paragraph for a list of screens.
Wrap it:

```python
from mygene_prompts import GeneCache, enriched_phenotype_context

cache = GeneCache.load()
base = bio_agent.get_phenotype_context(screen_names)
augmented = enriched_phenotype_context(
    screen_names,
    base,
    cache=cache,
    exemplar_genes_per_screen={
        "schmidt_2021_ifng": ["STAT1", "JAK1", "IFNG", "IRF1", "IRGM"],
    },
)
# Pass `augmented` into ExperimentAgent in place of the bare base context:
agent = ExperimentAgent(biology_context=augmented, ...)
```

### 6.2 Per-gene context

`BiologyKnowledgeAgent.batch_score_genes` formats a list of genes into
`_BATCH_SCORE_PROMPT`.  Insert per-gene MyGene context before the gene
list:

```python
from mygene_prompts import enriched_gene_block

gene_block = enriched_gene_block(
    gene_names=["STAT1", "JAK1", "IRGM"],
    phenotype="schmidt_2021_ifng",
    cache=cache,
)
# Splice `gene_block` into the prompt before sending to the LLM.
```

### 6.3 Feature-set selection hint

`BiologyKnowledgeAgent.recommend_features` asks the LLM to pick a
feature set per screen.  Prepend a hint paragraph that describes each
feature set in biological terms and the recurring pathway pattern of
each screen's regulators:

```python
from mygene_prompts import enriched_feature_recommendation_hint

hint = enriched_feature_recommendation_hint(
    screen_names=["schmidt_2021_ifng", "zhuang_2019_nk"],
    feature_sets=["achilles", "string", "ccle"],
    cache=cache,
    exemplar_genes_per_screen={
        "schmidt_2021_ifng": ["STAT1", "JAK1", "IFNG"],
        "zhuang_2019_nk":    ["PRF1", "GZMB", "KLRK1"],
    },
)
# Prepend `hint` to the prompt inside recommend_features.
```

### 6.4 Tool-use pattern (function-calling LLMs)

If you migrate `ExperimentAgent` to a function-calling provider,
expose `lookup_gene` as a tool:

```python
from mygene_prompts import lookup_gene_tool_schema, batch_fetch, context_targeted

tools = [lookup_gene_tool_schema()]

def lookup_gene(symbol: str, phenotype: str | None = None) -> str:
    rec = batch_fetch([symbol]).get(symbol)
    if phenotype:
        return context_targeted(rec, phenotype)
    from mygene_prompts import context_compact
    return context_compact(rec)
```

## 7. Adapting to future aims

The hackathon may consolidate around a different agent stack
(`ai/OrchestratorAgent`, a different LLM provider, a different
benchmark, …).  The adapters in `integration.py` are *examples* —
they assume a specific maehd surface.  The **stable contract** of
this module is the lower layer in `prompts.py`:

* `context_compact(rec)`
* `context_targeted(rec, phenotype)`
* `phenotype_summary(phenotype, exemplar_genes)`
* `feature_set_brief(feature_set_name)`
* `lookup_gene_tool_schema()`

These primitives have no `maehd` dependency.  An LLM (or a new
developer) can recompose them into whatever agent harness comes next:
feed `recs = batch_fetch(symbols)`, then route the records through
whichever builder fits the current call site.

## 8. Evidence-tier caveats

MyGene's GO annotations are bimodal in quality:

* ~30 % HIGH-tier (IDA / IMP / IGI / IPI / EXP — direct experiment)
* ~40 % MED-tier (ISS / IBA / TAS — homology, phylogeny, curator-reviewed)
* ~30 % LOW-tier (IEA — machine-generated, never reviewed)

`go_terms` defaults to `{HIGH, MED}` — LOW (IEA) annotations are
filtered out, because including them often surfaces propagated noise
("DNA binding" on every nuclear protein, etc.).  Pass
`evidence_tiers={"HIGH", "MED", "LOW"}` to include them.

Annotation bias also matters: well-studied genes (TP53, STAT1) have
hundreds of high-evidence GO terms, while many genes (especially
uncharacterised ORFs) have none.  `context_compact` will say
*"No high-evidence annotations available for this gene"* rather than
silently emit an empty block — pay attention to that string when
debugging downstream prompts.

## 9. Rate limits & politeness

* MyGene allows up to 1000 IDs per POST request.  Keep `chunk=1000`.
* Sleep 0.2–0.5 s between chunks (`pause=0.5` is the default in
  `build_full_cache`).
* Do **not** issue parallel requests — single-thread is enough to
  saturate any reasonable polite quota.
* For repeated calls in the pipeline, always pre-build the disk cache.
  Live fetches during a run will both slow it down and risk a 429.

## 10. License & data sources

This module is a thin client over public bioinformatic resources.

* [MyGene.info](https://mygene.info/) — aggregates the sources below.
* [Gene Ontology](http://geneontology.org/) — GO BP / MF / CC terms and evidence codes.
* [Reactome](https://reactome.org/), [WikiPathways](https://www.wikipathways.org/), [KEGG](https://www.kegg.jp/) — pathway memberships.
* [InterPro](https://www.ebi.ac.uk/interpro/) — protein domains.

The on-disk cache `ExperimentalDesign/data/mygene_cache.pkl.gz`
contains third-party data; it is **not** redistributed in this repo
(the cache lives under a gitignored directory).  Re-fetch from
MyGene.info if you need it on a fresh checkout.
