# Reference Papers

Key literature for the MAEHD (Multi-Agent Experimental Hit Discovery) project.

## Active Learning for Drug Discovery / Hit Discovery

1. **GeneDisco** — Mehrjou et al. (2021). "GeneDisco: A Benchmark for Experimental
   Design in Drug Discovery." ICML 2022.
   - The benchmark dataset we use (CRISPR screens: IFNγ, IL-2, NK, Tau, SARS-CoV-2)
   - Compares acquisition functions for sequential batch active learning
   - https://arxiv.org/abs/2110.11875

2. **Batch Active Learning** — Kirsch et al. (2019). "BatchBALD: Efficient and Diverse
   Batch Acquisition for Deep Bayesian Active Learning." NeurIPS 2019.
   - Batch acquisition with diversity using mutual information
   - https://arxiv.org/abs/1906.08158

3. **Bayesian Active Learning by Disagreement (BALD)** — Houlsby et al. (2011).
   "Bayesian Active Learning for Classification and Preference Learning."
   - Foundational uncertainty-based acquisition
   - https://arxiv.org/abs/1112.5745

## Multi-Target / Multi-Output Bayesian Optimization

4. **Multi-Output Gaussian Processes** — Alvarez et al. (2012). "Kernels for
   Vector-Valued Functions: A Review." Foundations and Trends in ML.
   - Theory of correlated multi-output GPs (LMC, ICM models)

5. **Multi-Objective Bayesian Optimization** — Daulton et al. (2020).
   "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective
   Bayesian Optimization." NeurIPS 2020.
   - Pareto-optimal batch acquisition for multiple objectives
   - https://arxiv.org/abs/2006.05078

## LLM Agents for Scientific Discovery

6. **ChemCrow** — Bran et al. (2023). "ChemCrow: Augmenting large-language models
   with chemistry tools." Nature Machine Intelligence.
   - LLM agent with domain-specific chemistry tools
   - https://arxiv.org/abs/2304.05376

7. **Coscientist** — Boiko et al. (2023). "Autonomous chemical research with
   large language models." Nature.
   - LLM orchestrating experiments, planning synthesis, running reactions
   - https://doi.org/10.1038/s41586-023-06792-0

8. **Scientific Discovery in the Age of AI** — Wang et al. (2023).
   "Scientific discovery in the age of artificial intelligence." Nature.
   - Review of AI-driven scientific discovery paradigms

## Foundation Models for Genes / Biology

9. **scGPT** — Cui et al. (2024). "scGPT: Toward building a foundation model for
   single-cell multi-omics using generative AI." Nature Methods.
   - Transformer-based foundation model for single-cell data
   - Gene embeddings from pre-training on 33M cells
   - https://doi.org/10.1038/s41592-024-02201-0

10. **Geneformer** — Theodoris et al. (2023). "Transfer learning enables predictions
    in network biology." Nature.
    - Transformer for gene regulatory networks
    - Gene embeddings capture functional relationships
    - https://doi.org/10.1038/s41586-023-06139-9

11. **Gene2Vec** — Du et al. (2019). "Gene2vec: distributed representation of genes
    based on co-expression." BMC Genomics.
    - Word2Vec-style gene embeddings from co-expression
    - https://doi.org/10.1186/s12864-018-5370-x

12. **GenePT** — Chen & Zou (2024). "GenePT: A Simple But Effective Foundation Model
    for Genes Using ChatGPT." bioRxiv.
    - Gene embeddings from GPT summaries of NCBI gene descriptions
    - Shows that LLM-derived embeddings capture functional similarity
    - https://doi.org/10.1101/2023.10.16.562533

## CRISPR Screen Biology (Our Datasets)

13. **Schmidt et al. (2021)** — "CRISPR activation and interference screens decode
    stimulation responses in primary human T cells." Science.
    - Source of IFNγ and IL-2 screening datasets
    - Genome-wide CRISPRa/CRISPRi in primary CD4+ T cells

14. **Zhuang et al. (2019)** — "Genome-wide CRISPR screen reveals cancer cell
    resistance mechanisms to NKcell-mediated cytotoxicity."
    - Source of NK cell cytotoxicity dataset

15. **Sanchez et al. (2021)** — iPSC-derived neuron CRISPR screen for Tau.
    - Source of Tau aggregation dataset

16. **Zhu et al. (2021)** — CRISPR screen for SARS-CoV-2 host factors.
    - Source of viral entry/replication dataset

## Gene Features We Use

17. **DepMap / Project Achilles** — Tsherniak et al. (2017). "Defining a Cancer
    Dependency Map." Cell.
    - Gene essentiality scores across 1000+ cell lines
    - Our "achilles" feature set: 808-dim gene dependency profile

18. **STRING Database** — Szklarczyk et al. (2023). "The STRING database in 2023."
    Nucleic Acids Research.
    - Protein-protein interaction network
    - Our "string" feature set: 800-dim network embeddings

19. **CCLE** — Ghandi et al. (2019). "Next-generation characterization of the
    Cancer Cell Line Encyclopedia." Nature.
    - Protein quantification across cancer cell lines
    - Our "ccle" feature set: 420-dim protein levels

## Probability of Hit / Acquisition Functions

20. **Thompson Sampling** — Thompson (1933). "On the likelihood that one unknown
    probability exceeds another in view of the evidence of two samples."
    - Classic exploration-exploitation strategy
    - We use for single-target acquisition

21. **GP-UCB** — Srinivas et al. (2010). "Gaussian Process Optimization in the
    Bandit Setting: No Regret and Experimental Design." ICML.
    - Upper confidence bound for GP-based optimization
    - https://arxiv.org/abs/0912.3995
