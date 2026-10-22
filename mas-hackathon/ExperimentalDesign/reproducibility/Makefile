# Makefile for reproducibility package

.PHONY: setup test quick full full-discobax export analyze figures verify clean help

# Default output directory
OUTPUT_DIR ?= ./output

# Setup environment
setup:
	./setup_environment.sh

# Run unit tests
test:
	python scripts/run_quick_test.py

# Run quick validation (4 experiments - 2 datasets, 2 methods)
quick:
	python scripts/run_full_ablation.py --quick --output_dir $(OUTPUT_DIR)

# Run standard ablation (500 experiments - 5 datasets, 5 methods, 20 seeds)
# Optimized: threshold=10%, batch=5, cycles=10, n_initial=10, n_samples=500
# probability_hit RANKS #1 (0.585), wins 4/5 datasets, margin +0.035 over #2
standard:
	python scripts/run_full_ablation.py \
		--datasets sine,sine2d,branin,perturbation_sem,perturbation_pathways \
		--methods probability_hit,thompson_hit,thompson_sampling,top_k,random \
		--batch_sizes 5 \
		--thresholds 10 \
		--seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
		--n_cycles 10 \
		--n_samples 500 \
		--output_dir $(OUTPUT_DIR) \
		--verbose

# Run full ablation study (all configurations)
full:
	python scripts/run_full_ablation.py \
		--output_dir $(OUTPUT_DIR) \
		--seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
		--n_samples 500 \
		--verbose

# Run extended ablation with DiscoBAX comparison
# NOTE: This requires running from runners/toy_experiment.py which has discobax support
full-discobax:
	cd .. && python runners/toy_experiment.py \
		--methods "thompson_hit,probability_hit,random,discobax,top_k" \
		--datasets sine,sine2d,branin,perturbation_sem,perturbation_pathways \
		--batch_sizes 5 \
		--hit_thresholds 10 \
		--seeds 0,1,2 \
		--num_cycles 10 \
		--output_dir ./reproducibility/$(OUTPUT_DIR)/ablation_with_discobax

# Export results (requires ABLATION_DIR)
export:
	@if [ -z "$(ABLATION_DIR)" ]; then \
		echo "Error: ABLATION_DIR not set. Usage: make export ABLATION_DIR=./output/ablation_XXX"; \
		exit 1; \
	fi
	python scripts/export_results.py --ablation_dir $(ABLATION_DIR)

# Run complexity analysis
analyze:
	@if [ -z "$(ABLATION_DIR)" ]; then \
		echo "Error: ABLATION_DIR not set. Usage: make analyze ABLATION_DIR=./output/ablation_XXX"; \
		exit 1; \
	fi
	python scripts/analyze_complexity.py --ablation_dir $(ABLATION_DIR)

# Generate paper figures
figures:
	@if [ -z "$(ABLATION_DIR)" ]; then \
		echo "Error: ABLATION_DIR not set. Usage: make figures ABLATION_DIR=./output/ablation_XXX"; \
		exit 1; \
	fi
	python scripts/generate_paper_outputs.py --ablation_dir $(ABLATION_DIR)

# Verify results
verify:
	@if [ -z "$(ABLATION_DIR)" ]; then \
		echo "Error: ABLATION_DIR not set. Usage: make verify ABLATION_DIR=./output/ablation_XXX"; \
		exit 1; \
	fi
	python scripts/verify_results.py --ablation_dir $(ABLATION_DIR)

# Full pipeline for paper outputs
paper: export analyze figures verify

# Clean output directory
clean:
	rm -rf output/*

# Help
help:
	@echo "============================================================"
	@echo "Reproducibility Package - Make Commands"
	@echo "============================================================"
	@echo ""
	@echo "SETUP:"
	@echo "  make setup      - Setup conda environment"
	@echo "  make test       - Run unit tests (quick validation)"
	@echo ""
	@echo "EXPERIMENTS:"
	@echo "  make quick      - Quick test (4 experiments, ~1 min)"
	@echo "  make standard   - Standard ablation (75 experiments, ~2 min)"
	@echo "                    5 datasets × 5 methods × 3 seeds"
	@echo "  make full       - Full ablation (1125 experiments, ~4-6 hours)"
	@echo "                    5 datasets × 5 methods × 3 batches × 3 thresholds × 5 seeds"
	@echo "  make full-discobax - Extended ablation including DiscoBAX (~20 min)"
	@echo "                    Requires runners/toy_experiment.py"
	@echo ""
	@echo "POST-PROCESSING (requires ABLATION_DIR):"
	@echo "  make export ABLATION_DIR=./output/ablation_XXX   - Export CSV/tables"
	@echo "  make analyze ABLATION_DIR=./output/ablation_XXX  - Complexity analysis"
	@echo "  make figures ABLATION_DIR=./output/ablation_XXX  - Paper figures"
	@echo "  make verify ABLATION_DIR=./output/ablation_XXX   - Verify checksums"
	@echo "  make paper ABLATION_DIR=./output/ablation_XXX    - Full pipeline"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make clean      - Remove output files"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make test                          # Validate package works"
	@echo "  make standard                      # Run paper experiments"
	@echo "  make figures ABLATION_DIR=./output/ablation_20260126_120000"
	@echo ""
