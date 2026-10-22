#!/usr/bin/env python3
"""
Unit tests for core reproducibility package functionality.

Usage:
    pytest tests/test_core.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reproducibility.shared.datasets import (
    Dataset, generate_dataset, list_datasets
)
from reproducibility.shared.acquisition import (
    get_acquisition_function,
    list_acquisition_functions,
    RandomAcquisition,
    ProbabilityHitAcquisition,
    ThompsonHitAcquisition,
    ThompsonSamplingAcquisition,
    TopKAcquisition,
)
from reproducibility.shared.metrics import compute_metrics
from reproducibility.shared.experiment import run_experiment, ExperimentConfig


class TestDatasets:
    """Tests for dataset generation."""

    def test_list_datasets(self):
        names = list_datasets()
        assert isinstance(names, list)
        assert len(names) >= 5
        assert 'sine' in names
        assert 'branin' in names

    def test_generate_sine(self):
        ds = generate_dataset('sine', n_samples=100, seed=0)
        assert isinstance(ds, Dataset)
        assert ds.X.shape == (100, 1)
        assert ds.y.shape == (100,)

    def test_generate_branin(self):
        ds = generate_dataset('branin', n_samples=100, seed=0)
        assert ds.X.shape == (100, 2)

    def test_all_datasets_generate(self):
        for name in list_datasets():
            ds = generate_dataset(name, n_samples=50, seed=42)
            assert ds.X.shape[0] == 50
            assert ds.y.shape[0] == 50

    def test_threshold_and_hits(self):
        ds = generate_dataset('sine', n_samples=100, seed=0)
        threshold = ds.get_threshold_value(10)
        hits = ds.get_hits(10)
        assert isinstance(threshold, float)
        assert len(hits) > 0
        assert len(hits) < 100
        assert np.all(ds.y[hits] >= threshold)

    def test_reproducibility(self):
        ds1 = generate_dataset('sine', n_samples=50, seed=7)
        ds2 = generate_dataset('sine', n_samples=50, seed=7)
        np.testing.assert_array_equal(ds1.X, ds2.X)
        np.testing.assert_array_equal(ds1.y, ds2.y)


class TestAcquisition:
    """Tests for acquisition functions."""

    def setup_method(self):
        np.random.seed(42)
        self.n_pool = 50
        self.batch_size = 5
        self.threshold = 1.0
        self.X = np.random.randn(self.n_pool, 2)
        self.gp_mean = np.random.randn(self.n_pool)
        self.gp_std = np.abs(np.random.randn(self.n_pool)) + 0.1

    def _check_selected(self, selected):
        assert len(selected) == self.batch_size
        assert len(set(selected.tolist())) == self.batch_size
        assert all(0 <= s < self.n_pool for s in selected)

    def test_random(self):
        acq = RandomAcquisition()
        selected = acq.acquire(self.X, self.gp_mean, self.gp_std,
                               batch_size=self.batch_size, threshold=self.threshold)
        self._check_selected(selected)

    def test_probability_hit(self):
        acq = ProbabilityHitAcquisition()
        selected = acq.acquire(self.X, self.gp_mean, self.gp_std,
                               batch_size=self.batch_size, threshold=self.threshold)
        self._check_selected(selected)

    def test_thompson_hit(self):
        acq = ThompsonHitAcquisition()
        selected = acq.acquire(self.X, self.gp_mean, self.gp_std,
                               batch_size=self.batch_size, threshold=self.threshold)
        self._check_selected(selected)

    def test_thompson_sampling(self):
        acq = ThompsonSamplingAcquisition()
        selected = acq.acquire(self.X, self.gp_mean, self.gp_std,
                               batch_size=self.batch_size, threshold=self.threshold)
        self._check_selected(selected)

    def test_top_k(self):
        acq = TopKAcquisition()
        selected = acq.acquire(self.X, self.gp_mean, self.gp_std,
                               batch_size=self.batch_size, threshold=self.threshold)
        self._check_selected(selected)
        # TopK should select the indices with the highest predicted mean
        expected = set(np.argsort(self.gp_mean)[-self.batch_size:].tolist())
        assert set(selected.tolist()) == expected

    def test_get_acquisition_function(self):
        for name in list_acquisition_functions():
            acq = get_acquisition_function(name)
            assert isinstance(acq, (
                RandomAcquisition, ProbabilityHitAcquisition,
                ThompsonHitAcquisition, ThompsonSamplingAcquisition,
                TopKAcquisition,
            ))

    def test_batch_size_equals_pool(self):
        acq = RandomAcquisition()
        selected = acq.acquire(self.X, self.gp_mean, self.gp_std,
                               batch_size=self.n_pool, threshold=self.threshold)
        assert len(selected) == self.n_pool


class TestMetrics:
    """Tests for metric computation."""

    def setup_method(self):
        np.random.seed(0)
        self.n = 80
        self.X = np.random.randn(self.n, 2)
        self.y = np.random.randn(self.n)
        self.true_hit_indices = np.where(self.y > np.percentile(self.y, 90))[0]

    def test_compute_metrics_returns_dict(self):
        acquired = np.arange(20)
        unseen = np.arange(20, self.n)
        y_pred = np.random.randn(self.n)
        metrics = compute_metrics(self.X, self.y, acquired, self.true_hit_indices, y_pred, unseen)
        assert isinstance(metrics, dict)
        assert 'RecoveredHitsRatio_Cumulative' in metrics

    def test_perfect_hit_recovery(self):
        acquired = self.true_hit_indices.copy()
        unseen = np.setdiff1d(np.arange(self.n), acquired)
        y_pred = self.y.copy()
        metrics = compute_metrics(self.X, self.y, acquired, self.true_hit_indices, y_pred, unseen)
        assert metrics['RecoveredHitsRatio_Cumulative'] == pytest.approx(1.0)

    def test_zero_hit_recovery(self):
        non_hits = np.setdiff1d(np.arange(self.n), self.true_hit_indices)
        acquired = non_hits[:5]
        unseen = np.setdiff1d(np.arange(self.n), acquired)
        y_pred = np.random.randn(self.n)
        metrics = compute_metrics(self.X, self.y, acquired, self.true_hit_indices, y_pred, unseen)
        assert metrics['RecoveredHitsRatio_Cumulative'] == pytest.approx(0.0)

    def test_metrics_with_no_predictions(self):
        acquired = np.arange(20)
        unseen = np.arange(20, self.n)
        metrics = compute_metrics(self.X, self.y, acquired, self.true_hit_indices,
                                  y_pred=None, unseen_indices=unseen)
        assert isinstance(metrics, dict)


class TestExperiment:
    """Tests for the experiment runner."""

    def test_run_experiment_succeeds(self):
        config = ExperimentConfig(
            dataset_name='sine',
            method_name='random',
            batch_size=5,
            threshold_pct=10,
            seed=0,
            n_cycles=3,
            n_samples=60,
            n_initial=10,
        )
        results = run_experiment(config, verbose=False)
        assert results.success
        assert len(results.cycles) == 3
        assert results.final_metrics

    def test_run_experiment_multiple_methods(self):
        for method in ['random', 'probability_hit', 'thompson_hit']:
            config = ExperimentConfig(
                dataset_name='sine',
                method_name=method,
                batch_size=5,
                threshold_pct=10,
                seed=0,
                n_cycles=2,
                n_samples=50,
                n_initial=10,
            )
            results = run_experiment(config, verbose=False)
            assert results.success, f"Method {method} failed: {results.error_message}"

    def test_run_experiment_all_datasets(self):
        for dataset in list_datasets():
            config = ExperimentConfig(
                dataset_name=dataset,
                method_name='random',
                batch_size=5,
                threshold_pct=10,
                seed=0,
                n_cycles=2,
                n_samples=50,
                n_initial=10,
            )
            results = run_experiment(config, verbose=False)
            assert results.success, f"Dataset {dataset} failed: {results.error_message}"

    def test_experiment_results_structure(self):
        config = ExperimentConfig(
            dataset_name='sine',
            method_name='top_k',
            batch_size=5,
            threshold_pct=10,
            seed=42,
            n_cycles=3,
            n_samples=60,
            n_initial=10,
        )
        results = run_experiment(config, verbose=False)
        assert results.success
        assert hasattr(results, 'cycles')
        assert hasattr(results, 'final_metrics')
        assert hasattr(results, 'config')
        for cycle in results.cycles:
            assert hasattr(cycle, 'metrics')
            assert 'RecoveredHitsRatio_Cumulative' in cycle.metrics

    def test_hit_recovery_non_decreasing(self):
        """Cumulative hit recovery should be non-decreasing across cycles."""
        config = ExperimentConfig(
            dataset_name='sine',
            method_name='top_k',
            batch_size=5,
            threshold_pct=10,
            seed=0,
            n_cycles=4,
            n_samples=80,
            n_initial=10,
        )
        results = run_experiment(config, verbose=False)
        assert results.success
        values = [c.metrics['RecoveredHitsRatio_Cumulative'] for c in results.cycles]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
