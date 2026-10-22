"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc, Clare Lyle, University of Oxford, Pascal Notin, University of Oxford
Copyright 2025 Andrea Rubbi, University of Cambridge
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import sys
import random as pyrandom
import csv
import ast
import pickle
import inspect
import torch
import pickle as pkl
import numpy as np
from typing import AnyStr, Dict, List, Optional
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches

path_list = ["../../", "../", "./"]
for path in path_list:
    sys.path.append(os.path.abspath(path))

import slingpy as sp
from slingpy.utils.logging import info, warn
from slingpy.utils.path_tools import PathTools
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource
from activehit.data import gt_data_synthesizer as ds
from activehit.data import toy_datasets
from activehit.methods.bayesian_optimization import ucb_sampling
from activehit.methods.bax_acquisition import bax_sampling_toy_experiment
from activehit.methods.thompson_sampling import thompson_sampling_toy_experiment
from activehit.methods.our_algorithms.thompson_hit.thompson_hit import ThompsonHitAcquisition
from activehit.methods.our_algorithms.thompson_ig.thompson_ig import ThompsonIGAcquisition
from activehit.methods.our_algorithms.greedy_kg_sum.greedy_kg_sum import GreedyKGSumAcquisition
from activehit.methods.our_algorithms.probability_hit.probability_hit import ProbabilityHitAcquisition
from activehit.methods.bax_acquisition import bax_sampling
from activehit.methods.random import random
from activehit.methods.our_algorithms.tok_k.top_k import TopKAcquisition
from activehit.methods.our_algorithms.q_thompson_fantasy.q_thompson_fantasy import QThompsonFantasyBatchAcquisition
from activehit.methods.our_algorithms.tq_thompson_sampling.tq_thompson_sampling import TQThompsonSamplingBatchAcquisition
from activehit.methods.our_algorithms.straddle.straddle import StraddleAcquisition
from activehit.methods.our_algorithms.truvar.truvar import TruVaRAcquisition
from activehit.methods.our_algorithms.eitau.eitau import EITauAcquisition
from activehit.models import toy_experiment_models

DEFAULT_NUM_SAMPLES = 250  # Default pool size; should be > batch_size * num_cycles
PLOT = True
DataSet = namedtuple(
    "DataSet",
    "training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y",
)

class CustomLoss(sp.TorchLoss):
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor]) -> torch.Tensor:
        loss = self.loss_fn(y_pred[0], y_true[0].float())
        return loss


class SingleCycleVizApplication(sp.AbstractBaseApplication):
    def __init__(
        self,
        model_name: AnyStr = "randomforest",
        acquisition_function_name: AnyStr = "random",
        output_directory: AnyStr = "",
        test_ratio: float = 0.2,
        hyperopt_children: bool = True,
        seed: int = 0,
        dataset_name: AnyStr = "branin",
        feature_set_name="",
        selected_indices_file_path="",
        test_indices_file_path="",
        **kwargs,
    ):
        self.model_name = model_name
        self.acquisition_function_name = acquisition_function_name
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.selected_indices_file_path = selected_indices_file_path
        self.test_indices_file_path = test_indices_file_path
        self.seed = seed
        with open(test_indices_file_path, "rb") as fp:
            self.test_indices = pickle.load(fp)
        with open(selected_indices_file_path, "rb") as fp:
            self.selected_indices = pkl.load(fp)

        super(SingleCycleVizApplication, self).__init__(
            output_directory=output_directory,
            seed=seed,
            evaluate=True,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
        )

        self.initialize_pool()
    # Compute and store hit threshold for toy datasets using hit_threshold_percentile
        try:
            y_full = None
            if hasattr(self, 'data_synthesizer') and self.data_synthesizer is not None:
                y_full = np.asarray(self.data_synthesizer.dataset[1]).reshape(-1)
            if y_full is not None and len(y_full) > 0:
                pct = getattr(self, 'hit_threshold_percentile', 20.0) / 100.0
                k = max(1, int(pct * len(y_full)))
                top_idx = np.argpartition(y_full, -k)[-k:]
                self.hit_threshold = float(np.min(y_full[top_idx]))
                print(f"[VizLoop] Using hit threshold τ = min(top {pct*100:.0f}%) = {self.hit_threshold:.4f} for dataset '{self.dataset_name}'.")
            else:
                self.hit_threshold = None
        except Exception:
            self.hit_threshold = None
        self.model = self.get_model()

    def initialize_pool(self):
        """Get the data as (x, value) pairs.
        
        Supported datasets (from manuscript):
        - sine: 1D sine wave (simple baseline)
        - branin: 2D Branin-Hoo function (optimization benchmark)
        - perturbation_sem: SEM-based perturbation with embeddings (6D)
        - perturbation_pathways: Pathway-structured perturbations (4D)
        """
        # Seed RNGs so that each cycle reuses the same dataset given the same seed
        try:
            np.random.seed(self.seed)
            pyrandom.seed(self.seed)
            torch.manual_seed(self.seed)
        except Exception:
            pass
        
        # Use unified toy_datasets module for all manuscript datasets
        num_samples = getattr(self, 'num_samples', None) or DEFAULT_NUM_SAMPLES
        if self.dataset_name in toy_datasets.DATASET_REGISTRY:
            X, y, meta = toy_datasets.get_dataset(self.dataset_name, n=num_samples, seed=self.seed)
            class _DataWrapper:
                """Minimal wrapper to mimic GTDataSynthesizer interface."""
                def __init__(self, X, y, meta=None):
                    self.dataset = (X, y)
                    self.meta = meta or {}
                def _generate_data(self):
                    return self.dataset
            gen_d = _DataWrapper(X, y, meta)
        else:
            available = ', '.join(toy_datasets.list_datasets())
            raise ValueError(f"Unknown dataset '{self.dataset_name}'. Available: {available}")
        
        self.data_synthesizer = gen_d
        self.data_synthesizer._generate_data()
        dataset_x = gen_d.dataset[0]
        dataset_y = gen_d.dataset[1]
        n = len(dataset_x)
        available_indices = list(range(n))
        test_indices = list(range(n))
        self.dataset = DataSet(**self.load_data())
        return dataset_x, available_indices, test_indices

    def init_data(self):
        self.datasets = DataSet(**self.load_data())

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        # Guard against any potential index mismatch by filtering indices to valid range
        n = len(self.data_synthesizer.dataset[0])
        try:
            sel = [int(i) for i in self.selected_indices if 0 <= int(i) < n]
        except Exception:
            sel = list(self.selected_indices)
            sel = [i for i in sel if isinstance(i, (int, np.integer)) and 0 <= int(i) < n]
        try:
            tst = [int(i) for i in self.test_indices if 0 <= int(i) < n]
        except Exception:
            tst = list(self.test_indices)
            tst = [i for i in tst if isinstance(i, (int, np.integer)) and 0 <= int(i) < n]
        if (len(sel) != len(self.selected_indices)) or (len(tst) != len(self.test_indices)):
            warn(
                f"Sanitized indices: selected {len(sel)}/{len(self.selected_indices)} valid, "
                f"test {len(tst)}/{len(self.test_indices)} valid for dataset length {n}."
            )
        return {
            "training_set_x": self.data_synthesizer.dataset[0][sel],
            "training_set_y": self.data_synthesizer.dataset[1][sel],
            "validation_set_x": self.data_synthesizer.dataset[0][tst],
            "validation_set_y": self.data_synthesizer.dataset[1][tst],
            "test_set_x": self.data_synthesizer.dataset[0][tst],
            "test_set_y": self.data_synthesizer.dataset[1][tst],
        }

    def get_metrics(self, set_name: AnyStr) -> List[sp.AbstractMetric]:
        return [
            sp.metrics.MeanAbsoluteError(),
            sp.metrics.RootMeanSquaredError(),
            sp.metrics.SymmetricMeanAbsolutePercentageError(),
            sp.metrics.SpearmanRho(),
            sp.metrics.TopKRecall(0.2, 0.1),
        ]

    def train_model(self, model) -> Optional[AbstractBaseModel]:
        self.model.fit(
            self.data_synthesizer.dataset[0][self.selected_indices],
            self.data_synthesizer.dataset[1][self.selected_indices],
        )

        return self.model

    def get_model(self) -> Optional[AbstractBaseModel]:
        # Hardcoded
        if self.model_name == "gp":
            ls = 1.0 if self.dataset_name == "branin" else 0.1
            sp_model = toy_experiment_models.GaussianProcessModel(
                None, noise_sigma=0.01, save_path_dir="./gpmodel", kernel_lengthscale=ls
            )
        elif self.model_name == "linear":
            # Need to adaptively update num dimensions
            sp_model = toy_experiment_models.BayesianLinearModel(
                1, prior_sigma=1.0, noise_sigma=0.1, save_path_dir="./linearmodel"
            )
        return sp_model

    def evaluate_model(
        self,
        model: AbstractBaseModel,
        dataset_x: AbstractDataSource,
        dataset_y: AbstractDataSource,
        with_print: bool = True,
        set_name: AnyStr = "",
        threshold=None,
    ):
        """
        Evaluates model performance.
        Args:
            model: The model to evaluate.
            dataset: The dataset used for evaluation.
            with_print: Whether or not to print results to stdout.
            set_name: The name of the dataset being evaluated.
            threshold: An evaluation threshold (derived in sample) for discrete classification metrics,
             or none if a threshold should automatically be selected.
        Returns:
            A dictionary with each entry corresponding to an evaluation metric with one or more associated values.
        """
        preds, var = model.predict(dataset_x)
        # var is now a 1D array of diagonal variances by default
        if var.ndim == 1:
            uncs = var
        else:
            uncs = np.diag(var)
        scores = {}

        for m in self.get_metrics(""):
            s = m.evaluate(preds.reshape(-1, 1), dataset_y.reshape(-1, 1), threshold)
            scores[str(m).split(" ")[0].split(".")[-1]] = s

        # Multi-K retrieval metrics (Top-K overlap between predicted and true top-K by f(x))
        y_arr = np.asarray(dataset_y).reshape(-1)
        p_arr = np.asarray(preds).reshape(-1)
        n_items = len(y_arr)
        Ks = [10, 20, 50]
        for K in Ks:
            kk = min(K, n_items)
            if kk <= 0:
                continue
            # fast top-k selection using argpartition, then form sets
            true_top_idx = np.argpartition(y_arr, -kk)[-kk:]
            pred_top_idx = np.argpartition(p_arr, -kk)[-kk:]
            set_true = set(int(i) for i in true_top_idx.tolist())
            set_pred = set(int(i) for i in pred_top_idx.tolist())
            overlap = len(set_true & set_pred)
            # Retrieval@K: fraction of predicted top-K that are truly in top-K
            retrieval_at_k = overlap / float(kk)
            scores[f"TopK@{kk}"] = retrieval_at_k
            # Also record the true/pred top-K index lists (as strings) for inspection
            # These will be ignored in aggregated numeric means but kept in per-run pickles.
            scores[f"TopTrueIdx@{kk}"] = ",".join(map(str, sorted(set_true)))
            scores[f"TopPredIdx@{kk}"] = ",".join(map(str, sorted(set_pred)))

        # Backward-compatible single-K recall (K=10) and expected downstream max
        top_y = set(sorted(set(np.argpartition(y_arr, -min(10, n_items))[-min(10, n_items):].tolist())))
        top_pred = set(sorted(set(np.argpartition(p_arr, -min(10, n_items))[-min(10, n_items):].tolist())))
        denom = max(1, len(top_pred))
        recall = len(top_y.intersection(top_pred)) / denom
        ds_score = downstream_score(dataset_x, y_arr, np.asarray(list(top_pred)).astype(int))
        scores["MyRecall"] = recall
        scores["ExpectedMax"] = ds_score
        return scores


class VizLoop(sp.AbstractBaseApplication):
    ACQUISITION_FUNCTIONS = [
        "random",
        "ucb",
        "topk_bax",
        "levelset_bax",
        "subsetmax_bax",
        "thompson_hit",
        "probability_hit",
        "top_k",
        "straddle",
        "truvar",
        "eitau",
    ]

    def __init__(
        self,
        model_name: AnyStr = "gp",
        acquisition_function_name: AnyStr = "random",
        acquisition_function_path: AnyStr = "custom",
        acquisition_batch_size: int = 1,
        num_active_learning_cycles: int = 20,
        feature_set_name: AnyStr = None,
        dataset_name: AnyStr = "perturbation_pathways",  # Default to manuscript dataset
        cache_directory: AnyStr = "cache",
        output_directory: AnyStr = "output_toy",
        test_ratio: float = 0.2,
        hyperopt_children: bool = True,
        schedule_on_slurm: bool = False,
        schedule_children_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 1,
        remote_execution_mem_limit_in_mb: int = 2048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        seed: int = 0,
        save_fig_dir: str = "toy_figures_nd",
        num_samples: int = None,  # Pool size; defaults to DEFAULT_NUM_SAMPLES
        hit_threshold_percentile: float = 20.0,  # Top X% are considered "hits" (5, 10, or 20)
    ):
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_path = acquisition_function_path
        self.seed = seed
        self.num_samples = num_samples if num_samples is not None else DEFAULT_NUM_SAMPLES
        self.hit_threshold_percentile = hit_threshold_percentile

        # Ensure the output directory depends on acquisition function
        output_directory = os.path.join(
            output_directory, acquisition_function_name, str(seed)
        )
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        PathTools.mkdir_if_not_exists(output_directory)
        self.acquisition_function = VizLoop.get_acquisition_function(
            self.acquisition_function_name, self.acquisition_function_path
        )

        self.acquisition_batch_size = acquisition_batch_size
        self.num_active_learning_cycles = num_active_learning_cycles
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.model_name = model_name
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.cache_directory = cache_directory
        self.schedule_children_on_slurm = schedule_children_on_slurm
        self.save_fig_dir = save_fig_dir
        # Store scheduling / remote exec params for later access (train_model, runner)
        self.schedule_on_slurm = schedule_on_slurm
        self.remote_execution_time_limit_days = remote_execution_time_limit_days
        self.remote_execution_mem_limit_in_mb = remote_execution_mem_limit_in_mb
        self.remote_execution_virtualenv_path = remote_execution_virtualenv_path
        self.remote_execution_num_cpus = remote_execution_num_cpus

        self._pathway_meta = None

        # Seed RNGs so all methods in a multi-run see the same target function and initial selection
        try:
            np.random.seed(self.seed)
            pyrandom.seed(self.seed)
            torch.manual_seed(self.seed)
        except Exception:
            pass
        self.initialize_pool()
        # Tidy plots: defer all plotting to per-cycle function
        super(VizLoop, self).__init__(
            output_directory=output_directory,
            seed=seed,
            evaluate=False,
            hyperopt=False,
            single_run=True,
            save_predictions=False,
            schedule_on_slurm=schedule_on_slurm,
            remote_execution_num_cpus=remote_execution_num_cpus,
            remote_execution_time_limit_days=remote_execution_time_limit_days,
            remote_execution_mem_limit_in_mb=remote_execution_mem_limit_in_mb,
            remote_execution_virtualenv_path=remote_execution_virtualenv_path,
        )

    @staticmethod
    def get_acquisition_function(
        acquisition_function_name: AnyStr, acquisition_function_path: AnyStr
    ):
        if acquisition_function_name == "random":
            #return ucb_sampling.RandomPlottingAcquisition()
            return random.RandomBatchAcquisitionFunction()
        elif acquisition_function_name == "ucb":
            return ucb_sampling.UCBAcquisition()
        elif acquisition_function_name == "topk_bax":
            return bax_sampling_toy_experiment.TopKBaxAcquisition()
        elif acquisition_function_name == "levelset_bax":
            return bax_sampling_toy_experiment.LevelSetBaxAcquisition()
        elif acquisition_function_name == "subsetmax_bax":
            return bax_sampling_toy_experiment.SubsetSelectBaxAcquisition()
        elif acquisition_function_name == "thompson_sampling":
            return thompson_sampling_toy_experiment.ThompsonSamplingcquisition()
        elif acquisition_function_name == "thompson_hit":
            return ThompsonHitAcquisition()
        elif acquisition_function_name == "thompson_ig":
            return ThompsonIGAcquisition()
        elif acquisition_function_name == "probability_hit":
            return ProbabilityHitAcquisition()
        elif acquisition_function_name == "top_k":
            return TopKAcquisition()
        elif acquisition_function_name == "q_thompson_fantasy":
            return QThompsonFantasyBatchAcquisition()
        elif acquisition_function_name == "tq_thompson_sampling":
            return TQThompsonSamplingBatchAcquisition()
        elif acquisition_function_name == "greedy_kg_sum":
            return GreedyKGSumAcquisition()
        elif acquisition_function_name == "straddle":
            return StraddleAcquisition(beta=1.96)
        elif acquisition_function_name == "truvar":
            return TruVaRAcquisition(beta=1.96)
        elif acquisition_function_name == "eitau":
            return EITauAcquisition()
        elif acquisition_function_name == "discobax":
            bax_topk_kvalue = 5
            bax_level_set_c = 1.0
            bax_subset_select_subset_size = 20
            bax_noise_type = "additive"
            bax_noise_lengthscale = 1.0
            bax_noise_outputscale = 1.0
            bax_num_samples_EIG = 20
            bax_num_samples_entropy = 20
            bax_entropy_average_mode = "arithmetic"
            bax_batch_selection_mode = "topk_EIG"
            return bax_sampling.BaxAcquisition(
            objective_function="discobax",
            subset_size=bax_subset_select_subset_size,
            noise_type=bax_noise_type,
            noise_lengthscale=bax_noise_lengthscale,
            noise_outputscale=bax_noise_outputscale,
            num_samples_EIG=bax_num_samples_EIG,
            num_samples_entropy=bax_num_samples_entropy,
            entropy_average_mode=bax_entropy_average_mode,
            batch_selection_mode=bax_batch_selection_mode,
            compat_mode="toy",
            )
        else:
            raise NotImplementedError()

    def initialize_pool(self):
        """Get the data as (x, value) pairs.
        
        Supported datasets (from manuscript):
        - sine: 1D sine wave (simple baseline)
        - branin: 2D Branin-Hoo function (optimization benchmark)
        - perturbation_sem: SEM-based perturbation with embeddings (6D)
        - perturbation_pathways: Pathway-structured perturbations (4D)
        """
        print(self.dataset_name, "Dataset in init")
        
        # Use unified toy_datasets module for all manuscript datasets
        if self.dataset_name in toy_datasets.DATASET_REGISTRY:
            X, y, meta = toy_datasets.get_dataset(self.dataset_name, n=self.num_samples, seed=self.seed)
            class _DataWrapper:
                """Minimal wrapper to mimic GTDataSynthesizer interface."""
                def __init__(self, X, y, meta=None):
                    self.dataset = (X, y)
                    self.meta = meta or {}
                def _generate_data(self):
                    return self.dataset
            gen_d = _DataWrapper(X, y, meta)
        else:
            available = ', '.join(toy_datasets.list_datasets())
            raise ValueError(f"Unknown dataset '{self.dataset_name}'. Available: {available}")

        # Materialize data and derive indices from actual dataset length to avoid off-by-one
        self.data_synthesizer = gen_d
        gen_d._generate_data()
        self._cache_pathway_metadata()
        dataset_x = gen_d.dataset[0]
        n_actual = len(dataset_x)
        available_indices = list(range(n_actual))
        test_indices = list(range(n_actual))
        return dataset_x, available_indices, test_indices

    def _cache_pathway_metadata(self):
        """Cache pathway metadata for perturbation_pathways dataset visualization."""
        self._pathway_meta = None
        try:
            if self.dataset_name == "perturbation_pathways" and hasattr(self, 'data_synthesizer'):
                meta = getattr(self.data_synthesizer, 'meta', None)
                if meta:
                    self._pathway_meta = meta
        except Exception as e:
            warn(f"Failed to cache pathway metadata for dataset {self.dataset_name}: {e}")
            self._pathway_meta = None

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return [
            sp.metrics.MeanAbsoluteError(),
            sp.metrics.RootMeanSquaredError(),
            sp.metrics.SymmetricMeanAbsolutePercentageError(),
            sp.metrics.SpearmanRho(),
        ]

    def get_model(self):
        return None

    def train_model(self, model) -> Optional[AbstractBaseModel]:
        single_cycle_application_args = {
            "model_name": self.model_name,
            "seed": self.seed,
            "remote_execution_time_limit_days": self.remote_execution_time_limit_days,
            "remote_execution_mem_limit_in_mb": self.remote_execution_mem_limit_in_mb,
            "remote_execution_virtualenv_path": self.remote_execution_virtualenv_path,
            "remote_execution_num_cpus": self.remote_execution_num_cpus,
            "schedule_on_slurm": self.schedule_children_on_slurm,
        }
        cumulative_indices = []
        dataset_x, available_indices, test_indices = self.initialize_pool()
    # Compute and store hit threshold for toy datasets: min of top X% f(x)
        try:
            y_full = None
            if hasattr(self, 'data_synthesizer') and self.data_synthesizer is not None:
                y_full = np.asarray(self.data_synthesizer.dataset[1]).reshape(-1)
            if y_full is not None and len(y_full) > 0:
                pct = getattr(self, 'hit_threshold_percentile', 20.0) / 100.0
                k = max(1, int(pct * len(y_full)))
                top_idx = np.argpartition(y_full, -k)[-k:]
                self.hit_threshold = float(np.min(y_full[top_idx]))
                print(f"[SingleCycle] Using hit threshold τ = min(top {pct*100:.0f}%) = {self.hit_threshold:.4f} for dataset '{self.dataset_name}'.")
            else:
                self.hit_threshold = None
        except Exception:
            self.hit_threshold = None
        # Reseed prior to initial random selection to ensure consistency across methods
        try:
            np.random.seed(self.seed)
            pyrandom.seed(self.seed)
            torch.manual_seed(self.seed)
        except Exception:
            pass
        last_selected_indices = sorted(
            list(
                np.random.choice(
                    available_indices,
                    size=int(self.acquisition_batch_size),
                    replace=False,
                )
            )
        )
        cumulative_indices += last_selected_indices
        result_records = list()
        for cycle_index in range(self.num_active_learning_cycles):
            current_cycle_directory = os.path.join(
                self.output_directory, f"cycle_{cycle_index}"
            )
            PathTools.mkdir_if_not_exists(current_cycle_directory)

            cumulative_indices_file_path = os.path.join(
                current_cycle_directory, "selected_indices.pickle"
            )
            with open(cumulative_indices_file_path, "wb") as fp:
                pickle.dump(cumulative_indices, fp)
            test_indices_file_path = os.path.join(
                current_cycle_directory, "test_indices.pickle"
            )
            with open(test_indices_file_path, "wb") as fp:
                pickle.dump(test_indices, fp)

            app = SingleCycleVizApplication(
                dataset_name=self.dataset_name,
                feature_set_name=self.feature_set_name,
                cache_directory=self.cache_directory,
                output_directory=current_cycle_directory,
                train_ratio=0.8,
                hyperopt=self.hyperopt_children,
                selected_indices_file_path=cumulative_indices_file_path,
                test_indices_file_path=test_indices_file_path,
                acquisition_function_name=self.acquisition_function_name,  # Pass the actual method name for correct logging
                **single_cycle_application_args,
            )

            results = app.run().run_result
            info(results.test_scores, "test scores")
            result_records.append(results.test_scores)
            available_indices = sorted(
                list(set(available_indices) - set(last_selected_indices))
            )

            trained_model_path = results.model_path

            # Try to load a persisted model; fall back to the in-memory one
            trained_model = None
            if hasattr(app, "model") and app.model is not None:
                try:
                    trained_model = app.model.load(app.model.get_model_save_file_name())
                except Exception:
                    trained_model = app.model
            if trained_model is None and hasattr(app, "get_model"):
                trained_model = app.get_model()
            # Ensure the model can predict
            if trained_model is None or not hasattr(trained_model, "predict"):
                raise ValueError("No predictive model available after training; cannot proceed with acquisition.")
            # Call acquisition with adaptive signature support
            acq_fn = self.acquisition_function
            sig = inspect.signature(acq_fn)
            param_names = [p.name for p in sig.parameters.values()]
            # Drop 'self' if present
            if param_names and param_names[0] == 'self':
                param_names = param_names[1:]

            call_args = [dataset_x, self.acquisition_batch_size, available_indices]
            # The 4th positional param can be either 'last_selected_indices' or 'cumulative_indices'
            if len(param_names) >= 4:
                fourth = param_names[3]
                if fourth == 'last_selected_indices':
                    call_args.append(last_selected_indices)
                elif fourth == 'cumulative_indices':
                    call_args.append(cumulative_indices)
            # Keyword extras
            acq_kwargs = {}
            if 'model' in param_names:
                acq_kwargs['model'] = trained_model
            if 'dataset_y' in param_names:
                # Provide ground-truth y for acquisitions that need values for conditioning/fantasies
                try:
                    acq_kwargs['dataset_y'] = self.data_synthesizer.dataset[1]
                except Exception:
                    pass

            # Provide hit threshold to methods that accept it
            if 'threshold' in param_names and hasattr(self, 'hit_threshold') and self.hit_threshold is not None:
                acq_kwargs['threshold'] = self.hit_threshold
                print(f"[VizLoop] Passing hit threshold τ={self.hit_threshold:.4f} to acquisition '{self.acquisition_function_name}'.")
            # Provide hit threshold to methods that accept it
            if 'threshold' in param_names and hasattr(self, 'hit_threshold') and self.hit_threshold is not None:
                acq_kwargs['threshold'] = self.hit_threshold
                print(f"[SingleCycle] Passing hit threshold τ={self.hit_threshold:.4f} to acquisition '{self.acquisition_function_name}'.")
            last_selected_indices = acq_fn(*call_args, **acq_kwargs)
            cumulative_indices.extend(last_selected_indices)
            cumulative_indices = sorted(list(set(cumulative_indices)))
            assert len(last_selected_indices) == self.acquisition_batch_size
            # Retrain a model including the just-sampled points for plotting and reporting
            try:
                updated_model = app.get_model()
                updated_model.fit(
                    self.data_synthesizer.dataset[0][cumulative_indices],
                    self.data_synthesizer.dataset[1][cumulative_indices],
                )
            except Exception:
                updated_model = trained_model
            # Record cumulative real/predicted sums of sampled points (post-selection)
            try:
                # Ground truth values
                y_true = self.data_synthesizer.dataset[1]
                y_true = np.asarray(y_true).reshape(-1)
                # Predictive mean and covariance over the full dataset (compute ONCE per cycle)
                try:
                    pred = updated_model.predict(dataset_x, return_std_and_margin=True)
                except TypeError:
                    pred = updated_model.predict(dataset_x)
                pred_std_vec = None
                if isinstance(pred, tuple):
                    pred_mean = pred[0]
                    second = pred[1]
                    try:
                        pred_std_vec = np.sqrt(np.clip(np.diag(second), 0.0, None))
                    except Exception:
                        pred_std_vec = np.asarray(second).reshape(-1)
                else:
                    pred_mean = pred
                    pred_std_vec = np.zeros_like(np.asarray(pred).reshape(-1))
                pred_mean = np.asarray(pred_mean).reshape(-1)
                # Indices to ints
                cum_idx = list(map(int, np.asarray(cumulative_indices).tolist()))
                real_sum = float(np.sum(y_true[cum_idx]))
                pred_sum = float(np.sum(pred_mean[cum_idx]))
                if result_records and isinstance(result_records[-1], dict):
                    result_records[-1]["CumulativeRealSum"] = real_sum
                    result_records[-1]["CumulativePredictedSum"] = pred_sum
                    # Ratio of recovered hit points (cumulative and per-cycle) if threshold available
                    try:
                        if hasattr(self, 'hit_threshold') and self.hit_threshold is not None:
                            tau = float(self.hit_threshold)
                            true_hit_mask = y_true >= tau
                            total_true_hits = int(np.sum(true_hit_mask))
                            # cumulative recovered hits
                            recovered_hits_cum = int(np.sum(true_hit_mask[np.asarray(cum_idx, dtype=int)]))
                            ratio_recovered_hits_cum = (recovered_hits_cum / total_true_hits) if total_true_hits > 0 else 0.0
                            result_records[-1]["RecoveredHits_Cumulative"] = recovered_hits_cum
                            result_records[-1]["RecoveredHitsRatio_Cumulative"] = ratio_recovered_hits_cum
                            # per-cycle recovered hits
                            new_idx = list(map(int, np.asarray(last_selected_indices).tolist()))
                            recovered_hits_cycle = int(np.sum(true_hit_mask[np.asarray(new_idx, dtype=int)]))
                            denom_cycle = len(new_idx) if len(new_idx) > 0 else 1
                            ratio_recovered_hits_cycle = recovered_hits_cycle / denom_cycle
                            result_records[-1]["RecoveredHits_Cycle"] = recovered_hits_cycle
                            result_records[-1]["RecoveredHitsRatio_Cycle"] = ratio_recovered_hits_cycle
                    except Exception as e:
                        warn(f"Failed to compute recovered hit ratios at cycle {cycle_index}: {e}")
                
                # Compute SMAPE on unseen (unsampled) points
                try:
                    all_indices = set(range(len(y_true)))
                    unseen_indices = sorted(list(all_indices - set(cum_idx)))
                    if len(unseen_indices) > 0:
                        y_unseen = y_true[unseen_indices]
                        pred_unseen = pred_mean[unseen_indices]
                        # SMAPE = 100 * mean(|y - pred| / (|y| + |pred| + eps))
                        eps = 1e-8
                        smape_unseen = 100.0 * np.mean(
                            np.abs(y_unseen - pred_unseen) / (np.abs(y_unseen) + np.abs(pred_unseen) + eps)
                        )
                        result_records[-1]["SMAPE_Unseen"] = float(smape_unseen)
                        # Also compute RMSE on unseen
                        rmse_unseen = np.sqrt(np.mean((y_unseen - pred_unseen) ** 2))
                        result_records[-1]["RMSE_Unseen"] = float(rmse_unseen)
                        # MAE on unseen
                        mae_unseen = np.mean(np.abs(y_unseen - pred_unseen))
                        result_records[-1]["MAE_Unseen"] = float(mae_unseen)
                except Exception as e:
                    warn(f"Failed to compute unseen metrics at cycle {cycle_index}: {e}")
                
                # Compute hit cluster recovery metrics
                try:
                    if hasattr(self, 'hit_threshold') and self.hit_threshold is not None:
                        tau = float(self.hit_threshold)
                        true_hit_mask = y_true >= tau
                        hit_indices = np.where(true_hit_mask)[0]
                        
                        if len(hit_indices) > 1:
                            # Cluster hits by feature proximity (DBSCAN-like grouping)
                            from scipy.cluster.hierarchy import fcluster, linkage
                            from scipy.spatial.distance import pdist
                            
                            hit_features = dataset_x[hit_indices]
                            # Use hierarchical clustering with distance threshold
                            if len(hit_indices) > 1:
                                distances = pdist(hit_features)
                                if len(distances) > 0 and np.max(distances) > 0:
                                    # Cluster with threshold = 20% of max distance
                                    threshold = 0.2 * np.max(distances)
                                    Z = linkage(distances, method='average')
                                    clusters = fcluster(Z, t=threshold, criterion='distance')
                                    n_hit_clusters = len(np.unique(clusters))
                                    
                                    # Count how many clusters are represented in sampled points
                                    sampled_hit_indices = np.intersect1d(hit_indices, cum_idx)
                                    if len(sampled_hit_indices) > 0:
                                        # Map sampled hits to their cluster IDs
                                        sampled_in_hits = np.isin(hit_indices, sampled_hit_indices)
                                        sampled_clusters = np.unique(clusters[sampled_in_hits])
                                        n_clusters_recovered = len(sampled_clusters)
                                    else:
                                        n_clusters_recovered = 0
                                    
                                    result_records[-1]["HitClusters_Total"] = n_hit_clusters
                                    result_records[-1]["HitClusters_Recovered"] = n_clusters_recovered
                                    result_records[-1]["HitClusters_RecoveryRatio"] = (
                                        n_clusters_recovered / n_hit_clusters if n_hit_clusters > 0 else 0.0
                                    )
                except Exception as e:
                    warn(f"Failed to compute hit cluster metrics at cycle {cycle_index}: {e}")
            except Exception as e:
                warn(f"Failed to compute cumulative sums at cycle {cycle_index}: {e}")
            if PLOT:
                # Reuse the just-computed predictive mean/std to avoid duplicate expensive predict
                self.plot_mean_and_sd(
                    updated_model,
                    dataset_x,
                    cycle_index,
                    cumulative_indices=cumulative_indices,
                    last_selected_indices=last_selected_indices,
                    precomputed_mean=pred_mean,
                    precomputed_std=pred_std_vec,
                )

        results_path = os.path.join(self.output_directory, "results.pickle")
        with open(results_path, "wb") as fp:
            pickle.dump(result_records, fp)
        return None

    def plot_mean_and_sd(
        self,
        model,
        dataset_x,
        iteration,
        cumulative_indices,
        last_selected_indices,
        precomputed_mean=None,
        precomputed_std=None,
    ):
        plt.clf()
        y_true = self.data_synthesizer.dataset[1]
        try:
            y_true = np.asarray(y_true).reshape(-1)
        except Exception:
            y_true = np.asarray(y_true)

        if precomputed_mean is not None:
            plot_mean = np.asarray(precomputed_mean).reshape(-1)
            plot_std = (
                np.asarray(precomputed_std).reshape(-1)
                if precomputed_std is not None
                else np.zeros_like(plot_mean)
            )
        else:
            try:
                pred = model.predict(dataset_x, return_std_and_margin=True)
                if isinstance(pred, tuple) and len(pred) >= 2:
                    plot_mean = np.asarray(pred[0]).reshape(-1)
                    second = pred[1]
                    try:
                        plot_std = np.sqrt(np.clip(np.diag(second), 0.0, None))
                    except Exception:
                        plot_std = np.asarray(second).reshape(-1)
                else:
                    plot_mean = np.asarray(pred).reshape(-1)
                    plot_std = np.zeros_like(plot_mean)
            except Exception:
                plot_mean = np.zeros(len(y_true))
                plot_std = np.zeros_like(plot_mean)

        total_points = len(plot_mean)
        pathway_ctx = self._prepare_pathway_context(total_points)

        if pathway_ctx is not None:
            fig = plt.figure(figsize=(10, 5.2))
            gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 0.55], hspace=0.06)
            ax = fig.add_subplot(gs[0])
            ribbon_ax = fig.add_subplot(gs[1], sharex=ax)
            ribbon_ax.set_yticks([])
            ax.tick_params(axis='x', labelbottom=False)
        else:
            fig = plt.figure(figsize=(8.2, 4.3))
            ax = fig.add_subplot(111)
            ribbon_ax = None

        order = None
        inv_order = None
        y_plot = y_true
        mean_plot = plot_mean
        std_plot = plot_std
        if pathway_ctx is not None:
            order = pathway_ctx['order']
            inv_order = pathway_ctx['inverse_order']
            try:
                y_plot = y_true[order]
                mean_plot = plot_mean[order]
                std_plot = plot_std[order]
            except Exception:
                y_plot = np.asarray(y_true)
                mean_plot = np.asarray(plot_mean)
                std_plot = np.asarray(plot_std)

        x_idx = np.arange(len(mean_plot))
        
        # Add pathway background shading
        if pathway_ctx is not None:
            ordered_ids = pathway_ctx['path_ids_ordered']
            color_lookup = pathway_ctx['color_lookup']
            unique_ids = pathway_ctx['unique_ids']
            for pid in unique_ids:
                mask = ordered_ids == pid
                if np.any(mask):
                    indices = np.where(mask)[0]
                    ax.axvspan(indices[0] - 0.5, indices[-1] + 0.5,
                              color=color_lookup[pid], alpha=0.12, zorder=0)
        
        try:
            ax.plot(x_idx, y_plot, color='k', linewidth=1.0, alpha=0.35, zorder=1)
        except Exception:
            pass
        ax.plot(x_idx, mean_plot, color='C0', linewidth=2.0, label='posterior mean', zorder=2)
        try:
            ax.fill_between(
                x_idx,
                mean_plot - std_plot,
                mean_plot + std_plot,
                color='C0',
                alpha=0.15,
                label='±1σ'
            )
        except Exception:
            pass

        def _sanitize_indices(seq):
            if seq is None:
                return []
            sanitized = []
            try:
                iter_seq = list(seq)
            except Exception:
                return sanitized
            for val in iter_seq:
                try:
                    iv = int(val)
                except Exception:
                    continue
                if 0 <= iv < len(y_true):
                    sanitized.append(iv)
            return sanitized

        cum_idx = _sanitize_indices(cumulative_indices)
        new_idx = _sanitize_indices(last_selected_indices)

        def _gather_points(indices):
            xs, ys, cols = [], [], []
            for idx in indices:
                if inv_order is not None:
                    if idx < 0 or idx >= len(inv_order):
                        continue
                    pos = int(inv_order[idx])
                else:
                    pos = idx
                xs.append(pos)
                ys.append(float(y_true[idx]))
                if pathway_ctx is not None:
                    pid = int(pathway_ctx['path_ids_original'][idx])
                    cols.append(pathway_ctx['color_lookup'].get(pid, '#666666'))
            return xs, ys, cols

        xs_cum, ys_cum, cols_cum = _gather_points(cum_idx)
        xs_new, ys_new, cols_new = _gather_points(new_idx)

        if xs_cum:
            if pathway_ctx is not None and len(cols_cum) == len(xs_cum):
                ax.scatter(xs_cum, ys_cum, c=cols_cum, s=30, alpha=0.75, linewidths=0, zorder=3)
            else:
                ax.scatter(xs_cum, ys_cum, color='0.6', s=30, alpha=0.75, linewidths=0, zorder=3)
        if xs_new:
            if pathway_ctx is not None and len(cols_new) == len(xs_new):
                ax.scatter(
                    xs_new,
                    ys_new,
                    c=cols_new,
                    edgecolors='k',
                    linewidths=0.6,
                    s=48,
                    zorder=4,
                )
            else:
                ax.scatter(xs_new, ys_new, color='C3', edgecolors='k', linewidths=0.6, s=48, zorder=4)

        legend_handles, legend_labels = ax.get_legend_handles_labels()
        try:
            if hasattr(self, 'hit_threshold') and self.hit_threshold is not None:
                th = float(self.hit_threshold)
                ax.axhline(
                    y=th,
                    color='C4',
                    linestyle='--',
                    linewidth=1.4,
                    alpha=0.85,
                    label=f'hit τ={th:.3f}'
                )
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        except Exception:
            pass

        # Only show legend entries for model curves / hit threshold,
        # not a separate entry for every pathway color (pathways are
        # already encoded in the ribbon beneath the plot).
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper right', frameon=True, fontsize=9)

        if pathway_ctx is not None and ribbon_ax is not None:
            ordered_ids = pathway_ctx['path_ids_ordered']
            color_lookup = pathway_ctx['color_lookup']
            if len(ordered_ids) > 0:
                color_row = [mcolors.to_rgba(color_lookup[int(pid)]) for pid in ordered_ids]
                ribbon_ax.imshow(
                    np.array(color_row)[np.newaxis, :, :],
                    aspect='auto',
                    interpolation='nearest'
                )
                ribbon_ax.set_xlim(-0.5, len(color_row) - 0.5)
                ribbon_ax.set_xticks([])
                ribbon_ax.set_xlabel(pathway_ctx['x_label'], fontsize=10)

        ax.set_ylabel('f(x) / posterior mean')
        if pathway_ctx is None:
            ax.set_xlabel('sample index')
        ax.set_title(f"{self.acquisition_function_name} – cycle {iteration}")

        plt.tight_layout()
        try:
            os.makedirs(self.save_fig_dir, exist_ok=True)
        except Exception:
            pass
        fig.savefig(f"{self.save_fig_dir}/mean_unc_{self.acquisition_function_name}_{iteration}.png", dpi=140)
        plt.close(fig)

    @staticmethod
    def _pathway_palette(num_items: int) -> List[str]:
        if num_items <= 0:
            return []
        base = plt.get_cmap('tab20')
        return [mcolors.to_hex(base(i % base.N)) for i in range(num_items)]

    def _prepare_pathway_context(self, total_points: int) -> Optional[dict]:
        """Pathway context is not used for manuscript-approved datasets."""
        return None

    def evaluate_model(
        self,
        model: AbstractBaseModel,
        dataset_x: AbstractDataSource,
        dataset_y: AbstractDataSource,
        with_print: bool = True,
        set_name: AnyStr = "",
        threshold=None,
    ):
        """
        Evaluates model performance.
        Args:
            model: The model to evaluate.
            dataset: The dataset used for evaluation.
            with_print: Whether or not to print results to stdout.
            set_name: The name of the dataset being evaluated.
            threshold: An evaluation threshold (derived in sample) for discrete classification metrics,
             or none if a threshold should automatically be selected.
        Returns:
            A dictionary with each entry corresponding to an evaluation metric with one or more associated values.
        """
        return NotImplementedError()


def downstream_score(
    x,
    true_y,
    test_indices,
    noise_dist=bax_sampling_toy_experiment.bernoulli_noise_sampler,
    noise_type="multiplicative",
    num_samples=5,
):
    maxes = []
    ys = []
    for i in range(num_samples):
        if noise_type == "multiplicative":
            downstream_y = noise_dist(x, true_y, lengthscale=1.0, seed=i)
        else:
            return NotImplementedError()
        maxes.append(np.max(downstream_y[test_indices]))
        ys.append(downstream_y)
    return np.mean(maxes)


class ToyExperimentRunner:
    def __init__(self, argv: list):
        self.argv = argv
        self.raw_methods_arg = None
        self.raw_out_dir = None
        self.raw_seeds_vals = []
        self.methods = None
        self.is_list_input = False

    def _parse_cli(self):
        all_methods = []
        for i, tok in enumerate(self.argv):
            if tok == "--acquisition_function_name" and i + 1 < len(self.argv):
                val = self.argv[i + 1]
                all_methods.append(val)
                if self.raw_methods_arg is None:
                    self.raw_methods_arg = val
            if tok == "--output_directory" and i + 1 < len(self.argv):
                self.raw_out_dir = self.argv[i + 1]
            if tok == "--seed" and i + 1 < len(self.argv):
                self.raw_seeds_vals.append(self.argv[i + 1])

        methods = None
        if len(all_methods) > 1:
            methods = [str(m).strip() for m in all_methods]
        else:
            if isinstance(self.raw_methods_arg, str):
                txt = self.raw_methods_arg.strip()
                parsed = None
                try:
                    if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("(") and txt.endswith(")")):
                        parsed = ast.literal_eval(txt)
                except Exception:
                    parsed = None
                if parsed is None:
                    parts = [p.strip() for p in txt.split(",") if p.strip()]
                    methods = parts if len(parts) > 1 else [txt]
                else:
                    methods = [str(m).strip() for m in parsed]
        self.methods = methods
        self.is_list_input = bool(methods and len(methods) > 1)

    def _parse_seeds(self) -> list:
        seeds_list = []
        if len(self.raw_seeds_vals) > 1:
            for s in self.raw_seeds_vals:
                try:
                    seeds_list.append(int(str(s).strip()))
                except Exception:
                    pass
        elif len(self.raw_seeds_vals) == 1:
            s0 = self.raw_seeds_vals[0]
            parsed = None
            txt = str(s0).strip()
            try:
                if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("(") and txt.endswith(")")):
                    parsed = ast.literal_eval(txt)
            except Exception:
                parsed = None
            if parsed is None:
                parts = [p.strip() for p in txt.split(",") if p.strip()]
                if len(parts) > 1:
                    for p in parts:
                        try:
                            seeds_list.append(int(p))
                        except Exception:
                            pass
                else:
                    try:
                        seeds_list = [int(txt)]
                    except Exception:
                        seeds_list = []
            else:
                for p in parsed:
                    try:
                        seeds_list.append(int(p))
                    except Exception:
                        pass
        return seeds_list

    def _adjust_argv_for_methods(self):
        if not self.is_list_input:
            return
        new_argv = []
        seen_first = False
        i = 0
        while i < len(self.argv):
            tok = self.argv[i]
            if tok == "--acquisition_function_name" and i + 1 < len(self.argv):
                if not seen_first:
                    new_argv.extend([tok, "ucb"])  # placeholder
                    seen_first = True
                i += 2
                while i < len(self.argv) and self.argv[i] == "--acquisition_function_name" and i + 1 < len(self.argv):
                    i += 2
                continue
            else:
                new_argv.append(tok)
                i += 1
        sys.argv = new_argv

    @staticmethod
    def _resolve_base_out_dir(base, raw_out_dir: Optional[str]) -> str:
        if raw_out_dir:
            base_out_dir = raw_out_dir
        else:
            base_out_dir = os.path.dirname(os.path.dirname(base.output_directory))
        os.makedirs(base_out_dir, exist_ok=True)
        return base_out_dir

    @staticmethod
    def _build_shared_kwargs(base, base_out_dir: str) -> dict:
        return dict(
            model_name=base.model_name,
            acquisition_batch_size=base.acquisition_batch_size,
            num_active_learning_cycles=base.num_active_learning_cycles,
            feature_set_name=base.feature_set_name,
            dataset_name=base.dataset_name,
            cache_directory=base.cache_directory,
            output_directory=base_out_dir,
            test_ratio=base.test_ratio,
            hyperopt_children=base.hyperopt_children,
            schedule_on_slurm=False,
            schedule_children_on_slurm=base.schedule_children_on_slurm,
            remote_execution_time_limit_days=base.remote_execution_time_limit_days,
            remote_execution_mem_limit_in_mb=base.remote_execution_mem_limit_in_mb,
            remote_execution_virtualenv_path=base.remote_execution_virtualenv_path,
            remote_execution_num_cpus=base.remote_execution_num_cpus,
            hit_threshold_percentile=base.hit_threshold_percentile,
        )

    @staticmethod
    def _run_one(meth: str, sed: int, base, shared_kwargs: dict, base_out_dir: str, save_fig_root: str) -> dict:
        try:
            np.random.seed(sed)
            pyrandom.seed(sed)
            torch.manual_seed(sed)
        except Exception:
            pass
        run_save_fig_dir = os.path.join(save_fig_root, meth, str(sed))
        loop = VizLoop(
            acquisition_function_name=meth,
            seed=sed,
            save_fig_dir=run_save_fig_dir,
            **shared_kwargs,
        )
        loop.run()
        plt.clf()

        method_out_dir = os.path.join(base_out_dir, meth, str(sed))
        res_path = os.path.join(method_out_dir, "results.pickle")
        last_scores = None
        if os.path.exists(res_path):
            try:
                with open(res_path, "rb") as fp:
                    recs = pickle.load(fp)
                if isinstance(recs, list) and len(recs) > 0:
                    last_scores = recs[-1]
            except Exception as e:
                warn(f"Failed to read results from {res_path}: {e}")
        if last_scores is None:
            last_scores = {"error": "no_results"}

        row = {"method": meth, "seed": sed}
        for k, v in last_scores.items():
            try:
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            row[k] = v
        return row

    @staticmethod
    def _write_per_seed_summary(base_out_dir: str, rows: list):
        per_seed_path = os.path.join(base_out_dir, "summary_by_seed.csv")
        if not rows:
            return
        header_keys = set()
        for r in rows:
            header_keys.update(r.keys())
        header = ["method", "seed"] + sorted([k for k in header_keys if k not in ("method", "seed")])
        os.makedirs(base_out_dir, exist_ok=True)
        with open(per_seed_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote per-seed summary CSV: {per_seed_path}")

    @staticmethod
    def _write_overall_agg(base_out_dir: str, rows: list):
        from collections import defaultdict
        agg = defaultdict(list)
        for row in rows:
            agg[row.get("method")].append(row)
        agg_rows = []
        for meth, meth_rows in agg.items():
            keys = set()
            for r in meth_rows:
                keys.update([k for k in r.keys() if k not in ("method", "seed")])
            out = {"method": meth}
            for k in sorted(keys):
                vals = []
                for r in meth_rows:
                    v = r.get(k)
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            vals.append(fv)
                    except Exception:
                        continue
                if len(vals) > 0:
                    out[f"{k}_mean"] = float(np.mean(vals))
                    out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            agg_rows.append(out)

        if agg_rows:
            agg_header_keys = set()
            for r in agg_rows:
                agg_header_keys.update(r.keys())
            agg_header = ["method"] + sorted([k for k in agg_header_keys if k != "method"])
            agg_path = os.path.join(base_out_dir, "summary_agg.csv")
            os.makedirs(base_out_dir, exist_ok=True)
            with open(agg_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=agg_header)
                writer.writeheader()
                for r in agg_rows:
                    writer.writerow(r)
            print(f"Wrote aggregated summary CSV (mean/std across seeds): {agg_path}")

    @staticmethod
    def _normalize_scores(d: dict) -> dict:
        norm = {}
        for k, v in d.items():
            try:
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            norm[k] = v
        return norm

    def _write_cycle_agg(self, methods_iter: list, seeds_iter: list, base_out_dir: str, total_cycles: Optional[int]):
        if not total_cycles or total_cycles < 5:
            return
        cycle_points = list(range(5, total_cycles + 1, 5))
        for c in cycle_points:
            agg_rows_cycle = []
            for meth in methods_iter:
                per_seed_metrics = []
                for sed in seeds_iter:
                    method_out_dir = os.path.join(base_out_dir, meth, str(sed))
                    res_path = os.path.join(method_out_dir, "results.pickle")
                    rec = None
                    if os.path.exists(res_path):
                        try:
                            with open(res_path, "rb") as fp:
                                recs = pickle.load(fp)
                            if isinstance(recs, list) and len(recs) >= c:
                                rec = recs[c - 1]
                        except Exception as e:
                            warn(f"Failed to read results for method {meth}, seed {sed} at cycle {c} from {res_path}: {e}")
                    if rec is None:
                        continue
                    per_seed_metrics.append(self._normalize_scores(rec))

                if not per_seed_metrics:
                    continue

                keys = set()
                for r in per_seed_metrics:
                    keys.update(r.keys())
                out = {"method": meth}
                for k in sorted(keys):
                    vals = []
                    for r in per_seed_metrics:
                        if k in r:
                            try:
                                fv = float(r[k])
                                if np.isfinite(fv):
                                    vals.append(fv)
                            except Exception:
                                pass
                    if len(vals) > 0:
                        out[f"{k}_mean"] = float(np.mean(vals))
                        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                agg_rows_cycle.append(out)

            if agg_rows_cycle:
                agg_keys = set()
                for r in agg_rows_cycle:
                    agg_keys.update(r.keys())
                agg_header = ["method"] + sorted([k for k in agg_keys if k != "method"])
                cyc_agg_path = os.path.join(base_out_dir, f"summary_cycle_{c}_agg.csv")
                os.makedirs(base_out_dir, exist_ok=True)
                with open(cyc_agg_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=agg_header)
                    writer.writeheader()
                    for r in agg_rows_cycle:
                        writer.writerow(r)
                print(f"Wrote aggregated summary CSV for cycle {c}: {cyc_agg_path}")

    def run(self):
        print("Starting")
        self._parse_cli()
        seeds_list = self._parse_seeds()
        self._adjust_argv_for_methods()

        base = sp.instantiate_from_command_line(VizLoop)
        # Always funnel through unified multi-run logic so plotting is consistent
        # (single-run previously short-circuited and skipped per-cycle plotting).
        # Removed early return to ensure figures are generated even for one method/seed.

        base_out_dir = self._resolve_base_out_dir(base, self.raw_out_dir)
        shared_kwargs = self._build_shared_kwargs(base, base_out_dir)

        methods_iter = self.methods if self.methods and len(self.methods) > 0 else [base.acquisition_function_name]
        seeds_iter = seeds_list if len(seeds_list) > 0 else [base.seed]

        per_seed_rows = []
        for meth in methods_iter:
            for sed in seeds_iter:
                row = self._run_one(meth, sed, base, shared_kwargs, base_out_dir, base.save_fig_dir)
                per_seed_rows.append(row)

        self._write_per_seed_summary(base_out_dir, per_seed_rows)
        self._write_overall_agg(base_out_dir, per_seed_rows)

        try:
            total_cycles = int(base.num_active_learning_cycles)
        except Exception:
            total_cycles = None
        self._write_cycle_agg(methods_iter, seeds_iter, base_out_dir, total_cycles)


def main():
    runner = ToyExperimentRunner(sys.argv[:])
    runner.run()


if __name__ == "__main__":
    main()
