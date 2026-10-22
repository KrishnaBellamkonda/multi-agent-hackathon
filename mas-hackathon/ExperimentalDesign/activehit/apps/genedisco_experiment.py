"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc, Clare Lyle, University of Oxford, Pascal Notin, University of Oxford
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

import pickle as pkl
import os
from collections import namedtuple
import random
import sys
import ast
import csv
import numpy as np
import pandas as pd
from typing import AnyStr, Dict, List, Optional
from collections import defaultdict

import slingpy as sp
from slingpy.utils.logging import info, warn

# from slingpy.evaluation.evaluator import Evaluator
from slingpy.utils.path_tools import PathTools
from slingpy import AbstractMetric, AbstractBaseModel, AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel

path_list = [os.path.abspath("."), os.path.abspath("..")]
for p in path_list:
    if p not in os.sys.path:
        os.sys.path.insert(0, p)

from activehit.apps import abstract_base_application as aba
from activehit.apps.genedisco_single_cycle_experiment import SingleCycleApplication
from activehit.models import clustering
from activehit.methods.bayesian_optimization import ucb_sampling
from activehit.methods.bax_acquisition import bax_sampling
from activehit.methods.thompson_sampling import thompson_sampling
from activehit.methods.our_algorithms.tok_k.top_k import TopKAcquisition
from activehit.methods.our_algorithms.thompson_hit.thompson_hit import ThompsonHitAcquisition
from activehit.methods.our_algorithms.thompson_ig.thompson_ig import ThompsonIGAcquisition
from activehit.methods.our_algorithms.greedy_kg_sum.greedy_kg_sum import GreedyKGSumAcquisition
from activehit.methods.our_algorithms.probability_hit.probability_hit import ProbabilityHitAcquisition
from activehit.methods.random import random
from activehit.methods.active_learning.kmeans import Kmeans
from activehit.methods.active_learning.core_set import CoreSet
from activehit.methods.active_learning.badge_sampling import BadgeSampling
from activehit.methods.active_learning.adversarial_BIM import AdversarialBIM
from activehit.methods.active_learning.top_soft_uncertainty import (
    TopUncertainAcquisition,
)
from activehit.methods.active_learning.top_soft_uncertainty import (
    SoftUncertainAcquisition,
)
from activehit.methods.active_learning.margin_sampling import MarginSamplingAcquisition
from activehit.methods.active_learning.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)

DataSet = namedtuple(
    "DataSet",
    "training_set_x, training_set_y, validation_set_x, validation_set_y, test_set_x, test_set_y",
)


class GeneDiscoLoop(aba.AbstractBaseApplication):
    def __init__(
        self,
        cache_directory: AnyStr,
        output_directory: AnyStr,
        dataset_name: AnyStr,
        feature_set_name: AnyStr = "achilles",  # Dattaset used for interventtion representation ["achilles" "ccle" "string"]
        model_name: AnyStr = "gp",
        acquisition_function_name: AnyStr = "random",
        acquisition_function_path: AnyStr = "custom",
        acquisition_batch_size: int = 20,
        num_active_learning_cycles: int = 20,
        test_ratio: float = 0.0,
        hyperopt_children: bool = False,
        schedule_on_slurm: bool = False,
        schedule_children_on_slurm: bool = False,
        remote_execution_time_limit_days: int = 1,
        remote_execution_mem_limit_in_mb: int = 5048,
        remote_execution_virtualenv_path: AnyStr = "",
        remote_execution_num_cpus: int = 1,
        seed: int = 1000,
        topk_percent: float = 0.1,
        hit_threshold: float = None,  # Manual threshold override for thompson_hit
        greedy_kg_H: int = None,  # Manual H override for greedy_kg_sum
        greedy_kg_L: int = 3,  # Number of fantasy samples for greedy_kg_sum (lower=faster)
        greedy_kg_batch_strategy: str = "topk-approx",  # "topk-approx" (fast) or "sequential" (slower but more accurate)
        performance_file_location: str = "output/performance_file.csv",
        bax_topk_kvalue: int = 5,
        bax_level_set_c: float = 1.0,
        bax_subset_select_subset_size: int = 20,
        bax_noise_type: str = "additive",
        bax_noise_lengthscale: float = 1.0,
        bax_noise_outputscale: float = 1.0,
        bax_num_samples_EIG: int = 20,
        bax_num_samples_entropy: int = 20,
        bax_entropy_average_mode: str = "arithmetic",
        bax_batch_selection_mode: str = "topk_EIG",
        num_topk_clusters: int = 20,
    ):
        self.acquisition_function_name = acquisition_function_name
        self.acquisition_function_path = acquisition_function_path
        if acquisition_function_name == "discobax":
            hyper = str(bax_subset_select_subset_size)
        elif acquisition_function_name == "topk_bax":
            hyper = str(bax_topk_kvalue)
        elif acquisition_function_name == "levelset_bax":
            hyper = str(bax_level_set_c)
        else:
            hyper = ""
        self.run_name = "_".join(
            [
                dataset_name,
                feature_set_name,
                str(topk_percent),
                model_name,
                acquisition_function_name,
                hyper,
                str(seed),
            ]
        )
        PathTools.mkdir_if_not_exists(output_directory)
        output_directory = output_directory + os.sep + self.run_name
        PathTools.mkdir_if_not_exists(output_directory)
        self.temp_folder_name = output_directory + os.sep + "tmp"
        PathTools.mkdir_if_not_exists(self.temp_folder_name)
        self.performance_file_location = performance_file_location

        self.bax_topk_kvalue = bax_topk_kvalue
        self.bax_subset_select_subset_size = bax_subset_select_subset_size
        self.bax_level_set_c = bax_level_set_c
        self.bax_noise_type = bax_noise_type
        self.bax_noise_lengthscale = bax_noise_lengthscale
        self.bax_noise_outputscale = bax_noise_outputscale
        self.bax_num_samples_EIG = bax_num_samples_EIG
        self.bax_num_samples_entropy = bax_num_samples_entropy
        self.bax_entropy_average_mode = bax_entropy_average_mode
        self.bax_batch_selection_mode = bax_batch_selection_mode

        # Set instance variables BEFORE calling get_acquisition_function()
        self.acquisition_batch_size = acquisition_batch_size
        self.num_active_learning_cycles = num_active_learning_cycles
        self.dataset_name = dataset_name
        self.feature_set_name = feature_set_name
        self.model_name = model_name
        self.hyperopt_children = hyperopt_children
        self.test_ratio = test_ratio
        self.cache_directory = cache_directory
        self.schedule_children_on_slurm = schedule_children_on_slurm
        self.topk_percent = topk_percent
        self.hit_threshold = hit_threshold  # Store manual threshold override
        self.greedy_kg_H = greedy_kg_H  # Store manual H override
        self.greedy_kg_L = greedy_kg_L  # Store fantasy samples count for greedy_kg_sum
        self.greedy_kg_batch_strategy = greedy_kg_batch_strategy  # Store batch strategy for greedy_kg_sum
        
        # Now we can safely call get_acquisition_function() which uses the above variables
        self.acquisition_function = self.get_acquisition_function()
        
        self.topk_indices = None
        self.topk_clusters_to_item = None
        self.topk_item_to_clusters = None
        self.num_topk_clusters = num_topk_clusters
        super(GeneDiscoLoop, self).__init__(
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

    def get_acquisition_function(self) -> BaseBatchAcquisitionFunction:
        if self.acquisition_function_name == "random":
            return random.RandomBatchAcquisitionFunction()
        elif self.acquisition_function_name == "ucb":
            return ucb_sampling.UCBAcquisition()
        elif self.acquisition_function_name == "topk_bax":
            return bax_sampling.BaxAcquisition(
                objective_function="topk",
                k=self.bax_topk_kvalue,
                num_samples_EIG=self.bax_num_samples_EIG,
                num_samples_entropy=self.bax_num_samples_entropy,
                entropy_average_mode=self.bax_entropy_average_mode,
                batch_selection_mode=self.bax_batch_selection_mode,
            )
        elif self.acquisition_function_name == "levelset_bax":
            return bax_sampling.BaxAcquisition(
                objective_function="level_set",
                c=self.bax_level_set_c,
                num_samples_EIG=self.bax_num_samples_EIG,
                num_samples_entropy=self.bax_num_samples_entropy,
                entropy_average_mode=self.bax_entropy_average_mode,
                batch_selection_mode=self.bax_batch_selection_mode,
            )
        elif self.acquisition_function_name == "discobax":
            return bax_sampling.BaxAcquisition(
                objective_function="discobax",
                subset_size=self.bax_subset_select_subset_size,
                noise_type=self.bax_noise_type,
                noise_lengthscale=self.bax_noise_lengthscale,
                noise_outputscale=self.bax_noise_outputscale,
                num_samples_EIG=self.bax_num_samples_EIG,
                num_samples_entropy=self.bax_num_samples_entropy,
                entropy_average_mode=self.bax_entropy_average_mode,
                batch_selection_mode=self.bax_batch_selection_mode,
            )
        elif self.acquisition_function_name == "jepig":
            return bax_sampling.BaxAcquisition(
                objective_function="jepig",
                subset_size=self.bax_subset_select_subset_size,
                num_samples_EIG=self.bax_num_samples_EIG,
                num_samples_entropy=self.bax_num_samples_entropy,
                entropy_average_mode=self.bax_entropy_average_mode,
                batch_selection_mode=self.bax_batch_selection_mode,
            )
        elif self.acquisition_function_name == "top_k":
            # Greedy top-K by predicted value with uncertainty-aware initial step.
            return TopKAcquisition()
        elif self.acquisition_function_name == "thompson_hit":
            # Thompson sampling on the number of hits above a threshold.
            # Threshold will be provided at call time via the keyword argument
            # `threshold`; if omitted it defaults to 0.0.
            return ThompsonHitAcquisition()
        elif self.acquisition_function_name == "thompson_ig":
            # Batched Thompson Sampling with Information-Gain Exploration
            # Combines Thompson sampling (early cycles) with information gain (final cycle)
            # for better exploration-exploitation trade-off
            return ThompsonIGAcquisition()
        elif self.acquisition_function_name == "probability_hit":
            # Probability-of-hit greedy batch acquisition (see toy_experiment for details)
            return ProbabilityHitAcquisition()
        elif self.acquisition_function_name == "greedy_kg_sum":
            # One-step knowledge-gradient style acquisition targeting cumulative sum.
            # OPTIMIZATION: Default to topk-approx strategy and L=3 for speed
            # topk-approx: O(n_candidates × L) vs sequential: O(batch_size × n_candidates × L)
            # L=3 vs L=8: ~2.7x speedup with minimal accuracy loss
            return GreedyKGSumAcquisition(
                L=self.greedy_kg_L,  # Number of fantasy samples (default 3, was 8)
                batch_strategy=self.greedy_kg_batch_strategy,  # "topk-approx" (fast) or "sequential" (accurate)
                compat_mode="auto"
            )
        elif self.acquisition_function_name == "thompson_sampling":
            return thompson_sampling.ThompsonSamplingcquisition()
        elif self.acquisition_function_name == "topuncertain":
            return TopUncertainAcquisition()
        elif self.acquisition_function_name == "softuncertain":
            return SoftUncertainAcquisition()
        elif self.acquisition_function_name == "marginsample":
            return MarginSamplingAcquisition()
        elif self.acquisition_function_name == "badge":
            return BadgeSampling()
        elif self.acquisition_function_name == "coreset":
            return CoreSet()
        elif self.acquisition_function_name == "kmeans_embedding":
            return Kmeans(representation="linear", n_init=10)
        elif self.acquisition_function_name == "kmeans_data":
            return Kmeans(representation="raw", n_init=10)
        elif self.acquisition_function_name == "adversarialBIM":
            return AdversarialBIM()
        else:
            raise NotImplementedError()

    def initialize_pool(self):
        dataset_x = SingleCycleApplication.get_dataset_x(
            self.feature_set_name, self.cache_directory
        )
        dataset_y = SingleCycleApplication.get_dataset_y(
            self.dataset_name, self.cache_directory
        )
        available_indices = sorted(
            list(
                set(dataset_x.get_row_names()).intersection(
                    set(dataset_y.get_row_names())
                )
            )
        )
        test_indices = sorted(
            list(
                np.random.choice(
                    available_indices,
                    size=int(self.test_ratio * len(available_indices)),
                    replace=False,
                )
            )
        )
        available_indices = sorted(list(set(available_indices) - set(test_indices)))
        dataset_x = dataset_x.subset(available_indices)
        dataset_y = dataset_y.subset(available_indices)
        self.topk_indices = self.get_topk_indices(dataset_y, self.topk_percent)

        # Get basic stats
        try:
            df = pd.DataFrame(dataset_y.get_data()[0])
            print(df.describe())
        except:
            print("Errors getting stats")

        # Get clusters
        path_cache_cluster_files_prefix = (
            self.cache_directory
            + os.sep
            + "_".join(
                [
                    "clusters",
                    self.dataset_name,
                    self.feature_set_name,
                    str(self.topk_percent),
                ]
            )
        )
        if os.path.isfile(
            path_cache_cluster_files_prefix
            + f"_topk_{self.num_topk_clusters}_clusters_to_items.pkl"
        ) and os.path.isfile(
            path_cache_cluster_files_prefix
            + f"_topk_items_to_{self.num_topk_clusters}_clusters.pkl"
        ):
            print("Loading clusters from cache")
            with open(
                path_cache_cluster_files_prefix
                + f"_topk_{self.num_topk_clusters}_clusters_to_items.pkl",
                "rb",
            ) as fp:
                self.topk_clusters_to_item = pkl.load(fp)
            with open(
                path_cache_cluster_files_prefix
                + f"_topk_items_to_{self.num_topk_clusters}_clusters.pkl",
                "rb",
            ) as fp:
                self.topk_item_to_clusters = pkl.load(fp)
        else:
            print("Did not find clusters in cache -- computing top cluster assignment")
            (
                self.topk_clusters_to_item,
                self.topk_item_to_clusters,
            ) = clustering.get_top_target_clusters(
                dataset_x,
                self.topk_indices,
                num_clusters=self.num_topk_clusters,
                plot_location=self.output_directory
                + os.sep
                + "plot_optimal_number_top_clusters.png",
                path_cache_cluster_files_prefix=path_cache_cluster_files_prefix,
            )
        return dataset_x, available_indices, test_indices

    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        return {}

    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        return []

    def get_model(self, **kwargs):
        return None

    def get_topk_indices(
        self, dataset_y: AbstractDataSource, topk_percent: float = 0.1
    ):
        y_indices = dataset_y.get_row_names()
        y_values = pd.DataFrame(
            dataset_y.get_data()[0].flatten(), index=y_indices, columns=["y_values"]
        )
        y_values.sort_values(by=["y_values"], ascending=False, axis=0, inplace=True)
        num_k = int(topk_percent * len(y_values.y_values))
        topk_indices = list(y_values.iloc[:num_k].index)
        return topk_indices
    
    def compute_threshold_from_topk(
        self, dataset_y: AbstractDataSource, topk_percent: float = 0.1
    ):
        """
        Compute the threshold value corresponding to the top-k percentile.
        This is the minimum y-value among the top-k% highest scoring items.
        
        Args:
            dataset_y: Target values
            topk_percent: Percentage of top items (default 0.1 = top 10%)
            
        Returns:
            float: The threshold value at the top-k percentile boundary
        """
        y_values = dataset_y.get_data()[0].flatten()
        y_values_sorted = np.sort(y_values)[::-1]  # Sort descending
        num_k = int(topk_percent * len(y_values_sorted))
        if num_k == 0:
            num_k = 1
        # The threshold is the value at the k-th position (the minimum value in top-k)
        threshold = float(y_values_sorted[num_k - 1])
        return threshold

    def get_cumulative_precision_recall_topk(
        self,
        cumulative_indices: List[int],
        dataset_y: AbstractDataSource,
        topk_percent: float = 0.1,
    ):
        """
        topk_percent is the percent of top target recalled over the course of the different cycles
        """
        if self.topk_indices is None:
            self.topk_indices = self.get_topk_indices(dataset_y, topk_percent)
        precision, recall = len(set(self.topk_indices) & set(cumulative_indices)) / len(
            set(cumulative_indices)
        ), len(set(self.topk_indices) & set(cumulative_indices)) / len(
            set(self.topk_indices)
        )
        return precision, recall

    def get_cumulative_proportion_top_clusters_recovered(self, cumulative_indices):
        assert (
            self.topk_clusters_to_item is not None
        ), "Top clusters have not been computed yet"
        num_distinct_top_clusters = len(self.topk_clusters_to_item.keys())
        recovered_clusters_dict = defaultdict(int)
        for item in cumulative_indices:
            if item in self.topk_item_to_clusters:
                recovered_clusters_dict[self.topk_item_to_clusters[item]] += 1
        proportion_top_clusters_recovered = len(recovered_clusters_dict.keys()) / float(
            num_distinct_top_clusters
        )
        return proportion_top_clusters_recovered

    def get_precision_topk_last_selected_indices(
        self,
        last_selected_indices,
        dataset_y: AbstractDataSource,
        topk_percent: float = 0.1,
    ):
        if self.topk_indices is None:
            self.topk_indices = self.get_topk_indices(dataset_y, topk_percent)
        precision_last_selected = len(
            set(self.topk_indices) & set(last_selected_indices)
        ) / len(set(last_selected_indices))
        return precision_last_selected

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
        dataset_y = SingleCycleApplication.get_dataset_y(
            self.dataset_name, self.cache_directory
        )
        dataset_y = dataset_y.subset(available_indices)
        
        # Compute or use manual threshold for thompson_hit
        if self.hit_threshold is not None:
            hit_threshold = float(self.hit_threshold)
            print(f"INFO: Using MANUAL hit threshold: {hit_threshold:.6f} (overrides automatic computation)")
        else:
            # Compute threshold from dataset - corresponds to the boundary of the top-k% items
            hit_threshold = self.compute_threshold_from_topk(dataset_y, self.topk_percent)
            print(f"INFO: Computed hit threshold for top {self.topk_percent*100}% of dataset: {hit_threshold:.6f}")
        
        # Compute or use manual H for greedy_kg_sum
        if self.greedy_kg_H is not None:
            greedy_kg_H = int(self.greedy_kg_H)
            print(f"INFO: Using MANUAL H={greedy_kg_H} for greedy_kg_sum (overrides automatic computation)")
        else:
            # H is the number of top items we want to maximize the sum over
            greedy_kg_H = max(1, int(self.topk_percent * len(available_indices)))
            print(f"INFO: Computed H={greedy_kg_H} for greedy_kg_sum (top {self.topk_percent*100}% of {len(available_indices)} items)")
        
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
        cumulative_recall_topk = list()
        cumulative_precision_topk = list()
        cumulative_proportion_top_clusters_recovered = list()
        cumulative_num_topk_recovered = list()  # Track absolute count of top-k recovered
        for cycle_index in range(self.num_active_learning_cycles):
            current_cycle_directory = os.path.join(
                self.output_directory, f"cycle_{cycle_index}"
            )

            PathTools.mkdir_if_not_exists(current_cycle_directory)

            cumulative_indices_file_path = os.path.join(
                current_cycle_directory, "selected_indices.pkl"
            )
            with open(cumulative_indices_file_path, "wb") as fp:
                pkl.dump(cumulative_indices, fp)
            test_indices_file_path = os.path.join(
                current_cycle_directory, "test_indices.pkl"
            )
            with open(test_indices_file_path, "wb") as fp:
                pkl.dump(test_indices, fp)
            app = SingleCycleApplication(
                dataset_name=self.dataset_name,
                feature_set_name=self.feature_set_name,
                cache_directory=self.cache_directory,
                output_directory=current_cycle_directory,
                train_ratio=1.0,
                hyperopt=self.hyperopt_children,
                selected_indices_file_path=cumulative_indices_file_path,
                test_indices_file_path=test_indices_file_path,
                **single_cycle_application_args,
            )
            results = app.run().run_result
            info(results.test_scores, "test scores")
            result_records.append(results.test_scores)
            available_indices = sorted(
                list(set(available_indices) - set(last_selected_indices))
            )

            trained_model = app.model.load(results.model_path)
            if trained_model is None:
                print("Could not find trained model at specified path in results")
                trained_model = app.get_model()

            # Build acquisition function kwargs based on which acquisition is being used
            acq_kwargs = {
                'dataset_x': dataset_x,
                'acquisition_batch_size': self.acquisition_batch_size,
                'available_indices': available_indices,
                'last_selected_indices': last_selected_indices,
                'cumulative_indices': cumulative_indices,
                'model': trained_model,
                'dataset_y': dataset_y,
                'temp_folder_name': self.temp_folder_name,
            }
            
            # Add parameters conditionally based on acquisition function
            acq_func_name = self.acquisition_function_name.lower()
            if 'thompson_hit' in acq_func_name:
                acq_kwargs['threshold'] = hit_threshold
            if 'thompson_ig' in acq_func_name:
                acq_kwargs['threshold'] = hit_threshold
                acq_kwargs['num_cycles'] = self.num_active_learning_cycles
                acq_kwargs['current_cycle'] = cycle_index + 1  # 1-indexed
            if 'greedy_kg' in acq_func_name:
                acq_kwargs['H'] = greedy_kg_H
            if 'probability_hit' in acq_func_name:
                acq_kwargs['threshold'] = hit_threshold
                acq_kwargs['select_size'] = self.acquisition_batch_size
                # Remove keys not accepted by ProbabilityHitAcquisition.__call__
                allowed_keys = {'dataset_x', 'available_indices', 'model', 'dataset_y', 'threshold', 'select_size'}
                filtered_acq_kwargs = {k: v for k, v in acq_kwargs.items() if k in allowed_keys}
                last_selected_indices = self.acquisition_function(**filtered_acq_kwargs)
            else:
                last_selected_indices = self.acquisition_function(**acq_kwargs)
            cumulative_indices.extend(last_selected_indices)
            cumulative_indices = sorted(list(set(cumulative_indices)))
            assert len(last_selected_indices) == self.acquisition_batch_size
            last_batch_precision_topk = self.get_precision_topk_last_selected_indices(
                last_selected_indices=last_selected_indices,
                dataset_y=dataset_y,
                topk_percent=self.topk_percent,
            )
            precision_topk, recall_topk = self.get_cumulative_precision_recall_topk(
                cumulative_indices=cumulative_indices,
                dataset_y=dataset_y,
                topk_percent=self.topk_percent,
            )
            # Calculate the absolute number of top-k recovered
            num_topk_recovered = len(set(self.topk_indices) & set(cumulative_indices))
            cumulative_precision_topk.append(precision_topk)
            cumulative_recall_topk.append(recall_topk)
            cumulative_num_topk_recovered.append(num_topk_recovered)
            proportion_top_clusters_recovered = (
                self.get_cumulative_proportion_top_clusters_recovered(
                    cumulative_indices
                )
            )
            cumulative_proportion_top_clusters_recovered.append(
                proportion_top_clusters_recovered
            )

        # Write performance data ONCE after all cycles complete (one row per method/seed)
        # Ensure the directory for performance file exists
        performance_dir = os.path.dirname(self.performance_file_location)
        if performance_dir and not os.path.exists(performance_dir):
            os.makedirs(performance_dir, exist_ok=True)
        
        performance_file_exists = os.path.exists(self.performance_file_location)
        with open(self.performance_file_location, "a") as performance_record:
            if not performance_file_exists:
                header = "dataset_name,feature_set_name,topk_percent,model_name,acquisition_function_name,acquisition_batch_size,num_active_learning_cycles,seed,num_total_items,num_topk_items,num_topk_clusters,final_num_topk_recovered,cumulative_num_topk_recovered,cumulative_precision_topk,cumulative_recall_topk,cumulative_proportion_top_clusters_recovered,bax_topk_kvalue,bax_subset_select_subset_size,bax_level_set_c,bax_batch_selection_mode,bax_noise_type,bax_noise_lengthscale,bax_noise_outputscale,bax_num_samples_EIG,bax_num_samples_entropy,bax_entropy_average_mode"
                performance_record.write(header + "\n")
            record = ",".join(
                [
                    str(x)
                    for x in [
                        self.dataset_name,
                        self.feature_set_name,
                        self.topk_percent,
                        self.model_name,
                        self.acquisition_function_name,
                        self.acquisition_batch_size,
                        self.num_active_learning_cycles,
                        self.seed,
                        len(dataset_y),
                        len(self.topk_indices),
                        len(self.topk_clusters_to_item.keys()),
                        cumulative_num_topk_recovered[-1],  # Final value
                        '"' + str(cumulative_num_topk_recovered) + '"',
                        '"' + str(cumulative_precision_topk) + '"',
                        '"' + str(cumulative_recall_topk) + '"',
                        '"'
                        + str(cumulative_proportion_top_clusters_recovered)
                        + '"',
                        self.bax_topk_kvalue,
                        self.bax_subset_select_subset_size,
                        self.bax_level_set_c,
                        self.bax_batch_selection_mode,
                        self.bax_noise_type,
                        self.bax_noise_lengthscale,
                        self.bax_noise_outputscale,
                        self.bax_num_samples_EIG,
                        self.bax_num_samples_entropy,
                        self.bax_entropy_average_mode,
                    ]
                ]
            )
            performance_record.write(record + "\n")

        print("Cumulative number of top-k recovered: {}".format(cumulative_num_topk_recovered))
        print("Cumulative precision topk (ratio): {}".format(cumulative_precision_topk))
        print("Cumulative recall topk: {}".format(cumulative_recall_topk))
        print(
            "cumulative proportion top clusters recovered: {}".format(
                cumulative_proportion_top_clusters_recovered
            )
        )
        results_path = os.path.join(self.output_directory, "results.pkl")
        with open(results_path, "wb") as fp:
            pkl.dump(result_records, fp)
        return None


class GeneDiscoRunner:
    """Small wrapper to support multiple acquisition methods and seeds.

    This mirrors the ToyExperimentRunner used in the toy experiments: it parses
    command-line arguments for acquisition function name(s) and seed(s), then
    instantiates and runs a separate GeneDiscoLoop for each combination.

    Examples
    --------
    Single run (backwards compatible):
        python -m discobax.apps.genedisco_experiment \
            --dataset_name schmidt_2021_ifng \
            --feature_set_name achilles \
            --acquisition_function_name thompson_sampling \
            --seed 42

    Multiple methods:
        python -m discobax.apps.genedisco_experiment \
            --dataset_name schmidt_2021_ifng \
            --feature_set_name achilles \
            --acquisition_function_name "random,thompson_sampling,top_k" \
            --seed 42

    Multiple seeds:
        python -m discobax.apps.genedisco_experiment \
            --dataset_name schmidt_2021_ifng \
            --feature_set_name achilles \
            --acquisition_function_name thompson_sampling \
            --seed "1,2,3"
    """

    def __init__(self, argv: List[str]):
        self.argv = argv
        self.raw_methods_arg: Optional[str] = None
        self.raw_seeds_vals: List[str] = []

    def _parse_cli(self) -> List[str]:
        """Parse acquisition function names from CLI.

        Supports any of the following:
        - multiple flags: --acquisition_function_name m1 --acquisition_function_name m2
        - comma-separated: --acquisition_function_name "m1,m2,m3"
        - Python-style list: --acquisition_function_name "['m1','m2']"
        """
        all_methods: List[str] = []
        for i, tok in enumerate(self.argv):
            if tok == "--acquisition_function_name" and i + 1 < len(self.argv):
                val = self.argv[i + 1]
                all_methods.append(val)
                if self.raw_methods_arg is None:
                    self.raw_methods_arg = val
            if tok == "--seed" and i + 1 < len(self.argv):
                self.raw_seeds_vals.append(self.argv[i + 1])

        # Resolve methods
        methods: List[str] = []
        if len(all_methods) > 1:
            methods = [str(m).strip() for m in all_methods]
        else:
            if isinstance(self.raw_methods_arg, str):
                txt = self.raw_methods_arg.strip()
                parsed = None
                try:
                    if (
                        (txt.startswith("[") and txt.endswith("]"))
                        or (txt.startswith("(") and txt.endswith(")"))
                    ):
                        parsed = ast.literal_eval(txt)
                except Exception:
                    parsed = None
                if parsed is None:
                    parts = [p.strip() for p in txt.split(",") if p.strip()]
                    methods = parts if len(parts) > 1 else [txt]
                else:
                    methods = [str(m).strip() for m in parsed]

        if not methods:
            # Fall back to single method; GeneDiscoLoop will pick default.
            methods = []
        return methods

    def _parse_seeds(self) -> List[int]:
        """Parse seed values from CLI similar to ToyExperimentRunner."""
        seeds_list: List[int] = []
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
                if (
                    (txt.startswith("[") and txt.endswith("]"))
                    or (txt.startswith("(") and txt.endswith(")"))
                ):
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

    @staticmethod
    def _adjust_argv_for_base(argv: List[str]) -> List[str]:
        """Return a copy of argv suitable for instantiating a *single* GeneDiscoLoop.

        When multiple --acquisition_function_name flags or list-like values are
        provided, we keep just one placeholder here; the specific acquisition
        function will be supplied via kwargs when instantiating GeneDiscoLoop.
        """
        new_argv: List[str] = []
        saw_acq = False
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok == "--acquisition_function_name" and i + 1 < len(argv):
                if not saw_acq:
                    new_argv.extend([tok, "random"])
                    saw_acq = True
                i += 2
                continue
            else:
                new_argv.append(tok)
                i += 1
        return new_argv

    def run(self) -> None:
        methods = self._parse_cli()
        seeds_list = self._parse_seeds()

        # Instantiate a base config once from sanitized argv
        base_argv = self._adjust_argv_for_base(self.argv)
        sys.argv = base_argv
        base: GeneDiscoLoop = sp.instantiate_from_command_line(GeneDiscoLoop)

        methods_iter = methods if methods else [base.acquisition_function_name]
        seeds_iter = seeds_list if seeds_list else [base.seed]
        
        # Clear the performance file at the start of a new run to avoid mixing old and new results
        if os.path.exists(base.performance_file_location):
            print(f"⚠️  Removing existing performance file: {base.performance_file_location}")
            print(f"   (This prevents mixing results from different runs)")
            os.remove(base.performance_file_location)

        for m in methods_iter:
            for s in seeds_iter:
                try:
                    info(
                        f"Running GeneDiscoLoop with acquisition_function_name={m}, seed={s}"
                    )
                except Exception:
                    print(
                        f"Running GeneDiscoLoop with acquisition_function_name={m}, seed={s}"
                    )
                loop = GeneDiscoLoop(
                    cache_directory=base.cache_directory,
                    output_directory=base.output_directory,
                    dataset_name=base.dataset_name,
                    feature_set_name=base.feature_set_name,
                    model_name=base.model_name,
                    acquisition_function_name=m,
                    acquisition_function_path=base.acquisition_function_path,
                    acquisition_batch_size=base.acquisition_batch_size,
                    num_active_learning_cycles=base.num_active_learning_cycles,
                    test_ratio=base.test_ratio,
                    hyperopt_children=base.hyperopt_children,
                    schedule_on_slurm=base.schedule_on_slurm,
                    schedule_children_on_slurm=base.schedule_children_on_slurm,
                    remote_execution_time_limit_days=base.remote_execution_time_limit_days,
                    remote_execution_mem_limit_in_mb=base.remote_execution_mem_limit_in_mb,
                    remote_execution_virtualenv_path=base.remote_execution_virtualenv_path,
                    remote_execution_num_cpus=base.remote_execution_num_cpus,
                    seed=s,
                    topk_percent=base.topk_percent,
                    hit_threshold=base.hit_threshold,  # Pass manual threshold
                    greedy_kg_H=base.greedy_kg_H,  # Pass manual H
                    performance_file_location=base.performance_file_location,
                    bax_topk_kvalue=base.bax_topk_kvalue,
                    bax_level_set_c=base.bax_level_set_c,
                    bax_subset_select_subset_size=base.bax_subset_select_subset_size,
                    bax_noise_type=base.bax_noise_type,
                    bax_noise_lengthscale=base.bax_noise_lengthscale,
                    bax_noise_outputscale=base.bax_noise_outputscale,
                    bax_num_samples_EIG=base.bax_num_samples_EIG,
                    bax_num_samples_entropy=base.bax_num_samples_entropy,
                    bax_entropy_average_mode=base.bax_entropy_average_mode,
                    bax_batch_selection_mode=base.bax_batch_selection_mode,
                    num_topk_clusters=base.num_topk_clusters,
                )
                try:
                    loop.run()
                except Exception as e:
                    import traceback
                    warn(
                        f"GeneDiscoLoop run failed for acquisition_function_name={m}, seed={s}: {e}"
                    )
                    warn("Full traceback:")
                    traceback.print_exc()



def main():
    runner = GeneDiscoRunner(sys.argv[:])
    runner.run()


if __name__ == "__main__":
    main()
