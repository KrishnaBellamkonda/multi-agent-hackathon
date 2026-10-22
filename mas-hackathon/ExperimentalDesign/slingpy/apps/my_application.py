
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np
import slingpy as sp
import torch.nn as nn
from sklearn.datasets import load_iris
from typing import AnyStr, Dict, List, Optional


class MLP(nn.Module, sp.ArgumentDictionary):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2, num_units: int = 128,
                 with_bn: bool = False, dropout: float = 0.0):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_units = num_units
        self.with_bn = with_bn
        self.dropout = dropout
        self.sequential = self.make_feedforward(input_dim, output_dim, num_layers, num_units, with_bn, dropout)

    def make_feedforward(self, input_dim: int, output_dim: int, num_layers: int = 2, num_units: int = 128,
                         with_bn: bool = False, dropout: float = 0.0,
                         activation: nn.Module = nn.SELU,
                         output_activation: nn.Module = nn.Softmax):
        layers, last_dim = [], input_dim
        for _ in range(num_layers):
            dense = nn.Linear(last_dim, num_units, bias=not with_bn)
            act = activation()
            last_dim = num_units
            layers.append(dense)
            if with_bn:
                bn = nn.BatchNorm1d(num_units)
                layers.append(bn)
            if not np.isclose(dropout, 0.0, atol=0, rtol=0):
                dp = nn.Dropout(dropout)
                layers.append(dp)
            layers.append(act)

        dense = nn.Linear(last_dim, output_dim)
        act = output_activation()
        layers.append(dense)
        layers.append(act)
        model = nn.Sequential(*layers)
        return model

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        x = self.sequential(x[0].float())
        return [x]


class CustomLoss(sp.TorchLoss):
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, y_pred: List[torch.Tensor], y_true: List[torch.Tensor]) -> torch.Tensor:
        loss = self.loss_fn(y_pred[0], y_true[0][:, 0].long())
        return loss


class MyApplication(sp.AbstractBaseApplication):
    def __init__(self, output_directory: AnyStr = "",
                 schedule_on_slurm: bool = False,
                 split_index_outer: int = 1,
                 split_index_inner: int = 1,
                 num_splits_outer: int = 5,
                 num_splits_inner: int = 5):
        super(MyApplication, self).__init__(
            output_directory=output_directory,
            save_predictions=True,
            schedule_on_slurm=schedule_on_slurm,
            split_index_outer=split_index_outer,
            split_index_inner=split_index_inner,
            num_splits_outer=num_splits_outer,
            num_splits_inner=num_splits_inner
        )

    def get_metrics(self, set_name: AnyStr) -> List[sp.AbstractMetric]:
        return [
            sp.metrics.AreaUnderTheCurve()
        ]

    def load_data(self) -> Dict[AnyStr, sp.AbstractDataSource]:
        iris = load_iris()
        x, y = iris['data'], iris['target'][:, np.newaxis]
        feature_names = iris['feature_names']

        h5_file_x = os.path.join(self.output_directory, "dataset_x.h5")
        h5_file_y = os.path.join(self.output_directory, "dataset_y.h5")
        sp.HDF5Tools.save_h5_file(h5_file_x, x, "dataset_x", column_names=feature_names)
        sp.HDF5Tools.save_h5_file(h5_file_y, y, "dataset_y")

        data_source_x = sp.HDF5DataSource(h5_file_x)
        data_source_y = sp.HDF5DataSource(h5_file_y)

        stratifier = sp.StratifiedSplit()
        rest_indices, test_indices = stratifier.split(data_source_y,
                                                      split_index=self.split_index_outer,
                                                      num_splits=self.num_splits_outer)
        validation_indices, training_indices = stratifier.split(data_source_y.subset(rest_indices),
                                                                split_index=self.split_index_inner,
                                                                num_splits=self.num_splits_inner)

        return {
            "training_set_x": data_source_x.subset(training_indices),
            "training_set_y": data_source_y.subset(training_indices),
            "validation_set_x": data_source_x.subset(validation_indices),
            "validation_set_y": data_source_y.subset(validation_indices),
            "test_set_x": data_source_x.subset(test_indices),
            "test_set_y": data_source_y.subset(test_indices)
        }

    def get_model(self) -> sp.AbstractBaseModel:
        num_features = self.datasets.training_set_x.get_shape()[0][-1]
        num_outputs = len(np.unique(self.datasets.training_set_y.get_data()))

        model = sp.TorchModel(MLP(num_features, num_outputs), loss=CustomLoss(), num_epochs=1,
                              target_transformer=sp.ScaleTransform([torch.Tensor([0.0])], [torch.Tensor([1.0])]))
        return model

    def train_model(self, model: sp.AbstractBaseModel) -> Optional[sp.AbstractBaseModel]:
        model.fit(self.datasets.training_set_x, self.datasets.training_set_y)
        return model


if __name__ == "__main__":
    app = MyApplication.instantiate_from_command_line(MyApplication)
    app.run()
