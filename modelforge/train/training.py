"""
This module contains classes and functions for training neural network potentials using PyTorch Lightning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import lightning.pytorch as pL
import torch
import torchmetrics
from lightning import Trainer
from loguru import logger as log
from openff.units import unit
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from modelforge.dataset.dataset import (
    BatchData,
    DataModule,
    DatasetParameters,
    NNPInput,
)
from modelforge.potential.parameters import (
    ANI2xParameters,
    PaiNNParameters,
    PhysNetParameters,
    SAKEParameters,
    SchNetParameters,
    TensorNetParameters,
)
from modelforge.train.parameters import RuntimeParameters, TrainingParameters

__all__ = [
    "Error",
    "FromPerAtomToPerMoleculeSquaredError",
    "Loss",
    "LossFactory",
    "PerMoleculeSquaredError",
    "ModelTrainer",
    "create_error_metrics",
    "ModelTrainer",
]


class Error(nn.Module, ABC):
    """
    Class representing the error calculation for predicted and true values.
    """

    def __init__(self, scale_by_number_of_atoms: bool = True):

        super().__init__()
        if not scale_by_number_of_atoms:
            # If scaling is not desired, override the method to just return the input error unchanged
            self.scale_by_number_of_atoms = (
                lambda error, atomic_subsystem_counts, prefactor=1: error
            )

    @abstractmethod
    def calculate_error(
        self, predicted: torch.Tensor, true: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the error between the predicted and true values
        """
        raise NotImplementedError

    @staticmethod
    def calculate_squared_error(
        predicted_tensor: torch.Tensor, reference_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the squared error between the predicted and true values.

        Parameters
        ----------
        predicted_tensor : torch.Tensor
            The predicted values.
        reference_tensor : torch.Tensor
            The reference values provided by the dataset.

        Returns
        -------
        torch.Tensor
            The calculated error.
        """
        squared_diff = (predicted_tensor - reference_tensor).pow(2)
        error = squared_diff.sum(dim=1, keepdim=True)
        return error

    @staticmethod
    def scale_by_number_of_atoms(
        error, atomic_subsystem_counts, prefactor: int = 1
    ) -> torch.Tensor:
        """
        Scales the error by the number of atoms in the atomic subsystems.

        Parameters
        ----------
        error : torch.Tensor
            The error to be scaled.
        atomic_subsystem_counts : torch.Tensor
            The number of atoms in the atomic subsystems.
        prefactor : int
           To consider the shape of the property, e.g., if the reference property has shape (N,3) it is necessary to further devide the result by 3
        Returns
        -------
        torch.Tensor
            The scaled error.
        """
        # divide by number of atoms
        scaled_by_number_of_atoms = error / (
            prefactor * atomic_subsystem_counts.unsqueeze(1)
        )  # FIXME: ensure that all per-atom properties have dimension (N, 1)
        return scaled_by_number_of_atoms


class FromPerAtomToPerMoleculeSquaredError(Error):
    """
    Calculates the per-atom error and aggregates it to per-molecule mean squared error.
    """

    def __init__(self, scale_by_number_of_atoms: bool = True):
        super().__init__(scale_by_number_of_atoms)

    def calculate_error(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the per-atom error.
        """
        return self.calculate_squared_error(per_atom_prediction, per_atom_reference)

    def forward(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
        batch: "NNPInput",
    ) -> torch.Tensor:
        """
        Computes the per-atom error and aggregates it to per-molecule mean squared error.

        Parameters
        ----------
        per_atom_prediction : torch.Tensor
            The predicted values.
        per_atom_reference : torch.Tensor
            The reference values provided by the dataset.
        batch : NNPInput
            The batch data containing metadata and input information.

        Returns
        -------
        torch.Tensor
            The aggregated per-molecule error.
        """

        # squared error
        per_atom_squared_error = self.calculate_error(
            per_atom_prediction, per_atom_reference
        )

        per_molecule_squared_error = torch.zeros_like(
            batch.metadata.E, dtype=per_atom_squared_error.dtype
        )
        # Aggregate error per molecule

        per_molecule_squared_error.scatter_add_(
            0,
            batch.nnp_input.atomic_subsystem_indices.long().unsqueeze(1),
            per_atom_squared_error,
        ).contiguous()
        # divide by number of atoms
        per_molecule_square_error_scaled = self.scale_by_number_of_atoms(
            per_molecule_squared_error,
            batch.metadata.atomic_subsystem_counts,
            prefactor=per_atom_prediction.shape[-1],
        ).contiguous()

        return per_molecule_square_error_scaled


class PerMoleculeSquaredError(Error):
    """
    Calculates the per-molecule mean squared error.
    """

    def __init__(self, scale_by_number_of_atoms: bool = True):
        super().__init__(scale_by_number_of_atoms)

    def forward(
        self,
        per_molecule_prediction: torch.Tensor,
        per_molecule_reference: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """
        Computes the per-molecule mean squared error.

        Parameters
        ----------
        per_molecule_prediction : torch.Tensor
            The predicted values.
        per_molecule_reference : torch.Tensor
            The true values.
        batch : Any
            The batch data containing metadata and input information.

        Returns
        -------
        torch.Tensor
            The mean per-molecule error.
        """

        per_molecule_squared_error = self.calculate_error(
            per_molecule_prediction, per_molecule_reference
        )
        per_molecule_square_error_scaled = self.scale_by_number_of_atoms(
            per_molecule_squared_error,
            batch.metadata.atomic_subsystem_counts,
        )

        return per_molecule_square_error_scaled

    def calculate_error(
        self,
        per_atom_prediction: torch.Tensor,
        per_atom_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the per-atom error.
        """
        return self.calculate_squared_error(per_atom_prediction, per_atom_reference)


class Loss(nn.Module):

    _SUPPORTED_PROPERTIES = ["per_atom_energy", "per_molecule_energy", "per_atom_force"]

    def __init__(self, loss_porperty: List[str], weight: Dict[str, float]):
        """
        Calculates the combined loss for energy and force predictions.

        Parameters
        ----------
        loss_property : List[str]
            List of properties to include in the loss calculation.
        weight : Dict[str, float]
            Dictionary containing the weights for each property in the loss calculation.

        Raises
        ------
        NotImplementedError
            If an unsupported loss type is specified.
        """
        super().__init__()
        from torch.nn import ModuleDict

        self.loss_property = loss_porperty
        self.weight = weight

        self.loss = ModuleDict()

        for prop, w in weight.items():
            if prop in self._SUPPORTED_PROPERTIES:
                if prop == "per_atom_force":
                    self.loss[prop] = FromPerAtomToPerMoleculeSquaredError(
                        scale_by_number_of_atoms=True
                    )
                elif prop == "per_atom_energy":
                    self.loss[prop] = PerMoleculeSquaredError(
                        scale_by_number_of_atoms=True
                    )  # FIXME: this is currently not working
                elif prop == "per_molecule_energy":
                    self.loss[prop] = PerMoleculeSquaredError(
                        scale_by_number_of_atoms=True
                    )
                self.register_buffer(prop, torch.tensor(w))
            else:
                raise NotImplementedError(f"Loss type {prop} not implemented.")

    def forward(
        self, predict_target: Dict[str, torch.Tensor], batch
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the combined loss for the specified properties.

        Parameters
        ----------
        predict_target : Dict[str, torch.Tensor]
            Dictionary containing predicted and true values for energy and per_atom_force.
        batch : Any
            The batch data containing metadata and input information.

        Returns
        -------
        Dict{str, torch.Tensor]
            Individual loss terms and the combined, total loss.
        """
        # save the loss as a dictionary
        loss_dict = {}
        # accumulate loss
        loss = torch.tensor(
            [0.0], dtype=batch.metadata.E.dtype, device=batch.metadata.E.device
        )
        # iterate over loss properties
        for prop in self.loss_property:
            # calculate loss per property
            loss_ = self.loss[prop](
                predict_target[f"{prop}_predict"], predict_target[f"{prop}_true"], batch
            )
            # add total loss
            loss = loss + (self.weight[prop] * loss_)
            # save loss
            loss_dict[f"{prop}"] = loss_

        # add total loss to results dict and return
        loss_dict["total_loss"] = loss

        return loss_dict


class LossFactory(object):
    """
    Factory class to create different types of loss functions.
    """

    @staticmethod
    def create_loss(loss_property: List[str], weight: Dict[str, float]) -> Type[Loss]:
        """
        Creates an instance of the specified loss type.

        Parameters
        ----------
        loss_property : List[str]
            List of properties to include in the loss calculation.
        weight : Dict[str, float]
            Dictionary containing the weights for each property in the loss calculation.
        Returns
        -------
        Loss
            An instance of the specified loss function.
        """

        return Loss(loss_property, weight)


from torch.nn import ModuleDict
from torch.optim import Optimizer


def create_error_metrics(loss_properties: List[str], loss: bool = False) -> ModuleDict:
    """
    Creates a ModuleDict of MetricCollections for the given loss properties.

    Parameters
    ----------
    loss_properties : List[str]
        List of loss properties for which to create the metrics.
    loss : bool, optional
        If True, only the loss metric is created, by default False.
    Returns
    -------
    ModuleDict
        A dictionary where keys are loss properties and values are MetricCollections.
    """
    from torchmetrics import MetricCollection
    from torchmetrics.aggregation import MeanMetric
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

    if loss:
        metric_dict = ModuleDict(
            {prop: MetricCollection([MeanMetric()]) for prop in loss_properties}
        )
        metric_dict["total_loss"] = MetricCollection([MeanMetric()])
        return metric_dict
    else:
        metric_dict = ModuleDict(
            {
                prop: MetricCollection(
                    [MeanAbsoluteError(), MeanSquaredError(squared=False)]
                )
                for prop in loss_properties
            }
        )
        return metric_dict


class CalculateProperties(torch.nn.Module):

    def __init__(self, requested_properties: List[str]):
        """
        A utility class for calculating properties such as energies and forces from batches using a neural network model.

        Parameters

        """
        super().__init__()
        self.requested_properties = requested_properties
        self.include_force = False
        if "per_atom_force" in self.requested_properties:
            self.include_force = True

    def _get_forces(
        self, batch: "BatchData", energies: Dict[str, torch.Tensor], train_mode: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.
        energies : Dict[str, torch.Tensor]
            A dictionary containing the predicted energies from the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true forces from the dataset and the predicted forces by the model.
        """
        nnp_input = batch.nnp_input
        per_atom_force_true = batch.metadata.F.to(torch.float32)

        if per_atom_force_true.numel() < 1:
            raise RuntimeError("No force can be calculated.")

        per_molecule_energy_predict = energies["per_molecule_energy_predict"]

        # Ensure E_predict and nnp_input.positions require gradients and are on the same device
        if not per_molecule_energy_predict.requires_grad:
            per_molecule_energy_predict.requires_grad = True
        if not nnp_input.positions.requires_grad:
            nnp_input.positions.requires_grad = True

        # Compute the gradient (forces) from the predicted energies
        grad = torch.autograd.grad(
            per_molecule_energy_predict,
            nnp_input.positions,
            grad_outputs=torch.ones_like(per_molecule_energy_predict),
            create_graph=train_mode,
            retain_graph=train_mode,
            allow_unused=True,
        )[0]

        if grad is None:
            raise RuntimeWarning("Force calculation did not return a gradient")

        per_atom_force_predict = -1 * grad  # Forces are the negative gradient of energy

        return {
            "per_atom_force_true": per_atom_force_true,
            "per_atom_force_predict": per_atom_force_predict.contiguous(),
        }

    def _get_energies(
        self, batch: "BatchData", model: Type[torch.nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the energies from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.
        model : Type[torch.nn.Module]
            The neural network model used to compute the energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true energies from the dataset and the predicted energies by the model.
        """
        nnp_input = batch.nnp_input
        per_molecule_energy_true = batch.metadata.E.to(torch.float32)
        per_molecule_energy_predict = model.forward(nnp_input)[
            "per_molecule_energy"
        ].unsqueeze(
            1
        )  # FIXME: ensure that all per-molecule properties have dimension (N, 1)
        assert per_molecule_energy_true.shape == per_molecule_energy_predict.shape, (
            f"Shapes of true and predicted energies do not match: "
            f"{per_molecule_energy_true.shape} != {per_molecule_energy_predict.shape}"
        )
        return {
            "per_molecule_energy_true": per_molecule_energy_true,
            "per_molecule_energy_predict": per_molecule_energy_predict,
        }

    def forward(
        self, batch: "BatchData", model: Type[torch.nn.Module], train_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the energies and forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.
        model : Type[torch.nn.Module]
            The neural network model used to compute the properties.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true and predicted energies and forces from the dataset and the model.
        """
        energies = self._get_energies(batch, model)
        if self.include_force:
            forces = self._get_forces(batch, energies, train_mode)
        else:
            forces = {}
        return {**energies, **forces}


class TrainingAdapter(pL.LightningModule):

    def __init__(
        self,
        *,
        potential_parameter: Union[
            ANI2xParameters,
            SAKEParameters,
            SchNetParameters,
            PhysNetParameters,
            PaiNNParameters,
            TensorNetParameters,
        ],
        dataset_statistic: Dict[str, Dict[str, unit.Quantity]],
        training_parameter: TrainingParameters,
        potential_seed: Optional[int] = None,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        potential_parameter : Union[ANI2xParameters, SAKEParameters, SchNetParameters, PhysNetParameters, PaiNNParameters, TensorNetParameters]
            Parameters for the potential model.
        dataset_statistic : Dict[str, float]
            The statistics of the dataset, such as mean and standard deviation.
        training_parameter : TrainingParameters
            Parameters for the training process.
        potential_seed : Optional[int], optional
            The seed to use for initializing the model, by default None.
        """
        from modelforge.potential.models import setup_potential

        super().__init__()
        self.save_hyperparameters()
        self.training_parameter = training_parameter

        self.potential = setup_potential(
            potential_parameter=potential_parameter,
            dataset_statistic=dataset_statistic,
            potential_seed=potential_seed,
            jit=False,
            use_training_mode_neighborlist=True,
        )

        def check_strides(module, grad_input, grad_output):
            print(f"Layer: {module.__class__.__name__}")
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    print(
                        f"Grad input {i}: size {grad.size()}, strides {grad.stride()}"
                    )
            # Handle grad_output
            if isinstance(grad_output, tuple) and isinstance(grad_output[0], dict):
                # If the output is a dict wrapped in a tuple, extract the dict
                grad_output = grad_output[0]
            if isinstance(grad_output, dict):
                for key, grad in grad_output.items():
                    if grad is not None:
                        print(
                            f"Grad output [{key}]: size {grad.size()}, strides {grad.stride()}"
                        )
            else:
                for i, grad in enumerate(grad_output):
                    if grad is not None:
                        print(
                            f"Grad output {i}: size {grad.size()}, strides {grad.stride()}"
                        )

        # Register the full backward hook
        if training_parameter.verbose is True:
            for module in self.potential.modules():
                module.register_full_backward_hook(check_strides)

        self.include_force = False
        if "per_atom_force" in training_parameter.loss_parameter.loss_property:
            self.include_force = True

        self.calculate_predictions = CalculateProperties(
            training_parameter.loss_parameter.loss_property
        )
        self.optimizer = training_parameter.optimizer
        self.learning_rate = training_parameter.lr
        self.lr_scheduler = training_parameter.lr_scheduler

        # verbose output, only True if requested
        if training_parameter.verbose:
            self.log_histograms = True
            self.log_on_training_step = True
        else:
            self.log_histograms = False
            self.log_on_training_step = False

        # initialize loss
        self.loss = LossFactory.create_loss(
            **training_parameter.loss_parameter.model_dump()
        )

        # Assign the created error metrics to the respective attributes
        self.test_error = create_error_metrics(
            training_parameter.loss_parameter.loss_property
        )
        self.val_error = create_error_metrics(
            training_parameter.loss_parameter.loss_property
        )
        self.train_error = create_error_metrics(
            training_parameter.loss_parameter.loss_property
        )

        # Initialize loss  metric
        self.loss_metric = create_error_metrics(
            training_parameter.loss_parameter.loss_property, loss=True
        )

    def forward(self, batch: "BatchData") -> Dict[str, torch.Tensor]:
        """
        Computes the energies and forces from a given batch using the model.

        Parameters
        ----------
        batch : BatchData
            A single batch of data, including input features and target energies.

        Returns
        -------
        Dict[str, torch.Tensor]
            The true and predicted energies and forces from the dataset and the model.
        """
        return self.potential(batch)

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.potential, "_config_prior"):
            return self.potential._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    def _update_metrics(
        self,
        error_dict: Dict[str, torchmetrics.MetricCollection],
        predict_target: Dict[str, torch.Tensor],
    ):
        """
        Updates the provided metric collections with the predicted and true targets.

        Parameters
        ----------
        error_dict : Dict[str, torchmetrics.MetricCollection]
            Dictionary containing metric collections for energy and force.
        predict_target : Dict[str, torch.Tensor]
            Dictionary containing predicted and true values for energy and force.
        """

        for property, metrics in error_dict.items():
            for _, error_log in metrics.items():
                error_log(
                    predict_target[f"{property}_predict"].detach(),
                    predict_target[f"{property}_true"].detach(),
                )

    def training_step(self, batch: "BatchData", batch_idx: int) -> torch.Tensor:
        """
        Training step to compute the MSE loss for a given batch.

        Parameters
        ----------
        batch : BatchData
            The batch of data provided for the training.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The loss tensor computed for the current training step.
        """

        # calculate energy and forces, Note that `predict_target` is a
        # dictionary containing the predicted and true values for energy and
        # force`
        predict_target = self.calculate_predictions(
            batch, self.potential, self.training
        )

        # Calculate the loss
        loss_dict = self.loss(predict_target, batch)

        # Update the loss metric with the different loss components
        for key, metric in loss_dict.items():
            self.loss_metric[key].update(metric.clone().detach(), batch.batch_size())

        loss = torch.mean(loss_dict["total_loss"])
        return loss.contiguous()

    def validation_step(self, batch: "BatchData", batch_idx: int) -> None:
        """
        Validation step to compute the RMSE/MAE across epochs.

        Parameters
        ----------
        batch : BatchData
            The batch of data provided for validation.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        None
        """

        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        with torch.inference_mode(False):

            # calculate energy and forces
            predict_target = self.calculate_predictions(
                batch, self.potential, self.training
            )

        self._update_metrics(self.val_error, predict_target)

    def test_step(self, batch: "BatchData", batch_idx: int) -> None:
        """
        Test step to compute the RMSE loss for a given batch.

        This method is called automatically during the test loop of the training process. It computes
        the loss on a batch of test data and logs the results for analysis.

        Parameters
        ----------
        batch : BatchData
            The batch of data to test the model on.
        batch_idx : int
            The index of the batch within the test dataset.

        Returns
        -------
        None
            The results are logged and not directly returned.
        """
        # Ensure positions require gradients for force calculation
        batch.nnp_input.positions.requires_grad_(True)
        # calculate energy and forces
        with torch.inference_mode(False):
            predict_target = self.calculate_predictions(
                batch, self.potential, self.training
            )
        # Update and log metrics
        self._update_metrics(self.test_error, predict_target)

    def on_test_epoch_end(self):
        """
        Operations to perform at the end of the test set pass.

        This method is automatically called by PyTorch Lightning at the end of
        the test epoch. It logs the accumulated metrics for the test phase.
        """
        self._log_on_epoch(log_mode="test")

    def on_train_epoch_end(self):
        """
        Operations to perform at the end of each training epoch.

        This method is automatically called by PyTorch Lightning at the end of
        each training epoch. It logs histograms of weights and biases, learning
        rate, and resets validation loss.
        """
        if self.log_histograms == True:
            for name, params in self.named_parameters():
                if params is not None:
                    self.logger.experiment.add_histogram(
                        name, params, self.current_epoch
                    )
                if params.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"{name}.grad", params.grad, self.current_epoch
                    )

        sch = self.lr_schedulers()
        try:
            self.log("lr", sch.get_last_lr()[0], on_epoch=True, prog_bar=True)
        except AttributeError:
            pass

        self._log_on_epoch()

    def _log_on_epoch(self, log_mode: str = "train"):
        """
        Logs all accumulated metrics at the end of an epoch.

        Parameters
        ----------
        log_mode : str, optional
            The phase of training for which metrics are being logged, by default "train".
            It can be "train", "val", or "test".
        """
        # convert long names to shorter versions
        conv = {
            "MeanAbsoluteError": "mae",
            "MeanSquaredError": "rmse",
            "MeanMetric": "mse",  # NOTE: MeanMetric is the MSE since we accumulate the squared error
        }  # NOTE: MeanSquaredError(squared=False) is RMSE

        # Log all accumulated metrics for train and val phases
        if log_mode == "train":
            errors = [
                ("train", self.train_error),
                ("val", self.val_error),
                ("loss", self.loss_metric),
            ]
        elif log_mode == "test":
            errors = [
                ("test", self.test_error),
            ]
        else:
            raise RuntimeError(f"Unrecognized mode: {log_mode}")

        for phase, error_dict in errors:
            # skip if log_on_training_step is not requested
            if phase == "train" and not self.log_on_training_step:
                continue

            for property, metrics_dict in error_dict.items():
                for name, metric in metrics_dict.items():
                    name = f"{phase}/{property}/{conv.get(name, name)}"
                    self.log(name, metric.compute(), prog_bar=True, sync_dist=True)
                    metric.reset()

    def configure_optimizers(self):
        """
        Configures the model's optimizers (and optionally schedulers).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the optimizer and optionally the learning rate scheduler
            to be used within the PyTorch Lightning training process.
        """

        optimizer = self.optimizer(self.potential.parameters(), lr=self.learning_rate)

        lr_scheduler = self.lr_scheduler.model_dump().copy()
        interval = lr_scheduler.pop("interval")
        frequency = lr_scheduler.pop("frequency")
        monitor = lr_scheduler.pop("monitor")

        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            **lr_scheduler,
        )

        lr_scheduler = {
            "scheduler": lr_scheduler,
            "monitor": monitor,  # Name of the metric to monitor
            "interval": interval,
            "frequency": frequency,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


from openff.units import unit


class ModelTrainer:
    """
    Class for training neural network potentials using PyTorch Lightning.
    """

    def __init__(
        self,
        *,
        dataset_parameter: DatasetParameters,
        potential_parameter: Union[
            ANI2xParameters,
            SAKEParameters,
            SchNetParameters,
            PhysNetParameters,
            PaiNNParameters,
            TensorNetParameters,
        ],
        training_parameter: TrainingParameters,
        runtime_parameter: RuntimeParameters,
        dataset_statistic: Dict[str, Dict[str, unit.Quantity]],
        use_default_dataset_statistic: bool,
        optimizer: Type[Optimizer] = torch.optim.AdamW,
        potential_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initializes the TrainingAdapter with the specified model and training configuration.

        Parameters
        ----------
        dataset_config : DatasetParameters
            Parameters for the dataset.
        potential_parameter : Union[ANI2xParameters, SAKEParameters, SchNetParameters, PhysNetParameters, PaiNNParameters, TensorNetParameters]
            Parameters for the potential model.
        training_config : TrainingParameters
            Parameters for the training process.
        runtime_config : RuntimeParameters
            Parameters for runtime configuration.
        lr_scheduler : Dict[str, Union[str, int, float]]
            The configuration for the learning rate scheduler.
        lr : float
            The learning rate for the optimizer.
        loss_parameter : Dict[str, Any]
            Configuration for the loss function.
        datamodule : DataModule
            The DataModule for loading datasets.
        optimizer : Type[Optimizer], optional
            The optimizer class to use for training, by default torch.optim.AdamW.
        verbose : bool, optional
            If True, enables verbose logging, by default False.
        """

        super().__init__()

        self.dataset_parameter = dataset_parameter
        self.potential_parameter = potential_parameter
        self.training_parameter = training_parameter
        self.runtime_parameter = runtime_parameter

        self.datamodule = self.setup_datamodule()
        self.dataset_statistic = (
            self.read_dataset_statistics()
            if not use_default_dataset_statistic
            else dataset_statistic
        )
        self.experiment_logger = self.setup_logger()
        self.model = self.setup_potential(potential_seed)
        self.callbacks = self.setup_callbacks()
        self.trainer = self.setup_trainer()
        self.optimizer = optimizer
        self.learning_rate = self.training_parameter.lr
        self.lr_scheduler = self.training_parameter.lr_scheduler

        # Verbose output
        if verbose:
            self.log_histograms = True
            self.log_on_training_step = True
        else:
            self.log_histograms = False
            self.log_on_training_step = False

        # Initialize loss
        self.loss = LossFactory.create_loss(
            **self.training_parameter.loss_parameter.model_dump()
        )

        # Assign the created error metrics to the respective attributes
        self.test_error = create_error_metrics(
            self.training_parameter.loss_parameter.loss_property
        )
        self.val_error = create_error_metrics(
            self.training_parameter.loss_parameter.loss_property
        )
        self.train_error = create_error_metrics(
            self.training_parameter.loss_parameter.loss_property
        )

    def read_dataset_statistics(
        self,
    ) -> Dict[str, float]:
        """
        Read and log dataset statistics.

        Returns
        -------
        Dict[str, float]
            The dataset statistics.
        """
        from modelforge.potential.utils import (
            read_dataset_statistics,
            convert_str_to_unit_in_dataset_statistics,
        )

        # read toml file
        dataset_statistic = read_dataset_statistics(
            self.datamodule.dataset_statistic_filename
        )
        # convert dictionary of str:str to str:units
        dataset_statistic = convert_str_to_unit_in_dataset_statistics(dataset_statistic)
        log.info(
            f"Setting per_atom_energy_mean and per_atom_energy_stddev for {self.potential_parameter.potential_name}"
        )
        log.info(
            f"per_atom_energy_mean: {dataset_statistic['training_dataset_statistics']['per_atom_energy_mean']}"
        )
        log.info(
            f"per_atom_energy_stddev: {dataset_statistic['training_dataset_statistics']['per_atom_energy_stddev']}"
        )
        return dataset_statistic

    def setup_datamodule(self) -> DataModule:
        """
        Set up the DataModule for the dataset.

        Returns
        -------
        DataModule
            Configured DataModule instance.
        """
        from modelforge.dataset.dataset import DataModule
        from modelforge.dataset.utils import REGISTERED_SPLITTING_STRATEGIES

        dm = DataModule(
            name=self.dataset_parameter.dataset_name,
            batch_size=self.training_parameter.batch_size,
            remove_self_energies=self.training_parameter.remove_self_energies,
            version_select=self.dataset_parameter.version_select,
            local_cache_dir=self.runtime_parameter.local_cache_dir,
            splitting_strategy=REGISTERED_SPLITTING_STRATEGIES[
                self.training_parameter.splitting_strategy.name
            ](
                seed=self.training_parameter.splitting_strategy.seed,
                split=self.training_parameter.splitting_strategy.data_split,
            ),
            regenerate_processed_cache=self.dataset_parameter.regenerate_processed_cache,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def setup_potential(
        self, potential_seed: Optional[int] = None
    ) -> pL.LightningModule:
        """
        Set up the model for training.

        Parameters
        ----------
        potential_seed : int, optional
            Seed to be used to initialize the potential, by default None.

        Returns
        -------
        nn.Module
            Configured model instance, wrapped in a TrainingAdapter.
        """

        # Initialize model
        return TrainingAdapter(
            potential_parameter=self.potential_parameter,
            dataset_statistic=self.dataset_statistic,
            training_parameter=self.training_parameter,
            potential_seed=potential_seed,
        )

    def setup_logger(self) -> pL.loggers.Logger:
        """
        Set up the experiment logger based on the configuration.

        Returns
        -------
        pL.loggers.Logger
            Configured logger instance.
        """
        if self.training_parameter.experiment_logger.logger_name == "tensorboard":
            from lightning.pytorch.loggers import TensorBoardLogger

            logger = TensorBoardLogger(
                save_dir=str(
                    self.training_parameter.experiment_logger.tensorboard_configuration.save_dir
                ),  # FIXME: same variable for all logger, maybe we can use a varable not bound to a logger for this?
                name=self._replace_placeholder_in_experimental_name(
                    self.runtime_parameter.experiment_name
                ),
            )
        elif self.training_parameter.experiment_logger.logger_name == "wandb":
            from modelforge.utils.io import check_import

            check_import("wandb")
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                save_dir=str(
                    self.training_parameter.experiment_logger.wandb_configuration.save_dir
                ),
                log_model=str(
                    self.training_parameter.experiment_logger.wandb_configuration.log_model
                ),
                project=self.training_parameter.experiment_logger.wandb_configuration.project,
                group=self.training_parameter.experiment_logger.wandb_configuration.group,
                job_type=self.training_parameter.experiment_logger.wandb_configuration.job_type,
                tags=self._add_tags(
                    self.training_parameter.experiment_logger.wandb_configuration.tags
                ),
                notes=self.training_parameter.experiment_logger.wandb_configuration.notes,
                name=self._replace_placeholder_in_experimental_name(
                    self.runtime_parameter.experiment_name
                ),
            )
        return logger

    def setup_callbacks(self) -> List[Any]:
        """
        Set up the callbacks for the trainer.

        Returns
        -------
        List[Any]
            List of configured callbacks.
        """
        from lightning.pytorch.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            StochasticWeightAveraging,
        )

        callbacks = []
        if self.training_parameter.stochastic_weight_averaging is not None:
            callbacks.append(
                StochasticWeightAveraging(
                    **self.training_parameter.stochastic_weight_averaging.model_dump()
                )
            )

        if self.training_parameter.early_stopping is not None:
            callbacks.append(
                EarlyStopping(**self.training_parameter.early_stopping.model_dump())
            )

        checkpoint_filename = (
            f"best_{self.potential_parameter.potential_name}-{self.dataset_parameter.dataset_name}"
            + "-{epoch:02d}-{val_loss:.2f}"
        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            monitor=self.training_parameter.monitor_for_checkpoint,
            filename=checkpoint_filename,
        )
        callbacks.append(checkpoint_callback)
        return callbacks

    def setup_trainer(self) -> Trainer:
        """
        Set up the Trainer for training.

        Returns
        -------
        Trainer
            Configured Trainer instance.
        """
        from lightning import Trainer

        # if devices is a list
        if isinstance(self.runtime_parameter.devices, list) or (
            isinstance(self.runtime_parameter.devices, int)
            and self.runtime_parameter.devices > 1
        ):
            from lightning.pytorch.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = "auto"

        trainer = Trainer(
            strategy=strategy,
            max_epochs=self.training_parameter.number_of_epochs,
            min_epochs=self.training_parameter.min_number_of_epochs,
            num_nodes=self.runtime_parameter.number_of_nodes,
            devices=self.runtime_parameter.devices,
            accelerator=self.runtime_parameter.accelerator,
            logger=self.experiment_logger,
            callbacks=self.callbacks,
            inference_mode=False,
            num_sanity_val_steps=2,
            log_every_n_steps=self.runtime_parameter.log_every_n_steps,
            enable_model_summary=True,
        )
        return trainer

    def train_potential(self) -> Trainer:
        """
        Run the training process.

        Returns
        -------
        Trainer
            The configured trainer instance after running the training process.
        """
        self.trainer.fit(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(
                num_workers=self.dataset_parameter.num_workers,
                pin_memory=self.dataset_parameter.pin_memory,
            ),
            val_dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=(
                self.runtime_parameter.checkpoint_path
                if self.runtime_parameter.checkpoint_path != "None"
                else None
            ),  # NOTE: automatically resumes training from checkpoint
        )

        self.trainer.validate(
            dataloaders=self.datamodule.val_dataloader(),
            ckpt_path="best",
            verbose=True,
        )

        self.trainer.test(
            dataloaders=self.datamodule.test_dataloader(),
            ckpt_path="best",
            verbose=True,
        )
        return self.trainer

    def config_prior(self):
        """
        Configures model-specific priors if the model implements them.
        """
        if hasattr(self.model, "_config_prior"):
            return self.model._config_prior()

        log.warning("Model does not implement _config_prior().")
        raise NotImplementedError()

    def _replace_placeholder_in_experimental_name(self, experiment_name: str) -> str:
        """
        Replace the placeholders in the experiment name with the actual values.

        Parameters
        ----------
        experiment_name : str
            The experiment name with placeholders.

        Returns
        -------
        str
            The experiment name with the placeholders replaced.
        """
        # replace placeholders in the experiment name
        experiment_name = experiment_name.replace(
            "{potential_name}", self.potential_parameter.potential_name
        )
        experiment_name = experiment_name.replace(
            "{dataset_name}", self.dataset_parameter.dataset_name
        )
        return experiment_name

    def _add_tags(self, tags: List[str]) -> List[str]:
        """
        Add tags to the wandb tags.

        Parameters
        ----------
        tags : List[str]
            List of tags to add to the experiment.

        Returns
        -------
        List[str]
            List of tags for the experiment.
        """

        # add version
        import modelforge

        tags.append(str(modelforge.__version__))
        # add dataset
        tags.append(self.dataset_parameter.dataset_name)
        # add potential name
        tags.append(self.potential_parameter.potential_name)
        # add information about what is included in the loss
        str_loss_property = "-".join(
            self.training_parameter.loss_parameter.loss_property
        )
        tags.append(f"loss-{str_loss_property}")

        return tags


from typing import List, Optional, Union


def read_config(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path: Optional[str] = None,
    runtime_parameter_path: Optional[str] = None,
    accelerator: Optional[str] = None,
    devices: Optional[Union[int, List[int]]] = None,
    number_of_nodes: Optional[int] = None,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    local_cache_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    log_every_n_steps: Optional[int] = None,
    simulation_environment: Optional[str] = None,
):
    """
    Reads one or more TOML configuration files and loads them into the pydantic models.

    Parameters
    ----------
    (Parameters as described earlier...)

    Returns
    -------
    Tuple
        Tuple containing the training, dataset, potential, and runtime parameters.
    """
    import toml

    # Initialize the config dictionaries
    training_config_dict = {}
    dataset_config_dict = {}
    potential_config_dict = {}
    runtime_config_dict = {}

    if condensed_config_path is not None:
        config = toml.load(condensed_config_path)
        log.info(f"Reading config from : {condensed_config_path}")

        training_config_dict = config.get("training", {})
        dataset_config_dict = config.get("dataset", {})
        potential_config_dict = config.get("potential", {})
        runtime_config_dict = config.get("runtime", {})

    else:
        if training_parameter_path:
            training_config_dict = toml.load(training_parameter_path).get(
                "training", {}
            )
        if dataset_parameter_path:
            dataset_config_dict = toml.load(dataset_parameter_path).get("dataset", {})
        if potential_parameter_path:
            potential_config_dict = toml.load(potential_parameter_path).get(
                "potential", {}
            )
        if runtime_parameter_path:
            runtime_config_dict = toml.load(runtime_parameter_path).get("runtime", {})

    # Override runtime configuration with command-line arguments if provided
    runtime_overrides = {
        "accelerator": accelerator,
        "devices": devices,
        "number_of_nodes": number_of_nodes,
        "experiment_name": experiment_name,
        "save_dir": save_dir,
        "local_cache_dir": local_cache_dir,
        "checkpoint_path": checkpoint_path,
        "log_every_n_steps": log_every_n_steps,
        "simulation_environment": simulation_environment,
    }

    for key, value in runtime_overrides.items():
        if value is not None:
            runtime_config_dict[key] = value

    # Load and instantiate the data classes with the merged configuration
    from modelforge.dataset.dataset import DatasetParameters
    from modelforge.potential import _Implemented_NNP_Parameters
    from modelforge.train.parameters import RuntimeParameters, TrainingParameters

    potential_name = potential_config_dict["potential_name"]
    PotentialParameters = (
        _Implemented_NNP_Parameters.get_neural_network_parameter_class(potential_name)
    )

    dataset_parameters = DatasetParameters(**dataset_config_dict)
    training_parameters = TrainingParameters(**training_config_dict)
    runtime_parameters = RuntimeParameters(**runtime_config_dict)
    potential_parameter = PotentialParameters(**potential_config_dict)

    return (
        training_parameters,
        dataset_parameters,
        potential_parameter,
        runtime_parameters,
    )


def read_config_and_train(
    condensed_config_path: Optional[str] = None,
    training_parameter_path: Optional[str] = None,
    dataset_parameter_path: Optional[str] = None,
    potential_parameter_path: Optional[str] = None,
    runtime_parameter_path: Optional[str] = None,
    accelerator: Optional[str] = None,
    devices: Optional[Union[int, List[int]]] = None,
    number_of_nodes: Optional[int] = None,
    experiment_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    local_cache_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    log_every_n_steps: Optional[int] = None,
    simulation_environment: Optional[str] = "PyTorch",
):
    """
    Reads one or more TOML configuration files and performs training based on the parameters.

    Parameters
    ----------
    condensed_config_path : str, optional
        Path to the TOML configuration that contains all parameters for the dataset, potential, training, and runtime parameters.
        Any other provided configuration files will be ignored.
    training_parameter_path : str, optional
        Path to the TOML file defining the training parameters.
    dataset_parameter_path : str, optional
        Path to the TOML file defining the dataset parameters.
    potential_parameter_path : str, optional
        Path to the TOML file defining the potential parameters.
    runtime_parameter_path : str, optional
        Path to the TOML file defining the runtime parameters. If this is not provided, the code will attempt to use
        the runtime parameters provided as arguments.
    accelerator : str, optional
        Accelerator type to use.  If provided, this  overrides the accelerator type in the runtime_defaults configuration.
    devices : int|List[int], optional
        Device index/indices to use.  If provided, this overrides the devices in the runtime_defaults configuration.
    number_of_nodes : int, optional
        Number of nodes to use.  If provided, this overrides the number of nodes in the runtime_defaults configuration.
    experiment_name : str, optional
        Name of the experiment.  If provided, this overrides the experiment name in the runtime_defaults configuration.
    save_dir : str, optional
        Directory to save the model.  If provided, this overrides the save directory in the runtime_defaults configuration.
    local_cache_dir : str, optional
        Local cache directory.  If provided, this overrides the local cache directory in the runtime_defaults configuration.
    checkpoint_path : str, optional
        Path to the checkpoint file.  If provided, this overrides the checkpoint path in the runtime_defaults configuration.
    log_every_n_steps : int, optional
        Number of steps to log.  If provided, this overrides the log_every_n_steps in the runtime_defaults configuration.
    simulation_environment : str, optional
        Simulation environment.  If provided, this overrides the simulation environment in the runtime_defaults configuration.

    Returns
    -------
    Trainer
        The configured trainer instance after running the training process.
    """
    (
        training_parameter,
        dataset_parameter,
        potential_parameter,
        runtime_parameter,
    ) = read_config(
        condensed_config_path=condensed_config_path,
        training_parameter_path=training_parameter_path,
        dataset_parameter_path=dataset_parameter_path,
        potential_parameter_path=potential_parameter_path,
        runtime_parameter_path=runtime_parameter_path,
        accelerator=accelerator,
        devices=devices,
        number_of_nodes=number_of_nodes,
        experiment_name=experiment_name,
        save_dir=save_dir,
        local_cache_dir=local_cache_dir,
        checkpoint_path=checkpoint_path,
        log_every_n_steps=log_every_n_steps,
        simulation_environment=simulation_environment,
    )
    from modelforge.potential.models import NeuralNetworkPotentialFactory

    model = NeuralNetworkPotentialFactory.generate_potential(
        use="training",
        potential_parameter=potential_parameter,
        training_parameter=training_parameter,
        dataset_parameter=dataset_parameter,
        runtime_parameter=runtime_parameter,
    )

    return model.train_potential()
