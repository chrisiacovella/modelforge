import pytest


def setup_methane():
    import torch

    device = torch.device("cpu")
    coordinates = torch.tensor(
        [
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ]
        ],
        requires_grad=True,
        device=device,
    )
    # In periodic table, C = 6 and H = 1
    species = torch.tensor([[1, 0, 0, 0, 0]], device=device)
    atomic_subsystem_indices = torch.tensor(
        [0, 0, 0, 0, 0], dtype=torch.int32, device=device
    )

    from modelforge.dataset.dataset import NNPInput

    nnp_input = NNPInput(
        atomic_numbers=torch.tensor([6, 1, 1, 1, 1], device=device),
        positions=coordinates.squeeze(0) / 10,
        atomic_subsystem_indices=atomic_subsystem_indices,
        total_charge=torch.tensor([0.0]),
    )

    return species, coordinates, device, nnp_input


def setup_two_methanes():
    import torch

    device = torch.device("cpu")

    coordinates = torch.tensor(
        [
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ],
            [
                [0.03192167, 0.00638559, 0.01301679],
                [-0.83140486, 0.39370209, -0.26395324],
                [-0.66518241, -0.84461308, 0.20759389],
                [0.45554739, 0.54289633, 0.81170881],
                [0.66091919, -0.16799635, -0.91037834],
            ],
        ],
        requires_grad=True,
        device=device,
    )
    # In periodic table, C = 6 and H = 1
    mf_species = torch.tensor([6, 1, 1, 1, 1, 6, 1, 1, 1, 1], device=device)
    ani_species = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], device=device)
    atomic_subsystem_indices = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.int32, device=device
    )

    atomic_numbers = mf_species
    from modelforge.dataset.dataset import NNPInput

    nnp_input = NNPInput(
        atomic_numbers=atomic_numbers,
        positions=torch.cat((coordinates[0], coordinates[1]), dim=0) / 10,
        atomic_subsystem_indices=atomic_subsystem_indices,
        total_charge=torch.tensor([0.0, 0.0]),
    )
    return ani_species, coordinates, device, nnp_input


def test_init():
    from modelforge.potential.ani import ANI2x
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    # read default parameters
    config = load_configs_into_pydantic_models("ani2x", "qm9")

    # initialize model
    model = ANI2x(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )


@pytest.mark.xfail
def test_forward_and_backward_using_torchani():
    # Test torchani ANI implementation
    # Test forward pass and backpropagation through network

    import torch
    import torchani

    species, coordinates, device, _ = setup_two_methanes()
    model = torchani.models.ANI2x(periodic_table_index=False).to(device)

    energy = model((species, coordinates)).energies
    derivative = torch.autograd.grad(energy.sum(), coordinates)[0]
    per_atom_force = -derivative


def test_forward_and_backward():
    # Test modelforge ANI implementation
    # Test forward pass and backpropagation through network
    from modelforge.potential.ani import ANI2x
    from modelforge.tests.test_models import load_configs_into_pydantic_models
    import torch

    # read default parameters
    config = load_configs_into_pydantic_models("ani2x", "qm9")

    _, _, _, mf_input = setup_two_methanes()
    device = torch.device("cpu")

    # initialize model
    model = ANI2x(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    ).to(device=device)
    energy = model(mf_input)
    derivative = torch.autograd.grad(
        energy["per_molecule_energy"].sum(), mf_input.positions
    )[0]
    per_atom_force = -derivative


def test_representation():
    # Compare the reference radial symmetry function
    # against the the implemented radial symmetry function
    import torch
    from modelforge.potential.utils import (
        AniRadialBasisFunction,
        CosineAttenuationFunction,
    )
    from openff.units import unit
    from .precalculated_values import (
        provide_reference_values_for_test_ani_test_compare_rsf,
    )

    # use d_ij in angstrom
    d_ij = torch.tensor([[3.5201], [2.6756], [2.1641], [3.0990], [4.5180]])
    radial_cutoff = 5.0  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 8

    # NOTE: we pass in Angstrom to ANI and in nanometer to mf
    rsf = AniRadialBasisFunction(
        number_of_radial_basis_functions=radial_dist_divisions,
        max_distance=radial_cutoff * unit.angstrom,
        min_distance=radial_start * unit.angstrom,
    )
    calculated_rsf = rsf(d_ij / 10)  # torch.Size([5,1, 8]) # NOTE: nanometer
    cutoff_module = CosineAttenuationFunction(radial_cutoff * unit.angstrom)

    rcut_ij = cutoff_module(d_ij / 10)  # torch.Size([5]) # NOTE: nanometer
    reference_rsf = provide_reference_values_for_test_ani_test_compare_rsf()
    calculated_rsf = calculated_rsf * rcut_ij
    assert torch.allclose(calculated_rsf, reference_rsf, rtol=1e-4)


def test_representation_with_diagonal_batching():
    import torch
    from modelforge.potential.utils import (
        AniRadialBasisFunction,
        CosineAttenuationFunction,
    )
    from openff.units import unit
    from modelforge.potential.models import Pairlist
    from .precalculated_values import (
        provide_reference_values_for_test_ani_test_compute_rsf_with_diagonal_batching,
    )

    # ------------ general setup -------------#
    ani_species, ani_coordinates, _, mf_input = setup_two_methanes()
    pairlist = Pairlist(only_unique_pairs=True)
    pairs = pairlist(
        mf_input.positions,
        mf_input.atomic_subsystem_indices,
    )
    d_ij = pairs.d_ij

    # ANI constants
    radial_cutoff = 5.1  # radial_cutoff
    radial_start = 0.8
    radial_dist_divisions = 16
    # ------------ Modelforge calculation ----------#
    device = torch.device("cpu")

    radial_symmetry_function = AniRadialBasisFunction(
        radial_dist_divisions,
        radial_cutoff * unit.angstrom,
        radial_start * unit.angstrom,
    ).to(device=device)

    cutoff_module = CosineAttenuationFunction(radial_cutoff * unit.angstrom).to(
        device=device
    )
    rcut_ij = cutoff_module(d_ij)

    calculated_rbf_output = radial_symmetry_function(d_ij)
    calculated_rbf_output = calculated_rbf_output * rcut_ij

    # test that both ANI and MF obtain the same radial symmetry outpu
    reference_rbf_output, ani_d_ij = (
        provide_reference_values_for_test_ani_test_compute_rsf_with_diagonal_batching()
    )
    assert torch.allclose(calculated_rbf_output, reference_rbf_output, atol=1e-4)
    assert torch.allclose(
        ani_d_ij, d_ij.squeeze(1) * 10, atol=1e-4
    )  # NOTE: unit mismatch

    assert calculated_rbf_output.shape == torch.Size([20, radial_dist_divisions])


def test_compare_angular_symmetry_features():
    # Compare the calculated angular symmetry function output
    # against the reference angular symmetry functino output

    import torch
    from modelforge.potential.utils import AngularSymmetryFunction, triple_by_molecule
    from openff.units import unit
    from modelforge.potential.models import Pairlist

    device = torch.device("cpu")

    # set up relevant system properties
    species, r, _, _ = setup_methane()
    pairlist = Pairlist(only_unique_pairs=True).to(device=device)
    pairs = pairlist(r[0], torch.tensor([0, 0, 0, 0, 0], device=device))
    d_ij = pairs.d_ij.squeeze(1)
    r_ij = pairs.r_ij.squeeze(1)

    # reformat for input
    species = species.flatten()
    atom_index12 = pairs.pair_indices
    # ANI constants
    # for angular features
    angular_cutoff = Rca = 3.5  # angular_cutoff
    angular_start = 0.8
    angular_dist_divisions = 8

    # get index in right order
    even_closer_indices = (d_ij <= Rca).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, even_closer_indices)
    r_ij = r_ij.index_select(0, even_closer_indices)
    central_atom_index, pair_index12, sign12 = triple_by_molecule(atom_index12)
    vec12 = r_ij.index_select(0, pair_index12.view(-1)).view(
        2, -1, 3
    ) * sign12.unsqueeze(-1)

    # now use formated indices and inputs to calculate the
    # angular terms, both with the modelforge AngularSymmetryFunction
    # and with its implementation in torchani

    # ref value
    from .precalculated_values import (
        provide_input_for_test_test_compare_angular_symmetry_features,
    )

    reference_angular_feature_vector = (
        provide_input_for_test_test_compare_angular_symmetry_features()
    )

    # set up modelforge angular features
    asf = AngularSymmetryFunction(
        angular_cutoff * unit.angstrom,
        angular_start * unit.angstrom,
        angular_dist_divisions,
        angle_sections=4,
    )
    # NOTE: ANI works with Angstrom, modelforge with nanometer
    # NOTE: ANI operates on a [nr_of_molecules, nr_of_atoms, 3] tensor
    calculated_angular_feature_vector = asf(vec12 / 10)
    # make sure that the output is the same
    assert (
        calculated_angular_feature_vector.size()
        == reference_angular_feature_vector.size()
    )

    # NOTE: the order of the angular_feature_vector is not guaranteed
    # as the triple_by_molecule function  used to prepare the inputs does not use stable sorting.
    # When stable sorting is used, the output is identical across platforms, but will not be
    # used here as it is slower and the order of the output is not important in practrice.
    # As such, to check for equivalence in a way that is not order dependent, we can just consider the sum.
    assert torch.isclose(
        torch.sum(calculated_angular_feature_vector),
        torch.sum(reference_angular_feature_vector),
        atol=1e-4,
    )


def test_compare_aev():
    """
    Compare the atomic enviornment vector generated by the reference implementation (torchani) and modelforge for the same input
    """
    import torch
    from .precalculated_values import provide_input_for_test_ani_test_compare_aev

    # methane input
    species, coordinates, device, mf_input = setup_methane()

    # generate modelforge ani representation
    from modelforge.potential import ANI2x

    # read default parameters
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    # read default parameters
    config = load_configs_into_pydantic_models("ani2x", "qm9")

    # Extract parameters

    mf_model = ANI2x(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )
    # perform input checks
    mf_model.compute_interacting_pairs._input_checks(mf_input)
    # prepare the input for the forward pass
    pairlist_output = mf_model.compute_interacting_pairs.prepare_inputs(mf_input)
    nnp_input = mf_model.core_module._model_specific_input_preparation(
        mf_input, pairlist_output
    )
    representation_module_output = mf_model.core_module.ani_representation_module(
        nnp_input
    )

    reference_aev = provide_input_for_test_ani_test_compare_aev()
    # test for equivalence
    assert torch.Size([5, 1008]) == representation_module_output.aevs.shape
    # compare a selected subsection
    assert torch.allclose(
        reference_aev, representation_module_output.aevs[::2, :50:5], atol=1e-4
    )
