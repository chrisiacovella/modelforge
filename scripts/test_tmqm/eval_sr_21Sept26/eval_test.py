import csv

import numpy as np
from modelforge.utils.remote import download_from_url

from modelforge.curate.sourcedataset import create_dataset_from_hdf5
from modelforge.curate.properties import *

from tqdm import tqdm
from openff.units import unit
import torch

# read in the file we downloaded
local_cache = "/home/cri/dataset_cache"

from modelforge.potential.potential import load_inference_model_from_checkpoint

checkpoint_file_path = f"/home/cri/PycharmProjects/tm_models/no_invalid_aimnet2_sr_17Sept26/checkpoints/model-rapwvh68_v0/model.ckpt"  # This is an example model used in testing
potential = load_inference_model_from_checkpoint(checkpoint_file_path, jit=False)
print(f"Loaded checkpoint: {checkpoint_file_path}")
potential.to("cuda")

from modelforge.ase import ModelForgeCalculator

from modelforge.dataset.dataset import initialize_datamodule

splitting_seed = 435
# dataset_version = "valid_dataset_v1.4_3sep26"
# dm = initialize_datamodule(
#     dataset_name="tmqm_openff_local",
#     batch_size=100000,
#     splitting_strategy=RandomRecordSplittingStrategy(
#         seed=splitting_seed, split=[0.8, 0.1, 0.1]
#     ),
#     regression_ase=False,
#     remove_self_energies=False,
#     version_select=dataset_version,
#     local_cache_dir="./local_cache",
#     dataset_cache_dir="~/dataset_cache",
#     properties_of_interest=[
#         "positions",
#         "dft_total_energy_corrected",
#         "atomic_numbers",
#         "total_charge",
#         "lowdin_partial_charges",
#     ],
#     properties_assignment={
#         "positions": "positions",
#         "E": "dft_total_energy_corrected",
#         "atomic_numbers": "atomic_numbers",
#         "total_charge": "total_charge",
#         "partial_charges": "lowdin_partial_charges",
#     },
#     local_yaml_file="/home/cri/Downloads/3Sep26/tmqm_openff_local.yaml",
# )

input_file = "/home/cri/mf_datasets/hdf5_files/tmqm_openff_dataset/v1.5_17sep26/tmqm_openff_full_dataset_corrected_v1.5.hdf5"


property_map = {
    "atomic_numbers": AtomicNumbers,
    "positions": Positions,
    "total_charge": TotalCharge,
    "per_system_spin_multiplicity": SpinMultiplicitiesPerSystem,
    "dft_total_energy": Energies,
    "dft_total_energy_corrected": Energies,
    "dft_total_force": Forces,
    "scf_dipole": DipoleMomentPerSystem,
    "scf_quadrupole": QuadrupoleMomentPerSystem,
    "mulliken_partial_charges": PartialCharges,
    "lowdin_partial_charges": PartialCharges,
    "spin_multiplicity_per_atom": SpinMultiplicitiesPerAtom,
}
n_records = 100

dataset = create_dataset_from_hdf5(
    hdf5_filename=input_file,
    dataset_name="tmqm_openff_test",
    dataset_local_db_dir=local_cache,
    # n_records=n_records,
    property_map=property_map,
)


# simple function that bypasses the need to have a dataset
def indices_for_test(splitting_seed, split=[0.8, 0.1, 0.1]):
    from modelforge.dataset.utils import (
        RandomRecordSplittingStrategy,
        calculate_size_of_splits,
    )

    strategy = RandomRecordSplittingStrategy(seed=splitting_seed, split=split)

    lengths = calculate_size_of_splits(
        dataset.total_records(), split_frac=split, enforce_sum_to_unity=True
    )
    record_indices = torch.randperm(sum(lengths), generator=strategy.generator).tolist()  # type: ignore[arg-type, call-overload]

    indices_by_split: List[List[int]] = []
    for offset, length in zip(np.cumsum(lengths), lengths):
        indices = []
        for record_idx in record_indices[offset - length : offset]:
            indices.append(record_idx)
        indices_by_split.append(indices)

    return indices_by_split


train_idx, val_idx, test_idx = indices_for_test(splitting_seed)


def rdkit_mol_to_ase(mol: Mol, smiles: str = "", index=0) -> Atoms:
    """
    Convert an RDKit Mol (with 3D conformer) to an ASE Atoms object.

    Parameters
    ----------
    mol    : RDKit Mol with at least one 3D conformer embedded
    smiles : optional label string for the Atoms object

    Returns
    -------
    ase.Atoms
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdchem import Mol
    from ase import Atoms

    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer. Embed 3D coordinates first.")

    conf = mol.GetConformer(index)
    positions = conf.GetPositions()  # (N, 3) float array in Angstrom

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    ase_system = Atoms(symbols=symbols, positions=positions)
    ase_system.info["smiles"] = smiles
    return ase_system


import os

record_names = dataset.record_names()
output_dir = "test"
count = 0

os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/molecules", exist_ok=True)
os.makedirs(f"{output_dir}/summary_plots", exist_ok=True)
os.makedirs(f"{output_dir}/trajectories", exist_ok=True)

dft_energies_all = []
model_energies_all = []
model_shifted_energies_all = []
spin_multiplicity_all = []

initial_mae_all = []
shifted_mae_all = []
spin_multiplicity_per_traj = []
estimated_shift_all = []

for idx in test_idx:

    # if count > 10:
    #    break

    record_name = record_names[idx]

    dft_energies = {1: [], 3: [], 5: []}
    model_energies = {1: [], 3: [], 5: []}
    try:
        record = dataset.get_record(record_name)
    except KeyError:
        continue
    atomic_numbers = record.get_property("atomic_numbers")

    corrected_energy = record.get_property("dft_total_energy_corrected").value

    spin_multiplicity = record.get_property("per_system_spin_multiplicity").value
    total_charge = record.get_property("total_charge").value

    diff_energies = {}
    for sm in [1, 3, 5]:
        diff_energies[sm] = []
    for sm in [1, 3, 5]:
        # we only want the configs where spin_multiplicity == sm
        indices = spin_multiplicity == sm
        for i in range(record.n_configs):
            if indices[i]:
                dft_energies[sm].append(corrected_energy[i])

                modelforge_calculator = ModelForgeCalculator(
                    potential,
                    per_system_total_charge=int(total_charge[i][0]),
                    per_system_spin_multiplicity=sm,
                    # per_system_spin_multiplicity=int(spin_multiplicity[i][0]),
                )
                print(
                    f"{record_name} : total charge: {total_charge[i][0]} sm: {sm} sm_A: {spin_multiplicity[i][0]}"
                )
                atoms = rdkit_mol_to_ase(record.to_rdkit(), index=i)
                atoms.calc = modelforge_calculator
                pe = atoms.get_potential_energy()

                model_energies[sm].append(pe * 96.485)  # convert eV to kJ/mol

                diff_energies[sm].append((pe * 96.485) - corrected_energy[i])
        if len(dft_energies[sm]) == 0:
            continue

        initial_mae = np.mean(np.abs(diff_energies[sm]))

        # shift DFT and model energy by the mean values

        estimated_shift = np.mean(np.array(dft_energies[sm]).flatten()) - np.mean(
            model_energies[sm]
        )

        estimated_shift_all.append(estimated_shift)
        model_energy_shifted = np.array(model_energies[sm]) + estimated_shift
        diff_energy_shifted = np.array(diff_energies[sm]) + model_energy_shifted
        shifted_mae = np.mean(
            np.abs(np.array(dft_energies[sm]).flatten() - model_energy_shifted)
        )
        # print(f"estimated_shift {estimated_shift}")
        # print(f"initial_mae: {initial_mae}\t shifted_mae: {shifted_mae}")
        # we want a metric that defines if a series is good

        # log all of our various energies and maes to the global arrays
        # we want a big 1d array

        for i in range(0, len(dft_energies[sm])):
            dft_energies_all.append(dft_energies[sm][i].flatten())
            model_energies_all.append(model_energies[sm][i])
            model_shifted_energies_all.append(model_energy_shifted[i])

            spin_multiplicity_all.append(sm)

        initial_mae_all.append(initial_mae)
        shifted_mae_all.append(shifted_mae)
        spin_multiplicity_per_traj.append(sm)

        metric = np.sum(diff_energies[sm]) / len(dft_energies[sm])
        print(metric)

        # save the molecule as an rdkit image
        rec = record.to_rdkit()
        filename = (
            f"{output_dir}/molecules/{record_name}_q{total_charge[i][0]}_image.png"
        )

        # save the rdkit molecule as an image
        from rdkit import Chem
        from rdkit.Chem import Draw

        img = Draw.MolToImage(rec, size=(300, 300))

        # 3. Save to a file
        img.save(filename)

        from matplotlib import pyplot as plt

        plt.plot(
            np.array(dft_energies[sm]),
            linestyle="-",
            marker="o",
            label="DFT",
            # label=f"MAE: {mae:.2f} kJ/mol",
        )
        plt.plot(
            np.array(model_energies[sm]),
            linestyle="-",
            label="Model",
            marker="x",
        )
        plt.plot(
            model_energy_shifted,
            linestyle="-",
            marker="o",
            label="model shifted",
            # label=f"MAE: {mae:.2f} kJ/mol",
        )
        plt.title(
            f"{record_name} sm: {sm} Q: {total_charge[i][0]} i: {initial_mae:.2f} s: {shifted_mae:.2f}"
        )
        plt.xlabel("config")
        plt.ylabel("Energies (kJ/mol)")
        plt.legend()
        # need to set reasonable bounds to ignore any major outliers

        # plt.legend()
        plt.savefig(
            f"{output_dir}/trajectories/traj_tmqm_{record_name}_sm{sm}_Q{total_charge[i][0]}_ext.png"
        )
        plt.cla()

        count = count + 1

output_dir = "test/summary_plots"
# now let us plot all of the global arrays

from matplotlib import pyplot as plt

# first plot the dft energy vs model for initial and shifted

plt.plot(
    np.array(model_energies_all).flatten(),
    np.array(dft_energies_all).flatten(),
    label="initial",
    linestyle="",
    marker="o",
    alpha=0.2,
)
plt.xlabel("model energy (kJ/mol)")
plt.ylabel("DFT energy (kJ/mol)")
plt.title("DFT energy vs model energy")
# plt.legend()
plt.savefig(f"{output_dir}/dft_model_energy.png")
plt.cla()

plt.plot(
    np.array(model_shifted_energies_all).flatten(),
    np.array(dft_energies_all).flatten(),
    label="shifted",
    linestyle="",
    marker="o",
    alpha=0.2,
)
plt.xlabel("model energy shifted (kJ/mol)")
plt.ylabel("DFT energy (kJ/mol)")
plt.title("DFT energy vs shifted model energy")
plt.savefig(f"{output_dir}/dft_shifted_model_energy.png")
plt.cla()

# now plot this for each sm

for sm in [1, 3, 5]:
    indices = np.array(spin_multiplicity_all) == sm
    plt.plot(
        np.array(model_energies_all).flatten()[indices],
        np.array(dft_energies_all).flatten()[indices],
        label=f"initial sm={sm}",
        linestyle="",
        marker="o",
        alpha=0.2,
    )
    plt.xlabel("model energy (kJ/mol)")
    plt.ylabel("DFT energy (kJ/mol)")
    plt.title(f"DFT energy vs initial sm={sm}")
    plt.savefig(f"{output_dir}/dft_model_sm_{sm}.png")

    plt.cla()

    plt.plot(
        np.array(model_shifted_energies_all).flatten()[indices],
        np.array(dft_energies_all).flatten()[indices],
        label=f"shifted sm={sm}",
        linestyle="",
        marker="o",
        alpha=0.2,
    )
    plt.xlabel("model energy shifted (kJ/mol)")
    plt.ylabel("DFT energy (kJ/mol)")
    plt.title(f"DFT energy vs shifted model energy={sm}")
    plt.savefig(f"{output_dir}/dft_model_shift_{sm}.png")
    plt.cla()

    # do histograms for mae for each sm
    indices = np.array(spin_multiplicity_per_traj) == sm
    plt.hist(np.array(initial_mae_all)[indices], label="initial mae", bins=100)
    plt.xlabel("initial mae (kJ/mol)")
    plt.ylabel("number of configurations")
    plt.title(f"histogram of initial mae for sm={sm}")
    plt.savefig(f"{output_dir}/hist_initial_mae_{sm}.png")

    plt.cla()

    plt.hist(np.array(shifted_mae_all)[indices], label="shifted mae", bins=100)
    plt.xlabel("shifted mae (kJ/mol)")
    plt.ylabel("number of configurations")
    plt.title(f"histogram of shifted mae for sm={sm}")
    plt.savefig(f"{output_dir}/hist_shifted_mae_{sm}.png")
    plt.cla()

    # now plot the estimated shift histogram
    plt.hist(np.array(estimated_shift_all)[indices], label="estimated shift", bins=100)
    plt.xlabel("estimated shift (kJ/mol)")
    plt.ylabel("number of configurations")
    plt.title(f"histogram of  shift for sm={sm}")
    plt.savefig(f"{output_dir}/hist_estimated_shift_{sm}.png")

    plt.cla()

    print(f"mean initial mae sm={sm}: {np.mean(np.array(initial_mae_all)[indices])}")
    print(f"std initial mae sm={np.std(np.array(initial_mae_all)[indices])}")
    print(f"mean shifted mae sm={sm}: {np.mean(np.array(shifted_mae_all)[indices])}")
    print(f"std shifted mae sm={np.std(np.array(shifted_mae_all)[indices])}")
# now let us plot histograms of mae values

print(f"mean initial mae sm={sm}: {np.mean(initial_mae_all)}")
print(f"std initial mae sm={np.std(initial_mae_all)}")
print(f"mean shifted mae sm={sm}: {np.mean(shifted_mae_all)}")
print(f"std shifted mae sm={np.std(shifted_mae_all)}")

plt.hist(initial_mae_all, bins=100)
plt.xlabel("initial MAE for trajectory")
plt.ylabel("number of configurations")
plt.title(f"initial MAE histogram")
plt.savefig(f"{output_dir}/hist_initial_mae.png")

plt.cla()

plt.hist(shifted_mae_all, bins=100)
plt.xlabel("shifted MAE for trajectory")
plt.ylabel("number of configurations")
plt.title(f"shifted MAE histogram")
plt.savefig(f"{output_dir}/hist_shifted_mae.png")
plt.cla()

plt.hist(estimated_shift_all, bins=100)
plt.xlabel("estimated shift for trajectory")
plt.ylabel("number of configurations")
plt.title(f"shift value")
plt.savefig(f"{output_dir}/hist_estimated_shift.png")
plt.cla()


#
# output_dir = "train"
# count = 0
# for idx in train_idx:
#
#     if count > 100:
#         break
#
#     record_name = record_names[idx]
#
#     dft_energies = {1: [], 3: [], 5: []}
#     model_energies = {1: [], 3: [], 5: []}
#
#     record = dataset.get_record(record_name)
#     atomic_numbers = record.get_property("atomic_numbers")
#
#     corrected_energy = record.get_property("dft_total_energy_corrected").value
#
#     spin_multiplicity = record.get_property("per_system_spin_multiplicity").value
#     total_charge = record.get_property("total_charge").value
#
#     diff_energies = {}
#     for sm in [1, 3, 5]:
#         diff_energies[sm] = []
#     for sm in [1, 3, 5]:
#         # we only want the configs where spin_multiplicity == sm
#         indices = spin_multiplicity == sm
#         for i in range(record.n_configs):
#             if indices[i]:
#                 dft_energies[sm].append(corrected_energy[i])
#
#                 modelforge_calculator = ModelForgeCalculator(
#                     potential,
#                     per_system_total_charge=int(total_charge[i][0]),
#                     per_system_spin_multiplicity=sm,
#                     # per_system_spin_multiplicity=int(spin_multiplicity[i][0]),
#                 )
#                 print(
#                     f"{record_name} : total charge: {total_charge[i][0]} sm: {sm} sm_A: {spin_multiplicity[i][0]}"
#                 )
#                 atoms = rdkit_mol_to_ase(record.to_rdkit(), index=i)
#                 atoms.calc = modelforge_calculator
#                 pe = atoms.get_potential_energy()
#
#                 model_energies[sm].append(pe * 96.485)  # convert eV to kJ/mol
#
#                 diff_energies[sm].append((pe * 96.485) - corrected_energy[i])
#         if len(dft_energies[sm]) == 0:
#             continue
#
#         # we want a metric that defines if a series is good
#
#         metric = np.sum(diff_energies[sm]) / len(dft_energies[sm])
#         print(metric)
#         # save the molecule as an rdkit image
#         rec = record.to_rdkit()
#         filename = f"{output_dir}/{record_name}_q{total_charge[i][0]}_image.png"
#
#         # save the rdkit molecule as an image
#         from rdkit import Chem
#         from rdkit.Chem import Draw
#
#         img = Draw.MolToImage(rec, size=(300, 300))
#
#         # 3. Save to a file
#         img.save(filename)
#
#         from matplotlib import pyplot as plt
#
#         plt.plot(
#             np.array(dft_energies[sm]),
#             linestyle="-",
#             marker="o",
#             label="DFT",
#             # label=f"MAE: {mae:.2f} kJ/mol",
#         )
#         plt.plot(
#             np.array(model_energies[sm]),
#             linestyle="-",
#             label="Model",
#             marker="x",
#         )
#         plt.title(
#             f"{record_name} sm: {sm} total_charge {total_charge[i][0]} mean diff {metric}"
#         )
#         plt.xlabel("config")
#         plt.ylabel("Energies (kJ/mol)")
#         plt.legend()
#         # need to set reasonable bounds to ignore any major outliers
#
#         # plt.legend()
#         plt.savefig(
#             f"{output_dir}/traj_tmqm_{record_name}_sm{sm}_Q{total_charge[i][0]}_ext.png"
#         )
#         plt.cla()
#
#         count = count + 1
