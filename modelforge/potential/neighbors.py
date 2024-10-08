import torch
from loguru import logger as log
from modelforge.potential.models import PairlistData
from modelforge.dataset.dataset import NNPInputTuple


class OrthogonalDisplacementFunction(torch.nn.Module):
    def __init__(self):
        """
        Compute displacement vectors between pairs of atoms, considering periodic boundary conditions if used.

        """
        super().__init__()

    def forward(
        self,
        coordinate_i: torch.Tensor,
        coordinate_j: torch.Tensor,
        box_vectors: torch.Tensor,
        is_periodic: torch.Tensor,
    ):
        """
        Compute displacement vectors and Euclidean distances between atom pairs.

        Parameters
        ----------
        coordinate_i : torch.Tensor
            Coordinates of the first atom in each pair. Shape: [n_pairs, 3].
        coordinate_j : torch.Tensor
            Coordinates of the second atom in each pair. Shape: [n_pairs, 3].
        box_vectors : torch.Tensor
            Box vectors defining the periodic boundary conditions. Shape: [3, 3].
        is_periodic : bool
            Whether to apply periodic boundary conditions.
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Displacement vectors (r_ij) of shape [n_pairs, 3] and distances (d_ij) of shape [n_pairs, 1].
        """
        r_ij = coordinate_i - coordinate_j

        if is_periodic == True:
            # Note, since box length may change, we need to update each time if periodic
            # reinitializing this vector each time does not have a significant performance impact

            box_lengths = torch.zeros(
                3, device=box_vectors.device, dtype=box_vectors.dtype
            )

            box_lengths[0] = box_vectors[0][0]
            box_lengths[1] = box_vectors[1][1]
            box_lengths[2] = box_vectors[2][2]

            r_ij = (
                torch.remainder(r_ij + box_lengths / 2, box_lengths) - box_lengths / 2
            )

        d_ij = torch.norm(r_ij, dim=1, keepdim=True, p=2)
        return r_ij, d_ij


class NeighborlistBruteNsq(torch.nn.Module):
    """
    Brute force N^2 neighbor list calculation for inference implemented fully in PyTorch.

    This is compatible with TorchScript.


    """

    def __init__(
        self,
        cutoff: float,
        displacement_function: OrthogonalDisplacementFunction,
        only_unique_pairs: bool = False,
    ):
        """
        Compute neighbor lists for inference, filtering pairs based on a cutoff distance.

        Parameters
        ----------
        cutoff : float
            The cutoff distance for neighbor list calculations.
        displacement_function : OrthogonalDisplacementFunction
            The function to calculate displacement vectors and distances between atom pairs, taking into account
            the specified boundary conditions.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by default False.
        """

        super().__init__()

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("only_unique_pairs", torch.tensor(only_unique_pairs))
        self.displacement_function = displacement_function

        self.indices = torch.tensor([])
        self.i_final_pairs = torch.tensor([])
        self.j_final_pairs = torch.tensor([])
        log.info("Initializing Brute Force N^2 Neighborlist")

    def _copy_to_nonunique(
        self,
        i_pairs: torch.Tensor,
        j_pairs: torch.Tensor,
        d_ij: torch.Tensor,
        r_ij: torch.Tensor,
        total_unique_pairs: int,
    ):

        r_ij_full = torch.zeros(
            total_unique_pairs * 2, 3, dtype=r_ij.dtype, device=r_ij.device
        )
        d_ij_full = torch.zeros(
            total_unique_pairs * 2, 1, dtype=d_ij.dtype, device=d_ij.device
        )

        r_ij_full[0:total_unique_pairs] = r_ij
        r_ij_full[total_unique_pairs : 2 * total_unique_pairs] = -r_ij

        d_ij_full[0:total_unique_pairs] = d_ij
        d_ij_full[total_unique_pairs : 2 * total_unique_pairs] = d_ij

        pairs_full = torch.zeros(
            2, total_unique_pairs * 2, dtype=torch.int64, device=i_pairs.device
        )

        pairs_full[0][0:total_unique_pairs] = i_pairs
        pairs_full[1][0:total_unique_pairs] = j_pairs
        pairs_full[0][total_unique_pairs : 2 * total_unique_pairs] = j_pairs
        pairs_full[1][total_unique_pairs : 2 * total_unique_pairs] = i_pairs

        return pairs_full, d_ij_full, r_ij_full

    def forward(self, data: NNPInputTuple):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInputTuple
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            Contains pair indices, distances (d_ij), and displacement vectors (r_ij) for atom pairs within the cutoff.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)

        # avoid reinitializing indices if they are already set and haven't changed
        if self.indices.shape[0] != n:
            # Generate a range of indices from 0 to n-1
            self.indices = torch.arange(n, device=atomic_subsystem_indices.device)

            # Create a meshgrid of indices
            self.i_final_pairs, self.j_final_pairs = torch.meshgrid(
                self.indices, self.indices, indexing="ij"
            )
            # We will only consider unique pairs; for non-unique pairs we can just appropriately copy
            # the data as it will be faster than extra computations.
            mask = self.i_final_pairs < self.j_final_pairs

            self.i_final_pairs = self.i_final_pairs[mask]
            self.j_final_pairs = self.j_final_pairs[mask]

        # calculate r_ij and d_ij
        r_ij, d_ij = self.displacement_function(
            positions[self.i_final_pairs],
            positions[self.j_final_pairs],
            data.box_vectors,
            data.is_periodic,
        )
        in_cutoff = (d_ij <= self.cutoff).squeeze()
        total_pairs = in_cutoff.sum()

        if self.only_unique_pairs:
            # using this instead of torch.stack to ensure that if we only have a single pair
            # we don't run into an issue with tensor shapes.
            # note this will fail if there are no interacting pairs

            pairs = torch.zeros(
                2, total_pairs, dtype=torch.int64, device=positions.device
            )

            pairs[0] = self.i_final_pairs[in_cutoff]
            pairs[1] = self.j_final_pairs[in_cutoff]

            return PairlistData(
                pair_indices=pairs,
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:

            pairs_full, d_ij_full, r_ij_full = self._copy_to_nonunique(
                self.i_final_pairs[in_cutoff],
                self.j_final_pairs[in_cutoff],
                d_ij[in_cutoff],
                r_ij[in_cutoff],
                total_pairs,
            )
            return PairlistData(
                pair_indices=pairs_full,
                d_ij=d_ij_full,
                r_ij=r_ij_full,
            )


class NeighborlistVerletNsq(torch.nn.Module):
    """
    Verlet neighbor list calculation for inference implemented fully in PyTorch.

    Rebuilding of the neighborlist uses an N^2 approach.  Rebuilding occurs when
    the maximum displacement of any particle exceeds half the skin distance.

    """

    def __init__(
        self,
        cutoff: float,
        skin: float,
        displacement_function: OrthogonalDisplacementFunction,
        only_unique_pairs: bool = False,
    ):
        """
        Compute neighbor lists for inference, filtering pairs based on a cutoff distance.

        Parameters
        ----------
        cutoff : float
            The cutoff distance for neighbor list calculations.
        skin : float
            The skin distance for neighbor list calculations.
        displacement_function : OrthogonalDisplacementFunction
            The function to calculate displacement vectors and distances between atom pairs, taking into account
            the specified boundary conditions.
        only_unique_pairs : bool, optional
            Whether to only use unique pairs in the pair list calculation, by
            default True. This should be set to True for all message passing
            networks.
        """

        super().__init__()

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.skin = skin
        self.half_skin = skin * 0.5
        self.cutoff_plus_skin = cutoff + skin
        self.only_unique_pairs = only_unique_pairs

        self.displacement_function = displacement_function
        self.indices = torch.tensor([])
        self.i_pairs = torch.tensor([])
        self.j_pairs = torch.tensor([])

        self.positions_old = torch.tensor([])
        self.nlist_pairs = torch.tensor([])
        self.builds = 0
        self.box_vectors = torch.zeros([3, 3])

        log.info("Initializing Verlet Neighborlist with N^2 building routine.")

    def _check_nlist(
        self, positions: torch.Tensor, box_vectors: torch.Tensor, is_periodic
    ):
        r_ij, d_ij = self.displacement_function(
            self.positions_old, positions, box_vectors, is_periodic
        )

        if torch.any(d_ij > self.half_skin):
            return True
        else:
            return False

    def _init_pairs(self, n_particles: int, device: torch.device):
        self.indices = torch.arange(n_particles, device=device)

        i_pairs, j_pairs = torch.meshgrid(
            self.indices,
            self.indices,
            indexing="ij",
        )

        mask = i_pairs < j_pairs
        self.i_pairs = i_pairs[mask]
        self.j_pairs = j_pairs[mask]

    def _build_nlist(
        self, positions: torch.Tensor, box_vectors: torch.Tensor, is_periodic
    ):
        r_ij, d_ij = self.displacement_function(
            positions[self.i_pairs], positions[self.j_pairs], box_vectors, is_periodic
        )

        in_cutoff = (d_ij < self.cutoff_plus_skin).squeeze()
        self.nlist_pairs = torch.stack(
            [self.i_pairs[in_cutoff], self.j_pairs[in_cutoff]]
        )
        self.builds += 1
        return r_ij[in_cutoff], d_ij[in_cutoff]

    def _copy_to_nonunique(
        self,
        pairs: torch.Tensor,
        d_ij: torch.Tensor,
        r_ij: torch.Tensor,
        total_unique_pairs: int,
    ):
        # this will allow us to copy the data for unique pairs to create the non-unique pairs data
        r_ij_full = torch.zeros(
            total_unique_pairs * 2, 3, dtype=r_ij.dtype, device=r_ij.device
        )
        d_ij_full = torch.zeros(
            total_unique_pairs * 2, 1, dtype=d_ij.dtype, device=d_ij.device
        )

        r_ij_full[0:total_unique_pairs] = r_ij

        # since we are swapping the order of the pairs, the sign changes
        r_ij_full[total_unique_pairs : 2 * total_unique_pairs] = -r_ij

        d_ij_full[0:total_unique_pairs] = d_ij
        d_ij_full[total_unique_pairs : 2 * total_unique_pairs] = d_ij

        pairs_full = torch.zeros(
            2, total_unique_pairs * 2, dtype=torch.int64, device=pairs.device
        )

        pairs_full[0][0:total_unique_pairs] = pairs[0]
        pairs_full[1][0:total_unique_pairs] = pairs[1]
        pairs_full[0][total_unique_pairs : 2 * total_unique_pairs] = pairs[1]
        pairs_full[1][total_unique_pairs : 2 * total_unique_pairs] = pairs[0]

        return pairs_full, d_ij_full, r_ij_full

    def forward(self, data: NNPInputTuple):
        """
        Prepares the input tensors for passing to the model.

        This method handles general input manipulation, such as calculating
        distances and generating the pair list. It also calls the model-specific
        input preparation.

        Parameters
        ----------
        data : NNPInputTuple
            The input data provided by the dataset, containing atomic numbers,
            positions, and other necessary information.

        Returns
        -------
        PairListOutputs
            Contains pair indices, distances (d_ij), and displacement vectors (r_ij) for atom pairs within the cutoff.
        """
        # ---------------------------
        # general input manipulation
        positions = data.positions
        atomic_subsystem_indices = data.atomic_subsystem_indices

        n = atomic_subsystem_indices.size(0)
        # if the initial build we haven't yet set box vectors so set them
        # this is necessary because we need to store them to know if we need to force a rebuild
        # because the box vectors have changed
        if self.builds == 0:
            self.box_vectors = data.box_vectors

        box_changed = torch.any(self.box_vectors != data.box_vectors)

        # avoid reinitializing indices if they are already set and haven't changed
        if self.indices.shape[0] != n:
            self.box_vectors = data.box_vectors
            self.positions_old = positions
            self._init_pairs(n, positions.device)
            r_ij, d_ij = self._build_nlist(
                positions, data.box_vectors, data.is_periodic
            )
        elif box_changed:
            # if the box vectors have changed, we need to rebuild the nlist
            # but do not need to regenerate the pairs
            self.box_vectors = data.box_vectors
            self.positions_old = positions
            r_ij, d_ij = self._build_nlist(
                positions, data.box_vectors, data.is_periodic
            )
        elif self._check_nlist(positions, data.box_vectors, data.is_periodic):
            self.positions_old = positions
            r_ij, d_ij = self._build_nlist(
                positions, data.box_vectors, data.is_periodic
            )
        else:
            r_ij, d_ij = self.displacement_function(
                positions[self.nlist_pairs[0]],
                positions[self.nlist_pairs[1]],
                data.box_vectors,
                data.is_periodic,
            )

        in_cutoff = (d_ij <= self.cutoff).squeeze()
        total_pairs = in_cutoff.sum()

        if self.only_unique_pairs:
            # using this instead of torch.stack to ensure that if we only have a single pair
            # we don't run into an issue with shapes.

            pairs = torch.zeros(
                2, total_pairs, dtype=torch.int64, device=positions.device
            )

            pairs[0] = self.nlist_pairs[0][in_cutoff]
            pairs[1] = self.nlist_pairs[1][in_cutoff]

            return PairlistData(
                pair_indices=pairs,
                d_ij=d_ij[in_cutoff],
                r_ij=r_ij[in_cutoff],
            )

        else:

            pairs_full, d_ij_full, r_ij_full = self._copy_to_nonunique(
                self.nlist_pairs[:, in_cutoff],
                d_ij[in_cutoff],
                r_ij[in_cutoff],
                total_pairs,
            )
            return PairlistData(
                pair_indices=pairs_full,
                d_ij=d_ij_full,
                r_ij=r_ij_full,
            )
