from typing import Literal

import torch
from diffsort import bitonic_network, odd_even_network
from diffsort.functional import execute_sort


class CustomDiffSortNet(torch.nn.Module):
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.
    Positional arguments:
    sorting_network_type -- which sorting network to use for sorting.
    vectors -- the matrix to sort along axis 1; sorted in-place
    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for sigmoid_phi interpolation (default 0.25)
    interpolation_type -- how to interpolate when swapping two numbers; supported: `logistic`, `logistic_phi`,
                 (default 'logistic_phi')
    """

    def __init__(
        self,
        sorting_network_type: Literal["odd_even", "bitonic"],
        size: int,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        interpolation_type: str = None,
        distribution: str = "cauchy",
    ):
        super().__init__()
        self.sorting_network_type = sorting_network_type
        self.size = size

        # Register the sorting network in the module buffer.
        self._sorting_network_structure = self._setup_sorting_network_structure(
            sorting_network_type, size
        )
        self._register_sorting_network(self._sorting_network_structure)

        if interpolation_type is not None:
            assert (
                distribution is None
                or distribution == "cauchy"
                or distribution == interpolation_type
            ), (
                "Two different distributions have been set (distribution={} and"
                " interpolation_type={}); however, they have the same interpretation and"
                " interpolation_type is a deprecated argument".format(
                    distribution, interpolation_type
                )
            )
            distribution = interpolation_type

        self.steepness = steepness
        self.art_lambda = art_lambda
        self.distribution = distribution

    def forward(self, vectors):
        assert len(vectors.shape) == 2
        assert vectors.shape[1] == self.size
        sorted_out, predicted_permutation = self.sort(
            vectors, self.steepness, self.art_lambda, self.distribution
        )
        return sorted_out, predicted_permutation

    def _setup_sorting_network_structure(self, network_type, n):
        """Setup the sorting network structure. Used for registering the sorting network in the module buffer.
        """

        def matrix_to_torch(m):
            return [[torch.from_numpy(matrix).float() for matrix in matrix_set] for matrix_set in m]

        if network_type == "bitonic":
            m = matrix_to_torch(bitonic_network(n))
        elif network_type == "odd_even":
            m = matrix_to_torch(odd_even_network(n))
        else:
            raise NotImplementedError(f"Sorting network `{network_type}` unknown.")

        return m

    def _register_sorting_network(self, m):
        """Register the sorting network in the module buffer."""
        for i, matrix_set in enumerate(m):
            for j, matrix in enumerate(matrix_set):
                self.register_buffer(f"sorting_network_{i}_{j}", matrix)

    def get_sorting_network(self):
        """Return the sorting network from the module buffer."""
        m = self._sorting_network_structure
        for i, _ in enumerate(m):
            yield (self.__getattr__(f"sorting_network_{i}_{j}") for j, _ in enumerate(m[i]))

    def sort(
        self,
        vectors: torch.Tensor,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = "cauchy",
    ):
        """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.

        Positional arguments:
        sorting_network
        vectors -- the matrix to sort along axis 1; sorted in-place

        Keyword arguments:
        steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
        art_lambda -- relevant for logistic_phi interpolation (default 0.25)
        distribution -- how to interpolate when swapping two numbers; (default 'cauchy')
        """
        assert self.sorting_network_0_0.device == vectors.device, (
            f"The sorting network is on device {self.sorting_network_0_0.device} while the vectors"
            f" are on device {vectors.device}, but they both need to be on the same device."
        )
        sorting_network = self.get_sorting_network()
        return execute_sort(
            sorting_network=sorting_network,
            vectors=vectors,
            steepness=steepness,
            art_lambda=art_lambda,
            distribution=distribution,
        )
