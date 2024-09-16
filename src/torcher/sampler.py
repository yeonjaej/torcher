"""Used to define which dataset entries to load at each iteration"""

import time
import numpy as np
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class AbstractBatchSampler(Sampler):
    """Abstract sampler class.

    Samplers that inherit from this class should work out of the box.
    Just define the __len__ and __iter__ functions. __init__ defines
    self.num_samples and self.batch_size as well as a self._random
    RNG, if needed.
    """

    def __init__(self, dataset, seed=None):
        """Check and store the values passed to the initializer,
        set the seeds appropriately.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to sampler from
        seed : int, optional
            Seed to use for random sampling
        """
        # Initialize parent class
        super().__init__()
        self._dataset = dataset
        # Initialize the random number generator with a seed
        if seed is None:
            seed = int(time.time())
        else:
            assert isinstance(seed, int), (
                    f"The sampler seed must be an integer, got: {seed}.")

        self._random = np.random.RandomState(seed=seed) # pylint: disable=E1101



    def __len__(self):
        """Provides the full length of the sampler.

        The length of the sampler can differ from the number of elements in
        the underlying dataset, if the last batch is smaller than the requested
        size and is dropped.

        Returns
        -------
        int
            Total number of entries to sample
        """
        return len(self._dataset)

    def __iter__(self):
        """Placeholder to be overridden by children classes."""
        raise NotImplementedError

