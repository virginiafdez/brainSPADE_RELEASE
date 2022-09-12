import monai
import numpy as np
import nibabel as nib
from monai.transforms import MapTransform


class LoadNPZ(MapTransform):
    """
    Based on MONAI function LoadImageD, instances MapTransform and loads NPZs files.
    """

    def __init__(
        self,
        keys: dict,
        npz_keys: dict,
    ) -> None:
        """
        :param keys: string containing the keys to be returned.
        """
        super().__init__(keys, allow_missing_keys=True)
        self.keys = keys
        self.npz_keys = npz_keys
        if len(self.keys) != len(self.npz_keys):
            ValueError("Different lengths for npz keys and keys.")

    def __call__(self, data):
        """
        """
        d = dict(data)

        for d_ind, d_name in enumerate(self.keys):
            if d_name in d.keys():
                file = np.load(d[d_name], allow_pickle=True)
                d[d_name] = np.asarray(file[self.npz_keys[d_ind]])
        return d


