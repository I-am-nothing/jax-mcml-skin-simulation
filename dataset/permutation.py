from functools import reduce

import numpy as np
import os


class PermutationDataset:
    def __init__(self, config):

        self.float_dtype = np.float32 if config["fp_16"] is None or config["fp_16"] == False else np.float16
        self.cold_start = config["cold_start"]
        self.file_path = config["chrom_permutation_path"]

        self.params = np.array(config["chrom_parameters"], dtype=self.float_dtype)
        self.params = self.params.reshape(-1, 3)
        self.perms = self.params[:, 2].astype(np.int16).tolist()

        self.step = (self.params[:, 1] - self.params[:, 0]) \
                    / np.where(self.params[:, 2] == 1, self.params[:, 2], self.params[:, 2] - 1.)

        if os.path.isfile(self.file_path) and self.cold_start:
            print("loading permutations... ", end="")
            self.data = np.load(self.file_path)
            print("DONE")
            np.save("data/skin_perm_01.npy", self.data[:1000000])
        else:
            self.data = self.__generate_chrom_permutations()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        return data[0].item(), (data[1:].astype(self.float_dtype) * self.step + self.params[:, 0]).reshape(3, 8)


    def __generate_chrom_permutations(self):
        print("generating chrom permutations...")
        total = reduce(lambda x, y: x * y, self.perms)

        perms = [i - 1 for i in self.perms]
        tensor = np.zeros((total, len(perms) + 1), dtype=np.int64)

        count = 0
        while count < total:
            print("\rprocessing ", f"{count}/{total}", f"{count / total * 100:.2f}%", end='')
            tensor[count] = np.array([count] + perms, dtype=np.int64)
            xi = 1
            for i, num in enumerate(self.perms):
                perms[i] = perms[i] - xi
                if perms[i] < 0:
                    xi = 1
                    perms[i] = num - 1
                else:
                    break
            count += 1

        print("\nsaving chrom permutations...")
        np.save(self.file_path, tensor)

        return tensor

if __name__ == "__main__":
    config = {
        "fp_16": False,
        "cold_start": True,

        "chrom_permutation_path": "../data/skin_perm.npy",
        "chrom_parameters": [
            [0.0000, 0.0000, 1],
            [0.0000, 0.0000, 1],
            [0.0195, 0.0198, 3],
            [0.0000, 0.0000, 1],
            [0.0000, 0.0000, 1],
            [0.0002, 0.0005, 3],
            [0.1829, 0.1850, 3],
            [0.6231, 0.6498, 3],
            [0.5915, 0.6837, 3],
            [0.0974, 0.0997, 3],
            [0.0974, 0.0997, 3],
            [0.1948, 0.1995, 3],
            [0.0974, 0.0997, 3],
            [0.0300, 0.0300, 1],
            [0.0000, 0.0000, 1],
            [0.8768, 0.8976, 3],
            [0.7979, 0.7794, 3],
            [0.8768, 0.8976, 3],
            [0.0000, 0.0000, 1],
            [0.0020, 0.1000, 3],
            [0.0974, 0.0997, 3],
            [1.4613, 1.4960, 3],
            [1.2665, 1.2965, 3],
            [1.3639, 1.3963, 3]
        ]
    }
    dataset = PermutationDataset(config)

    for i in range(10):
        xdd = dataset[i]
        print()
