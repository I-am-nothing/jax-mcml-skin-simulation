import math
import multiprocessing as mp
import time

import numpy as np
from numba import njit
from structure.layers import Layer
from structure.photons import Photons


class SkinDataLoader:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        self.num_workers = config["number_workers"]
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]

        self.ph_num = config["number_of_photons"]
        self.ph_batch = config["photons_batch"]
        self.lambdas = config["spectrum"][2]

        self.size = math.ceil(self.lambdas * self.ph_num * len(self.dataset) / self.ph_batch)

        self.index_range = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.index_range)

        self.queue = mp.Queue(maxsize=2 * self.num_workers)
        self.stop_event = mp.Event()
        self.shared_index = mp.Value('i', 0)
        self.current_index = mp.Value('i', 0)
        self.workers = []

    def stop_loading(self):
        self.stop_event.set()
        for worker in self.workers:
            worker.terminate()
            worker.join()
        self.queue.close()

    def _next_index(self):
        with self.shared_index.get_lock():
            if self.shared_index.value < len(self):
                idx = self.shared_index.value
                self.shared_index.value += 1
                return idx
            else:
                return -1

    def _worker(self):
        batch = []
        while not self.stop_event.is_set():

            index = self._next_index()
            if index == -1:
                break
            batch.append(self.__getitem__(index))

            if len(batch) == self.batch_size:
                # while self.current_index.value != index:
                #     time.sleep(0.01)
                self.queue.put(batch)
                time.sleep(0.1)
                batch = []

            # self.current_index.value += 1

    def __next__(self):
        """Retrieve the next item in order from the queue."""
        if self.queue.empty() and self.current_index.value >= len(self):
            self.stop_loading()
            raise StopIteration

        while self.queue.empty():
            if self.stop_event.is_set():
                self.stop_loading()
                raise StopIteration
            time.sleep(0.01)

        return self.queue.get()

    def __iter__(self):
        self.shared_index.value = 0
        self.current_index.value = 0
        self.stop_event.clear()

        for _ in range(self.num_workers):
            worker = mp.Process(target=self._worker, args=())
            worker.start()
            self.workers.append(worker)

        return self

    def __len__(self):
        return self.size

    def __del__(self):
        try:
            self.stop_loading()
        except:
            pass

    def __getitem__(self, index):
        skin_idx = index * self.ph_batch / self.lambdas / self.ph_num

        ly_curr = (skin_idx - int(skin_idx)) * self.lambdas
        ph_curr = math.ceil((ly_curr - int(ly_curr)) * self.ph_num)

        skin_idx = int(skin_idx)
        ly_curr = int(ly_curr)

        if ph_curr == self.ph_num:
            ly_curr += 1
            ph_curr = 0
            if ly_curr == self.lambdas:
                skin_idx += 1
                ly_curr = 0

        ph_num = 0
        ly_num = 0
        id_curr = []
        ph = Photons(self.ph_batch, self.config)
        ly = []

        while ph_num < self.ph_batch:
            if skin_idx >= len(self.dataset):
                break

            idx, ly_next = self.dataset[skin_idx]

            ph_remain = self.ph_batch - ph_num
            ly_remain = ly_next.shape[0] - ly_curr

            ph_insert = min(ph_remain, ly_remain * self.ph_num - ph_curr)
            ly_idx = np.repeat(list(range(ly_num, ly_num + ly_remain)), self.ph_num)

            ph.idx[ph_num: ph_num + ph_insert] = len(id_curr)
            ph.ly_idx[ph_num: ph_num + ph_insert] = ly_idx[ph_curr:ph_curr + ph_insert]

            ly.append(ly_next[ly_curr: ly_curr + math.ceil(ph_insert / self.ph_num)])

            ph_num += ph_insert
            ly_num += ly_remain

            id_curr.append(idx)
            ph_curr = 0
            ly_curr = 0
            skin_idx += 1

        ly = Layer.cat(ly)

        return id_curr, ph, ly
