import os
import numpy as np
import torch
import random
import itertools
from glob import glob
from PIL import Image
from scipy import io, misc
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader, _DataLoaderIter
from torch.utils.data.sampler import Sampler, SequentialSampler, \
				     RandomSampler


class DatasetPair(Dataset):
    def __init__(self, cover_dir, stego_dir, transform = None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.cover_list = [x.split('/')[-1] for x in glob(cover_dir + '/*')]
        self.transform = transform
        assert len(self.cover_list) !=0, 'cover_dir is empty'

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        labels = np.array([0,1], dtype = 'int32')
        cover_path = os.path.join(self.cover_dir, self.cover_list[idx])
        cover = Image.open(cover_path)
        images = np.empty((2, cover.size[0], cover.size[1],1),\
                          dtype = 'uint8')
        images[0,:,:,0] = np.array(cover)
        stego_path = os.path.join(self.stego_dir,self.cover_list[idx])
        images[1,:,:,0] = misc.imread(stego_path)
        samples = {'images': images, 'labels': labels}
        if self.transform:
            samples = self.transform(samples)
        return samples


class DataLoaderIterWithReshape(_DataLoaderIter):
    def next(self):
        if self.num_workers == 0:
            indices = next(self.sample_iter)
            batch = self._reshape(self.collate_fn(\
                                  [self.dataset[i] for i in indices]))
        if self.pin_memory:
            batch = pin_memory_batch(batch)
        return batch

        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._reshape(self._process_next_batch(batch))
        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                self.reorder_dict[idx] = batch
                continue
            return self._reshape(self._process_next_batch(batch))

    def _reshape(self, batch):
        images, labels = batch['images'], batch['labels']
        shape = list(images.size())
        return {'images': images.view(shape[0] * shape[1], * shape[2:]),\
       		'labels': labels.view(-1)}

class DataLoaderStego(DataLoader):
    def __init__(self, cover_dir, stego_dir, shuffle = False,\
                 batch_size = 1, transform = None, num_workers = 0,\
                 pin_memory = False):
        dataset = DatasetPair(cover_dir, stego_dir,transform)
        _batch_size = int(batch_size / 2)
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        super(DataLoaderStego, self).__init__(dataset, _batch_size, None, \
              sampler,None, num_workers, pin_memory = pin_memory,\
              drop_last=True)
        self.shuffle = shuffle


    def __iter__(self):
        return DataLoaderIterWithReshape(self)

class ToTensor(object):
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']
        images = images.transpose((0,3,1,2))
        return {'images': torch.from_numpy(images),\
                'labels': torch.from_numpy(labels).long()}

class RandomRot(object):
    def __call__(self, samples):
        images = samples['images']
        rot = random.randint(0,3)
        return {'images': np.rot90(images, rot, axes = [1,2]).copy(),\
                'labels': samples['labels']}

class RandomFlip(object):
    def __call__(self, samples):
        if random.random() < 0.5:
            images = samples['images']
            return {'images': np.flip(images, axis=2).copy(),\
	    	    'labels': samples['labels']}
        else:
            return samples




