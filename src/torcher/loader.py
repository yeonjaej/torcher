import numpy as np

class RAMLoader:
    def __init__(self, dataset, batch_size=1, drop_last=False, shuffle=False):
        self.data  = dataset
        self.shuffle = shuffle
        self.datasize = len(self.data)
        self.idx_v = np.arange(self.datasize)
        self.bsize = batch_size
        self.iter_ctr = int(self.datasize/self.bsize)
        if self.iter_ctr*self.bsize < len(self.data) and not drop_last:
            self.iter_ctr += 1

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.idx_v)
        for idx in range(self.iter_ctr):
            start = idx*self.bsize
            end   = (idx+1)*self.bsize
            if end > self.datasize:
                end = self.datasize
            bidx_v = self.idx_v[start:end]
            yield self.data[bidx_v]
            #yield {key:value[bidx_v] for key,value in self.data.items()}

    def __len__(self):
        return self.iter_ctr

    @property
    def batch_size(self):
        return self.bsize
    