#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import logging
import numpy as np
from operator import itemgetter
from pathlib import Path
from espnet2.samplers.abs_sampler import AbsSampler
import random
from espnet2.fileio.read_text import read_2column_text
def get_lenth_id(shapefile,loader_type="csv_int"):
    if loader_type == "text_int":
        delimiter = " "
        dtype = int
    elif loader_type == "text_float":
        delimiter = " "
        dtype = float
    elif loader_type == "csv_int":
        delimiter = ","
        dtype = int
    elif loader_type == "csv_float":
        delimiter = ","
        dtype = float
    else:
        raise ValueError(f"Not supported loader_type={loader_type}")
    uids = []
    lengths_list = []
    with Path(shapefile).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            uids.append(sps[0])
            lengths_list.append(int(sps[1].split(delimiter)[0]))
    return uids,lengths_list

def write_log(content, logger=None, level=None, **other_params):
    logging.info(content)

class MultiEpochDynamicBatchMultiSampler(AbsSampler):

    def __init__(self, lengths_list, batch_size, max_batch_size=None,min_batch_size=0, 
                 dynamic=True, bucket_length_multiplier=1.1, bucket_shuffle=True, 
                 bucket_internal_shuffle=True, bucket_boundaries=None, seed=42, 
                 batch_start=0, logger=None,max_sample_len=None,min_sample_len=None):
        
        if isinstance(lengths_list,str):
            lengths_list = [ float(value) for value in read_2column_text(lengths_list).values() ]

        #ngpu batchnum >= world_size
        self.min_batch_size = min_batch_size
        logging.info(f"min_batch_size == {min_batch_size}")
        # only first epoch could start from batch_start
        self.first_epoch = True

        # filter too long or too short uttrance
        left_bucket_length = min_sample_len if min_sample_len else min(lengths_list)
        max_length = max_sample_len if max_sample_len else max(lengths_list) 
        total_sample_num = len(lengths_list)
        assert max_length > left_bucket_length
        # generate boundaries for every bucket
        bucket_boundaries = self.get_data_boundaries(max_length=max_length, 
                                                bucket_boundaries=bucket_boundaries, 
                                                left_bucket_length=left_bucket_length, 
                                                bucket_length_multiplier=bucket_length_multiplier) 
        self._bucket_boundaries = np.array(bucket_boundaries)
        bucket_num = self._bucket_boundaries.shape[0] - 1
        
        # put samples into the bucket, some sample might be dropped if max_sample_len or min_sample_len !=none
        sample_point_sum = 0
        drop_num = 0
        self._bucket_samples = [[] for _ in range(bucket_num)]
        for idx in range(total_sample_num):
            item_len = lengths_list[idx]
            if item_len > max_length or item_len < left_bucket_length:
                drop_num += 1
                continue
            sample_point_sum +=  item_len
            bucket_id = max(np.searchsorted(self._bucket_boundaries, item_len) - 1, 0)
            self._bucket_samples[bucket_id].append(idx)
           

        # calculate bucket bucket_sample_num, bucket batch size and bucket batch number
        bucket_sample_num = []
        self._bucket_batch_sizes = []
        bucket_batch_num = []
        for i in range(bucket_num):
            bucket_sample_num.append(len(self._bucket_samples[i]))
            if dynamic:
                raw_batch_size = max(int(round(batch_size * max_length / self._bucket_boundaries[i+1])), 1)
                if max_batch_size is not None:
                    raw_batch_size = min(raw_batch_size, max_batch_size)
            else:
                raw_batch_size = batch_size
            self._bucket_batch_sizes.append(raw_batch_size)
            bucket_batch_num.append(np.ceil(len(self._bucket_samples[i]) / float(raw_batch_size)))
        epoch_batch_num = np.sum(bucket_batch_num)

        self._seed = seed
        self._bucket_shuffle = bucket_shuffle
        self._bucket_internal_shuffle = bucket_internal_shuffle
        self.begin = batch_start
        self.num_per_epoch = epoch_batch_num

        assert batch_start <= epoch_batch_num 
        write_log(content='MultiEpochDynamicBatchMultiSampler: Total {} samples. Used {} buckets. Max_len {} s, Min_len {} s.\n Left {} samples, {} hours (Only for old Espnet)'.format(
                                total_sample_num, bucket_num, max_length/16000, left_bucket_length/16000, total_sample_num-drop_num, sample_point_sum/(16000*3600), ), 
                logger=logger, level='info')
        boundary_left = np.around(self._bucket_boundaries[0], 2)
        for i in range(bucket_num):
            boundary_right = np.around(self._bucket_boundaries[i+1], 2)
            write_log(content='MultiEpochDynamicBatchMultiSampler: Bucket {} with boundary {}s-{}s and batch_size {} has {} batches {} samples which counts {}%'.format(
                      i, boundary_left/16000, boundary_right/16000, self._bucket_batch_sizes[i], bucket_batch_num[i], bucket_sample_num[i], round(bucket_sample_num[i]/(total_sample_num-drop_num)*100,2)), logger=logger, level='info')
            boundary_left = boundary_right          
 
    def generate(self, epoch_idx): # only bucket shuffle   
        epoch_batches = []
        if self._bucket_shuffle or self._bucket_internal_shuffle:
            g = torch.Generator()
            g.manual_seed(self._seed + epoch_idx)
        else:
            g = None
        if self._bucket_shuffle:
            bucket_index = torch.randperm(len(self._bucket_samples), generator=g).tolist()
        else:
            bucket_index = range(len(self._bucket_samples))
        
        for bucket_idx in bucket_index:
            bucket_internal_samples = torch.tensor(self._bucket_samples[bucket_idx])
            bucket_batch_size = self._bucket_batch_sizes[bucket_idx]
            bucket_internal_multisamples = []
            bucket_internal_multisamples.append(bucket_internal_samples)
            bucket_internal_multisamples = torch.stack(bucket_internal_multisamples, dim=-1).split(bucket_batch_size, dim=0)

            for batch_indexes in bucket_internal_multisamples:
                batch_indexes_list = batch_indexes.squeeze(-1).tolist()
                if len(batch_indexes_list) >= self.min_batch_size:
                    epoch_batches.append(batch_indexes_list)
        if self.first_epoch and epoch_idx == 0: #if resume start from 0 batch
            self._batches = epoch_batches[self.begin:]
            logging.info(f"MultiEpochDynamicBatchMultiSampler: {epoch_idx} epoch create {len(epoch_batches)} batches, start from Num.{self.begin} batches")
        else:
            self._batches = epoch_batches
            logging.info(f"MultiEpochDynamicBatchMultiSampler: {epoch_idx} epoch create {len(epoch_batches)} batches, start from Num.0 batches")
        self.first_epoch = False

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return int(self.num_per_epoch)
    
    def get_data_boundaries(self, max_length, bucket_boundaries, left_bucket_length, bucket_length_multiplier):
        if not bucket_boundaries:
            if left_bucket_length <= 0:
                raise ValueError('left_bucket_length must be >0 if no bucket_boundaries set')
            if bucket_length_multiplier < 1.0:
                raise ValueError('bucket_length_multiplier must be >1.0 if no bucket_boundaries '
                                'set')
            bucket_boundaries = [left_bucket_length]
            bucket_boundary = float(left_bucket_length)
            while True:
                bucket_boundary *= bucket_length_multiplier
                if bucket_boundary >= max_length:
                    break
                bucket_boundaries.append(bucket_boundary)
            bucket_boundaries.append(max_length)
        else:
            if not all([x >= 1 for x in bucket_boundaries]):
                raise ValueError('All elements in bucket boundaries should be >= 1.')
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError('Bucket_boundaries should not contain duplicates.')
        return bucket_boundaries
       
class DynamicBucketBatchSampler(AbsSampler):
   
    def __init__(self, lengths_list, batch_size, max_batch_size=None, mini_batch_size=1,dynamic=True, bucket_length_multiplier=1.1,
                 shuffle=True, bucket_boundaries=None,drop_last=False, batch_start=0, logger=None,seed=3):

        if max_batch_size:
            assert max_batch_size >= mini_batch_size
        self.seed = seed
        self.begin_batch_id = batch_start
        self.mini_batch_size = mini_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size = batch_size
        self.lengths_list = lengths_list
        left_bucket_length = min(lengths_list)
        max_length = max(lengths_list)
        
        # take length of examples from this argument and bypass length_key
        self._ex_lengths = {}
        for index in range(len(lengths_list)):
            self._ex_lengths[str(index)] = lengths_list[index]

        if bucket_boundaries is not None:
            if not all([x >= 1 for x in bucket_boundaries]):
                raise ValueError('All elements in bucket boundaries should be >= 1.')
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError('Bucket_boundaries should not contain duplicates.')
        
        # generate boundaries for every bucket
        self._bucket_boundaries = np.array(
            self.get_data_boundaries(
                max_length=max_length, bucket_boundaries=bucket_boundaries,
                left_bucket_length=left_bucket_length, bucket_length_multiplier=bucket_length_multiplier)
                )

        
        # calculate bucket bucket_sample_num, bucket batch size and bucket batch number
        self.bucket_num = self._bucket_boundaries.shape[0] - 1
        self._shuffle = shuffle
        self.logger = logger
        self._drop_last = drop_last
        self._bucket_lens = []

        for i in range(1, len(self._bucket_boundaries)):
            if dynamic:
                if max_batch_size is not None:
                    self._bucket_lens.append(
                        np.clip(int(batch_size*max_length / self._bucket_boundaries[i]), mini_batch_size, max_batch_size))
                else:
                    self._bucket_lens.append(max(int(batch_size * max_length / self._bucket_boundaries[i]), mini_batch_size))
            else:
                self._bucket_lens.append(batch_size)
        self.first_init = True
        
    def _batches_change(self,batches):
        return batches

    def _generate_batches(self, epoch_num):

        if self._shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(epoch_num + self.seed)
            sampler = torch.randperm(len(self._ex_lengths), generator=g).tolist()  # type: ignore
        else:
            sampler = range(len(self._ex_lengths))  # type: ignore

        self._batches = []
        bucket_batches = [[] for _ in self._bucket_lens]
        bucket_stats = [0 for _ in self._bucket_lens]
        for idx in sampler:
            item_len = self._ex_lengths[str(idx)]
            bucket_id = max(np.searchsorted(self._bucket_boundaries, item_len) - 1, 0)
            bucket_batches[bucket_id].append(idx)
            bucket_stats[bucket_id] += 1
            if len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]:
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
        # Dump remaining batches - we might even want to shuffle those
        if not self._drop_last:
            for batch in bucket_batches:
                if batch:
                    if len(batch) >= self.mini_batch_size:
                        self._batches.append(batch)
        if self.first_init:  # only log at first epoch
            self.log_status(bucket_stats)
            self.first_init = False
        self._batches = self._batches[self.begin_batch_id:]
        self._batches = self._batches_change(self._batches)
    
    def log_status(self,bucket_stats):
        write_log(
                content='DynamicBatchSampler: Created {} batches, {} buckets used, begin at {} batchid.'.format(
                    len(self._batches), sum(np.array(bucket_stats) != 0), self.begin_batch_id), logger=self.logger, level='info')
        bucket_left_boundary = self._bucket_boundaries[0]
        bucket_id = 0
        for i in range(1, len(self._bucket_boundaries)):
            if bucket_stats[i-1] != 0:
                write_log(
                    content='DynamicBatchSampler: Bucket {} with boundary {}-{} and batch_size {} has {} '
                            'examples.'.format(bucket_id, np.around(bucket_left_boundary, 2),
                                                np.around(self._bucket_boundaries[i], 2),
                                                self._bucket_lens[i-1], bucket_stats[i-1]),
                    logger=self.logger, level='info')
                bucket_left_boundary = self._bucket_boundaries[i]
                bucket_id = bucket_id + 1
    
    def generate(self,epoch_num):
        self._generate_batches(epoch_num)

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"N-bucket={self.bucket_num},"
            f"batch_size={self.batch_size}, "
            f"max_batch_size={self.max_batch_size},"
            f"shape_file={self.shape_file}, "
        )

    def __len__(self):
        return len(self._batches)

    @staticmethod
    def get_data_boundaries(max_length, bucket_boundaries, left_bucket_length, bucket_length_multiplier):
        if not bucket_boundaries:
            if left_bucket_length <= 0:
                raise ValueError('left_bucket_length must be >0 if no bucket_boundaries set')
            if bucket_length_multiplier < 1.0:
                raise ValueError('bucket_length_multiplier must be >1.0 if no bucket_boundaries set')
            bucket_boundaries = {left_bucket_length}
            bucket_boundary = float(left_bucket_length)
            while True:
                bucket_boundary *= bucket_length_multiplier
                if bucket_boundary >= max_length:
                    break
                bucket_boundaries.add(bucket_boundary)
            bucket_boundaries.add(max_length)
        return list(sorted(bucket_boundaries))

 
if __name__ == "__main__":
    from espnet2.fileio.read_text import read_2column_text
    import argparse
    parser = argparse.ArgumentParser("dataset statistic")
    parser.add_argument( "--shape_files", type=str, nargs="+",default=None, help="")
    parser.add_argument( "--min_max_time", type=str, default="0.1:20", help="min 0.1 max 20")
    parser.add_argument( "--bucket_args", type=str, default="0.1:20:1", help="min 0.1 max 20 interval 1")
    args = parser.parse_args()
    shape_files = args.shape_files
    batch_size = 5
    max_batch_size = 32
    logging.basicConfig(level=logging.INFO)
    bucket_boundaries =  list(range(int(float(args.bucket_args.split(":")[0])*16000),int(float(args.bucket_args.split(":")[1])*16000),int(float(args.bucket_args.split(":")[2])*16000)))
    for shape_file in shape_files:
        lengths_list = [ float(value) for value in read_2column_text(shape_file).values() ]
        sampler = MultiEpochDynamicBatchMultiSampler(lengths_list=lengths_list, bucket_boundaries=bucket_boundaries,\
                                                    batch_size=batch_size, max_batch_size=max_batch_size, \
                                                    min_sample_len=16000*float(args.min_max_time.split(":")[0]),max_sample_len=16000*float(args.min_max_time.split(":")[1]))
  