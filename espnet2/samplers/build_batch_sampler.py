from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from typeguard import check_argument_types
from typeguard import check_return_type
from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler
from espnet2.samplers.length_batch_sampler import LengthBatchSampler
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.samplers.sorted_batch_sampler import SortedBatchSampler
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.samplers.bucket_batch_sampler import MultiEpochDynamicBatchMultiSampler
import logging

BATCH_TYPES = dict(
    unsorted="UnsortedBatchSampler has nothing in particular feature and "
    "just creates mini-batches which has constant batch_size. "
    "This sampler doesn't require any length "
    "information for each feature. "
    "'key_file' is just a text file which describes each sample name."
    "\n\n"
    "    utterance_id_a\n"
    "    utterance_id_b\n"
    "    utterance_id_c\n"
    "\n"
    "The fist column is referred, so 'shape file' can be used, too.\n\n"
    "    utterance_id_a 100,80\n"
    "    utterance_id_b 400,80\n"
    "    utterance_id_c 512,80\n",
    sorted="SortedBatchSampler sorts samples by the length of the first input "
    " in order to make each sample in a mini-batch has close length. "
    "This sampler requires a text file which describes the length for each sample "
    "\n\n"
    "    utterance_id_a 1000\n"
    "    utterance_id_b 1453\n"
    "    utterance_id_c 1241\n"
    "\n"
    "The first element of feature dimensions is referred, "
    "so 'shape_file' can be also used.\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
    folded="FoldedBatchSampler supports variable batch_size. "
    "The batch_size is decided by\n"
    "    batch_size = base_batch_size // (L // fold_length)\n"
    "L is referred to the largest length of samples in the mini-batch. "
    "This samples requires length information as same as SortedBatchSampler\n",
    length="LengthBatchSampler supports variable batch_size. "
    "This sampler makes mini-batches which have same number of 'bins' as possible "
    "counting by the total lengths of each feature in the mini-batch. "
    "This sampler requires a text file which describes the length for each sample. "
    "\n\n"
    "    utterance_id_a 1000\n"
    "    utterance_id_b 1453\n"
    "    utterance_id_c 1241\n"
    "\n"
    "The first element of feature dimensions is referred, "
    "so 'shape_file' can be also used.\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
    numel="NumElementsBatchSampler supports variable batch_size. "
    "Just like LengthBatchSampler, this sampler makes mini-batches"
    " which have same number of 'bins' as possible "
    "counting by the total number of elements of each feature "
    "instead of the length. "
    "Thus this sampler requires the full information of the dimension of the features. "
    "\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",

    bucket="MultiEpochDynamicBatchMultiSampler supports variable batch_size"
    "This BatchSampler batches examples together by grouping them by their length."
    "Every example in the batch have approximate the same length and thus padding is minimized."
     "This enables faster training on datasets where length of examples can vary significantly."
    "Inspired by: https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length."
    "This sampler requires a text file which describes the length for each sample. "
    "\n\n"
    "    utterance_id_a 1000\n"
    "    utterance_id_b 1453\n"
    "    utterance_id_c 1241\n"
    "\n"
    "The first element of feature dimensions is referred, "
    "so 'shape_file' can be also used.\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
)


def build_batch_sampler(
    type: str,
    batch_size: int,
    batch_bins: int,
    shape_files: Union[Tuple[str, ...], List[str]],
    sort_in_batch: str = "descending",
    sort_batch: str = "ascending",
    drop_last: bool = False,
    min_batch_size: int = 1,
    fold_lengths: Sequence[int] = (),
    padding: bool = True,
    utt2category_file: str = None,
    max_batch_size: int = 32,
    bucket_length_multiplier : float=1.1,
    bucket_boundaries: list= None,
    bucket_bsize_dynamic:bool=True,
    bucket_batch_size:int=2,
    seed:int=3,
    batch_start:int=0,
    addfixes:dict=None,
    max_sample_len=None,
    min_sample_len=None,
) -> AbsSampler:
    """Helper function to instantiate BatchSampler.

    Args:
        type: mini-batch type. "unsorted", "sorted", "folded", "numel", or, "length"
        batch_size: The mini-batch size. Used for "unsorted", "sorted", "folded" mode
        batch_bins: Used for "numel" model
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        sort_in_batch:
        sort_batch:
        drop_last:
        min_batch_size:  Used for "numel" or "folded" mode
        fold_lengths: Used for "folded" mode
        padding: Whether sequences are input as a padded tensor or not.
            used for "numel" mode
        max_batch_size: used for "bucket" mode
        bucket_length_multiplier: used for "bucket" mode
        bucket_boundaries: used for "bucket" mode
    """
    assert check_argument_types()
    if len(shape_files) == 0:
        raise ValueError("No shape file are given")

    if type == "unsorted":
        retval = UnsortedBatchSampler(
            batch_size=batch_size, key_file=shape_files[0], drop_last=drop_last
        )

    elif type == "sorted":
        retval = SortedBatchSampler(
            batch_size=batch_size,
            shape_file=shape_files[0],
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
        )

    elif type == "folded":
        if len(fold_lengths) != len(shape_files):
            raise ValueError(
                f"The number of fold_lengths must be equal to "
                f"the number of shape_files: "
                f"{len(fold_lengths)} != {len(shape_files)}"
            )
        retval = FoldedBatchSampler(
            batch_size=batch_size,
            shape_files=shape_files,
            fold_lengths=fold_lengths,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            min_batch_size=min_batch_size,
            utt2category_file=utt2category_file,
        )

    elif type == "numel":
        retval = NumElementsBatchSampler(
            batch_bins=batch_bins,
            shape_files=shape_files,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            padding=padding,
            min_batch_size=min_batch_size,
        )

    elif type == "length":
        retval = LengthBatchSampler(
            batch_bins=batch_bins,
            shape_files=shape_files,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            padding=padding,
            min_batch_size=min_batch_size,
        )

    elif type == "bucket":
        logging.info(f"bucket_min_batch_size == {min_batch_size}")
        retval = MultiEpochDynamicBatchMultiSampler(
                lengths_list=shape_files[0], 
                batch_size=bucket_batch_size, 
                max_batch_size=max_batch_size, 
                bucket_length_multiplier=bucket_length_multiplier, 
                bucket_boundaries=bucket_boundaries, 
                dynamic=bucket_bsize_dynamic,
                seed=seed,
                batch_start=batch_start,
                max_sample_len=max_sample_len,
                min_sample_len=min_sample_len,
                min_batch_size=min_batch_size
                )
    
    else:
        raise ValueError(f"Not supported: {type}")
    assert check_return_type(retval)
    return retval