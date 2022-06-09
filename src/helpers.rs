#![allow(dead_code)]
use std::mem;

use succinct::storage::BlockType;
use succinct::IntVector;

/// Our own version of `num_traits::Zero` so we don't need to depend on it just for the two types.
pub(crate) trait NumZero {
    fn create_zero() -> Self;
}

impl NumZero for usize {
    fn create_zero() -> Self {
        0
    }
}

impl NumZero for u64 {
    fn create_zero() -> Self {
        0
    }
}

/// Quicker all-zero initialization of an IntVector.
pub(crate) fn all_zero_intvector<T>(element_bits: usize, len: usize) -> IntVector<T>
where
    T: BlockType + NumZero,
{
    let n_blocks = {
        let bits = mem::size_of::<T>()
            .checked_mul(8)
            .expect("Table size too large")
            .checked_mul(len)
            .expect("Table size too large");
        let blocks = bits / element_bits;
        let res = bits % element_bits;
        if res != 0 {
            blocks + 1
        } else {
            blocks
        }
    };
    IntVector::block_with_fill(element_bits, n_blocks, T::create_zero())
}
