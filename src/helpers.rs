#![allow(dead_code)]

use succinct::IntVector;
use succinct::storage::BlockType;

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
        let total_bits = len.checked_mul(element_bits).expect("Table size too large");
        let single_block_bits = size_of::<T>().checked_mul(8).expect("Block size too large");
        let blocks = total_bits / single_block_bits;
        let res = total_bits % single_block_bits;
        if res != 0 { blocks + 1 } else { blocks }
    };
    IntVector::block_with_fill(element_bits, n_blocks, T::create_zero())
}

#[cfg(test)]
mod tests {
    use succinct::{BitVec, IntVector};

    use super::all_zero_intvector;

    #[test]
    fn test_all_zero_intvector_size() {
        let v: IntVector<u64> = all_zero_intvector(10, 5);
        // block_len is the length of the actual Vec<u64>
        assert_eq!(v.block_len(), 1);
    }
}
