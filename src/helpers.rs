use std::mem;

use num_traits::Zero;
use succinct::storage::BlockType;
use succinct::IntVector;

/// Quicker all-zero initialization of an IntVector.
pub(crate) fn all_zero_intvector<T>(element_bits: usize, len: usize) -> IntVector<T>
where
    T: BlockType + Zero,
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
    IntVector::block_with_fill(element_bits, n_blocks, T::zero())
}
