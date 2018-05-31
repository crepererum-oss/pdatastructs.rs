//! A collection of data structures that are based probability theory and therefore only provide
//! the correct answer if a certain probability. In exchange they have a better runtime and memory
//! complexity compared to traditional data structures.

#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unused_extern_crates)]

extern crate fixedbitset;
pub extern crate num_traits;
pub extern crate rand;

mod bloomfilter;
mod countminsketch;
mod hyperloglog;
mod reservoirsampling;
mod topk;
mod utils;

pub use bloomfilter::BloomFilter;
pub use countminsketch::CountMinSketch;
pub use hyperloglog::HyperLogLog;
pub use reservoirsampling::ReservoirSampling;
pub use topk::TopK;
pub use utils::{BuildHasherSeeded, MyBuildHasherDefault};
