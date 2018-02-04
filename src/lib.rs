//! A collection of data structures that are based probability theory and therefore only provide
//! the correct answer if a certain probability. In exchange they have a better runtime and memory
//! complexity compared to traditional data structures.

#![deny(missing_docs)]

extern crate bit_vec;
extern crate rand;

mod bloomfilter;
mod countminsketch;
mod hyperloglog;
mod reservoirsampling;
mod topk;
mod utils;

pub use bloomfilter::BloomFilter;
pub use countminsketch::{CountMinSketch, Counter};
pub use hyperloglog::HyperLogLog;
pub use reservoirsampling::ReservoirSampling;
pub use topk::TopK;
pub use utils::MyBuildHasherDefault;
