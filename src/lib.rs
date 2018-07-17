//! A collection of data structures that are based probability theory and therefore only provide
//! the correct answer if a certain probability. In exchange they have a better runtime and memory
//! complexity compared to traditional data structures.

#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unused_extern_crates)]

extern crate bytecount;
extern crate fixedbitset;
pub extern crate num_traits;
pub extern crate rand;

pub mod bloomfilter;
pub mod countminsketch;
pub mod cuckoofilter;
pub mod hash_utils;
pub mod hyperloglog;
pub mod reservoirsampling;
pub mod topk;
