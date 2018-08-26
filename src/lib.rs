//! A collection of data structures that are based probability theory and therefore only provide
//! the correct answer if a certain probability. In exchange they have a better runtime and memory
//! complexity compared to traditional data structures.

#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![deny(unused_extern_crates)]

extern crate bytecount;
extern crate fixedbitset;
pub extern crate num_traits;
pub extern crate rand;
extern crate succinct;

pub mod bloomfilter;
pub mod countminsketch;
pub mod cuckoofilter;
pub mod hash_utils;
pub mod hyperloglog;
mod hyperloglog_data;
pub mod reservoirsampling;
pub mod topk;
