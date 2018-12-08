//! A collection of data structures that are based probability theory and therefore only provide
//! the correct answer if a certain probability. In exchange they have a better runtime and memory
//! complexity compared to traditional data structures.

#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![deny(unused_extern_crates)]

pub use num_traits;
pub use rand;

pub mod countminsketch;
pub mod filters;
pub mod hash_utils;
mod helpers;
pub mod hyperloglog;
mod hyperloglog_data;
pub mod reservoirsampling;
pub mod topk;
