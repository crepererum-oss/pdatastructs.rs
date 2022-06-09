//! A collection of data structures that are based probability theory and therefore only provide
//! the correct answer if a certain probability. In exchange they have a better runtime and memory
//! complexity compared to traditional data structures.

#![deny(const_err)]
#![deny(anonymous_parameters)]
#![deny(bare_trait_objects)]
#![deny(dead_code)]
#![deny(illegal_floating_point_literal_pattern)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(non_camel_case_types)]
#![deny(non_snake_case)]
#![deny(non_upper_case_globals)]
#![deny(unknown_lints)]
#![deny(unreachable_code)]
#![deny(unreachable_patterns)]
#![deny(unreachable_pub)]
#![deny(unsafe_code)]
#![deny(unused_extern_crates)]

#[cfg(feature = "num-traits")]
pub use num_traits;

#[cfg(feature = "rand")]
pub use rand;

#[cfg(feature = "num-traits")]
pub mod countminsketch;

pub mod filters;

pub mod hash_utils;

#[cfg(feature = "succinct")]
mod helpers;

#[cfg(feature = "bytecount")]
pub mod hyperloglog;

#[cfg(feature = "bytecount")]
mod hyperloglog_data;

#[cfg(feature = "rand")]
pub mod reservoirsampling;

#[cfg(feature = "rand")]
pub mod tdigest;

pub mod topk;

#[cfg(test)]
mod test_util;
