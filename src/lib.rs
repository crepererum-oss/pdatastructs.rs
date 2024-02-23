#![doc = concat!(include_str!("../README.md"), "\n\n", include_str!("../CHANGELOG.md"))]
#![deny(
    anonymous_parameters,
    bare_trait_objects,
    clippy::clone_on_ref_ptr,
    clippy::explicit_iter_loop,
    clippy::future_not_send,
    clippy::use_self,
    dead_code,
    illegal_floating_point_literal_pattern,
    missing_copy_implementations,
    missing_debug_implementations,
    missing_docs,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    rust_2018_idioms,
    rustdoc::bare_urls,
    rustdoc::broken_intra_doc_links,
    unknown_lints,
    unreachable_code,
    unreachable_patterns,
    unreachable_pub,
    unsafe_code,
    unused_extern_crates
)]

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

#[cfg(feature = "rand")]
pub mod reservoirsampling;

#[cfg(feature = "rand")]
pub mod tdigest;

pub mod topk;

#[cfg(test)]
mod test_util;
