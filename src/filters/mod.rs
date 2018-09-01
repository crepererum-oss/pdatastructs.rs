//! Filters, Approximate Membership Queries (AMQs).

pub mod bloomfilter;
pub mod cuckoofilter;

use std::hash::Hash;

/// A filter is a set-like data structure, that keeps track of elements it has seen without
/// the need to store them. Looking up values has a certain false positive rate, but a false
/// negative rate of 0%.
///
/// This kind of lookup is also referred to as Approximate Membership Queries (AMQs).
pub trait Filter {
    /// Check if filters is empty, i.e. contains no elements.
    fn is_empty(&self) -> bool;

    /// Guess if the given element was added to the filter.
    fn query<T>(&self, obj: &T) -> bool
    where
        T: Hash;
}
