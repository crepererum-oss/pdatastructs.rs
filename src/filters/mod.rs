//! Filters, Approximate Membership Queries (AMQs).

#[cfg(feature = "fixedbitset")]
pub mod bloomfilter;

pub mod compat;

#[cfg(all(feature = "rand", feature = "succinct"))]
pub mod cuckoofilter;

#[cfg(all(feature = "fixedbitset", feature = "succinct"))]
pub mod quotientfilter;

use std::fmt::Debug;
use std::hash::Hash;

/// A filter is a set-like data structure, that keeps track of elements it has seen without
/// the need to store them. Looking up values has a certain false positive rate, but a false
/// negative rate of 0%.
///
/// This kind of lookup is also referred to as Approximate Membership Queries (AMQs).
pub trait Filter<T>
where
    T: Hash + ?Sized,
{
    /// Error type that may occur during insertion.
    type InsertErr: Debug;

    /// Clear state of the filter, so that it behaves like a fresh one.
    fn clear(&mut self);

    /// Insert new element into the filter.
    ///
    /// In success-case, it will be reported if the element was likely already part of the filter.
    /// If the element was already known, `false` is returned, otherwise `true`. You may get the
    /// same result by calling `query`, but calling insert is more efficient then calling `query`
    /// first and then using `insert` on demand.
    ///
    /// The method may return an error under certain conditions. When this happens, the
    /// user-visible state is not altered, i.e. the element was not added to the filter. The
    /// internal state may have changed though.
    fn insert(&mut self, obj: &T) -> Result<bool, Self::InsertErr>;

    /// Add all elements from `other` into `self`.
    ///
    /// The result is the same as adding all elements added to `other` to `self` in the first
    /// place.
    fn union(&mut self, other: &Self) -> Result<(), Self::InsertErr>
    where
        Self: Sized;

    /// Check if filters is empty, i.e. contains no elements.
    fn is_empty(&self) -> bool;

    /// Return guessed number of elements in the filter.
    fn len(&self) -> usize;

    /// Guess if the given element was added to the filter.
    fn query(&self, obj: &T) -> bool;
}
