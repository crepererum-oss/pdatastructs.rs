//! CountMinSketch implementation.
use std::collections::hash_map::DefaultHasher;
use std::f64;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash};
use std::marker::PhantomData;

use num_traits::{CheckedAdd, One, Unsigned, Zero};

use crate::hash_utils::HashIterBuilder;

/// A CountMinSketch is a data structure to estimate the frequency of elements in a data stream.
///
/// The type parameter `C` sets the type of the counter in the internal table and can be used to
/// reduce memory consumption when low counts are expected (e.g. use `u16` instead of `u64` if you
/// only expected up to `2^16 - 1 = 65535` measurements for each element).
///
/// # Examples
/// ```
/// use pdatastructs::countminsketch::CountMinSketch;
///
/// // set up filter
/// let epsilon = 0.1;  // error threshold
/// let delta = 0.2;  // epsilon is hit in (1 - 0.2) * 100% = 80%
/// let mut cms = CountMinSketch::<&str, u32>::with_point_query_properties(epsilon, delta);
///
/// // add some data
/// cms.add(&"my super long string");
/// cms.add(&"my super long string");
///
/// // later
/// assert_eq!(cms.query_point(&"my super long string"), 2);
/// assert_eq!(cms.query_point(&"another super long string"), 0);
/// ```
///
/// # Applications
/// - when a lot of data is streamed in an approximative counter is good enough, e.g. for access
///   counts in network filters
/// - as a part of the TopK data structure
///
/// # How It Works
///
/// ## Setup
/// The filter is made up of a 2-dimensional table of `w` columns and `d` rows, with evey cell
/// containing a counter initially set to 0.
///
/// ```text
/// w = 2
/// d = 4
///
/// +---+---+
/// | 0 | 0 |
/// +---+---+
/// | 0 | 0 |
/// +---+---+
/// | 0 | 0 |
/// +---+---+
/// | 0 | 0 |
/// +---+---+
/// ```
///
/// ## Insertion
/// When adding an element, for every row `i in [0, d)` a hash function `h_i(x)` is calculated that
/// provides a column `x in [0, w)` each. The counters in the selected `w` cells (one in each row,
/// but not necessarily one in each column) are increased by 1.
///
/// ```text
/// w = 2
/// d = 4
///
/// +---+---+       +---+---+
/// |[0]| 0 |       | 1 | 0 |
/// +---+---+       +---+---+
/// | 0 |[0]|       | 0 | 1 |
/// +---+---+  ==>  +---+---+
/// | 0 |[0]|       | 0 | 1 |
/// +---+---+       +---+---+
/// |[0]| 0 |       | 1 | 0 |
/// +---+---+       +---+---+
/// ```
///
/// ## Query
/// When querying an element, cells are selected by the same scheme and the minimum of the counters
/// is returned as a guess.
///
/// ```text
/// w = 2
/// d = 4
///
/// +---+---+
/// |[7]| 0 |
/// +---+---+
/// | 3 |[4]|
/// +---+---+
/// | 2 |[5]|
/// +---+---+
/// |[6]| 1 |
/// +---+---+
///
/// min(7, 4, 5, 6) = 4
/// ```
///
/// # See Also
/// - `std::collections::HashMap`: exact count but needs to store all elements
/// - `pdatastructs::bloomfilter::BloomFilter`: when you only want to measure if an element was
///   observed or not and the actual count does not matter
///
/// # References
/// - ["Count-Min Sketch", Graham Cormode, 2009](http://dimacs.rutgers.edu/~graham/pubs/papers/cmencyc.pdf)
/// - [Wikipedia: Count–min sketch](https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch)
#[derive(Clone)]
pub struct CountMinSketch<T, C = usize, B = BuildHasherDefault<DefaultHasher>>
where
    T: Hash,
    C: CheckedAdd + Clone + One + Ord + Unsigned + Zero,
    B: BuildHasher + Clone + Eq,
{
    table: Vec<C>,
    w: usize,
    d: usize,
    builder: HashIterBuilder<B>,
    phantom: PhantomData<T>,
}

impl<T, C> CountMinSketch<T, C>
where
    T: Hash,
    C: CheckedAdd + Clone + One + Ord + Unsigned + Zero,
{
    /// Create new CountMinSketch based on table size.
    ///
    /// - `w` sets the number of columns
    /// - `d` sets the number of rows
    pub fn with_params(w: usize, d: usize) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_params_and_hasher(w, d, bh)
    }

    /// Create new CountMinSketch with the following properties for a point query:
    ///
    /// - `a` is the real count of observed objects
    /// - `a'` is the guessed count of observed objects
    /// - `N` is the total count in the internal table
    /// - `a <= a'` always holds
    /// - `a' <= a + epsilon * N` holds with `p > 1 - delta`
    ///
    /// The following conditions must hold:
    ///
    /// - `epsilon > 0`
    /// - `delta > 0` and `delta < 1`
    ///
    /// Panics when the input conditions do not hold.
    pub fn with_point_query_properties(epsilon: f64, delta: f64) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_point_query_properties_and_hasher(epsilon, delta, bh)
    }
}

impl<T, C, B> CountMinSketch<T, C, B>
where
    T: Hash,
    C: CheckedAdd + Clone + One + Ord + Unsigned + Zero,
    B: BuildHasher + Clone + Eq,
{
    /// Same as `with_params` but with a specific `BuildHasher`.
    pub fn with_params_and_hasher(w: usize, d: usize, buildhasher: B) -> Self {
        let table = vec![C::zero(); w.checked_mul(d).unwrap()];
        Self {
            table,
            w,
            d,
            builder: HashIterBuilder::new(w, d, buildhasher),
            phantom: PhantomData,
        }
    }

    /// Same as `with_params_and_hasher` but with a specific `BuildHasher`.
    pub fn with_point_query_properties_and_hasher(
        epsilon: f64,
        delta: f64,
        buildhasher: B,
    ) -> Self {
        assert!(epsilon > 0., "epsilon must be greater than 0");
        assert!(
            (delta > 0.) & (delta < 1.),
            "delta ({}) must be greater than 0 and smaller than 1",
            delta
        );

        let w = (f64::consts::E / epsilon).ceil() as usize;
        let d = (1. / delta).ln().ceil() as usize;
        CountMinSketch::with_params_and_hasher(w, d, buildhasher)
    }

    /// Get number of columns of internal counter table.
    pub fn w(&self) -> usize {
        self.w
    }

    /// Get number of rows of internal counter table.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Get `BuildHasher`
    pub fn buildhasher(&self) -> &B {
        self.builder.buildhasher()
    }

    /// Check whether the CountMinSketch is empty (i.e. no elements seen yet).
    pub fn is_empty(&self) -> bool {
        self.table.iter().all(|x| x.is_zero())
    }

    /// Add one to the counter of the given element.
    ///
    /// Returns guessed count after addition.
    pub fn add(&mut self, obj: &T) -> C {
        self.add_n(&obj, &C::one())
    }

    /// Add `n` to the counter of the given element.
    pub fn add_n(&mut self, obj: &T, n: &C) -> C {
        let mut result = C::zero();
        for (i, pos) in self.builder.iter_for(obj).enumerate() {
            let x = i * self.w + pos;
            let current = self.table[x].clone();
            result = if i == 0 {
                current.clone()
            } else {
                result.min(current.clone())
            };
            self.table[x] = current.checked_add(n).unwrap();
        }
        result.checked_add(n).unwrap()
    }

    /// Runs a point query, i.e. a query for the count of a single object.
    ///
    /// Returns guessed count after addition.
    pub fn query_point(&self, obj: &T) -> C {
        self.builder
            .iter_for(obj)
            .enumerate()
            .map(|(i, pos)| i * self.w + pos)
            .map(|x| self.table[x].clone())
            .min()
            .unwrap()
    }

    /// Merge self with another CountMinSketch.
    ///
    /// After this operation `self` will be in the same state as when it would have seen all
    /// elements from `self` and `other`.
    ///
    /// Panics when `d`, `w` or `buildhasher` from `self` and `other` differ.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(
            self.d, other.d,
            "number of rows (d) must be equal (left={}, right={})",
            self.d, other.d
        );
        assert_eq!(
            self.w, other.w,
            "number of columns (w) must be equal (left={}, right={})",
            self.w, other.w
        );
        assert!(
            self.buildhasher() == other.buildhasher(),
            "buildhasher must be equal"
        );

        self.table = self
            .table
            .iter()
            .zip(other.table.iter())
            .map(|x| x.0.checked_add(x.1).unwrap())
            .collect();
    }

    /// Clear internal counters to a fresh state (i.e. no objects seen).
    pub fn clear(&mut self) {
        self.table = vec![C::zero(); self.w.checked_mul(self.d).unwrap()];
    }
}

impl<T> fmt::Debug for CountMinSketch<T>
where
    T: Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CountMinSketch {{ w: {}, d: {} }}", self.w, self.d)
    }
}

impl<T> Extend<T> for CountMinSketch<T>
where
    T: Hash,
{
    fn extend<S: IntoIterator<Item = T>>(&mut self, iter: S) {
        for elem in iter {
            self.add(&elem);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CountMinSketch;
    use crate::hash_utils::BuildHasherSeeded;

    #[test]
    fn getter() {
        let cms = CountMinSketch::<u64>::with_params(10, 20);
        assert_eq!(cms.w(), 10);
        assert_eq!(cms.d(), 20);
        cms.buildhasher();
    }

    #[test]
    fn properties() {
        let cms = CountMinSketch::<u64>::with_point_query_properties(0.01, 0.1);
        assert_eq!(cms.w(), 272);
        assert_eq!(cms.d(), 3);
    }

    #[test]
    #[should_panic(expected = "epsilon must be greater than 0")]
    fn properties_panics_epsilon0() {
        CountMinSketch::<u64>::with_point_query_properties(0., 0.1);
    }

    #[test]
    #[should_panic(expected = "delta (0) must be greater than 0 and smaller than 1")]
    fn properties_panics_delta0() {
        CountMinSketch::<u64>::with_point_query_properties(0.01, 0.);
    }

    #[test]
    #[should_panic(expected = "delta (1) must be greater than 0 and smaller than 1")]
    fn properties_panics_delta1() {
        CountMinSketch::<u64>::with_point_query_properties(0.01, 1.);
    }

    #[test]
    fn empty() {
        let cms = CountMinSketch::<u64>::with_params(10, 10);
        assert_eq!(cms.query_point(&1u64), 0);
        assert!(cms.is_empty());
    }

    #[test]
    fn add_1() {
        let mut cms = CountMinSketch::<u64>::with_params(10, 10);

        assert_eq!(cms.add(&1), 1);
        assert_eq!(cms.query_point(&1), 1);
        assert_eq!(cms.query_point(&2), 0);
    }

    #[test]
    fn add_2() {
        let mut cms = CountMinSketch::<u64>::with_params(10, 10);

        assert_eq!(cms.add(&1), 1);
        assert_eq!(cms.add(&1), 2);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 0);
    }

    #[test]
    fn add_2_1a() {
        let mut cms = CountMinSketch::<u64>::with_params(10, 10);

        assert_eq!(cms.add(&1), 1);
        assert_eq!(cms.add(&2), 1);
        assert_eq!(cms.add(&1), 2);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 1);
        assert_eq!(cms.query_point(&3), 0);
    }

    #[test]
    fn add_2_1b() {
        let mut cms = CountMinSketch::<u64>::with_params(10, 10);

        assert_eq!(cms.add_n(&1, &2), 2);
        assert_eq!(cms.add(&2), 1);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 1);
        assert_eq!(cms.query_point(&3), 0);
    }

    #[test]
    fn merge() {
        let mut cms1 = CountMinSketch::<u64>::with_params(10, 10);
        let mut cms2 = CountMinSketch::<u64>::with_params(10, 10);

        cms1.add_n(&1, &1);
        cms1.add_n(&2, &2);
        assert_eq!(cms1.query_point(&1), 1);
        assert_eq!(cms1.query_point(&2), 2);
        assert_eq!(cms1.query_point(&3), 0);
        assert_eq!(cms1.query_point(&4), 0);

        cms2.add_n(&2, &20);
        cms2.add_n(&3, &30);
        assert_eq!(cms2.query_point(&1), 0);
        assert_eq!(cms2.query_point(&2), 20);
        assert_eq!(cms2.query_point(&3), 30);
        assert_eq!(cms2.query_point(&4), 0);

        cms1.merge(&cms2);
        assert_eq!(cms1.query_point(&1), 1);
        assert_eq!(cms1.query_point(&2), 22);
        assert_eq!(cms1.query_point(&3), 30);
        assert_eq!(cms1.query_point(&4), 0);
    }

    #[test]
    #[should_panic(expected = "number of columns (w) must be equal (left=10, right=20)")]
    fn merge_panics_w() {
        let mut cms1 = CountMinSketch::<u64>::with_params(10, 10);
        let cms2 = CountMinSketch::<u64>::with_params(20, 10);
        cms1.merge(&cms2);
    }

    #[test]
    #[should_panic(expected = "number of rows (d) must be equal (left=10, right=20)")]
    fn merge_panics_d() {
        let mut cms1 = CountMinSketch::<u64>::with_params(10, 10);
        let cms2 = CountMinSketch::<u64>::with_params(10, 20);
        cms1.merge(&cms2);
    }

    #[test]
    #[should_panic(expected = "buildhasher must be equal")]
    fn merge_panics_buildhasher() {
        let mut cms1 = CountMinSketch::<u64, usize, BuildHasherSeeded>::with_params_and_hasher(
            10,
            10,
            BuildHasherSeeded::new(0),
        );
        let cms2 = CountMinSketch::<u64, usize, BuildHasherSeeded>::with_params_and_hasher(
            10,
            10,
            BuildHasherSeeded::new(1),
        );
        cms1.merge(&cms2);
    }

    #[test]
    fn clear() {
        let mut cms = CountMinSketch::<u64>::with_params(10, 10);

        cms.add(&1);
        assert_eq!(cms.query_point(&1), 1);
        assert!(!cms.is_empty());

        cms.clear();
        assert_eq!(cms.query_point(&1), 0);
        assert!(cms.is_empty());
    }

    #[test]
    fn clone() {
        let mut cms1 = CountMinSketch::<u64>::with_params(10, 10);

        cms1.add(&1);
        assert_eq!(cms1.query_point(&1), 1);
        assert_eq!(cms1.query_point(&2), 0);

        let cms2 = cms1.clone();
        assert_eq!(cms2.query_point(&1), 1);
        assert_eq!(cms2.query_point(&2), 0);

        cms1.add(&1);
        assert_eq!(cms1.query_point(&1), 2);
        assert_eq!(cms1.query_point(&2), 0);
        assert_eq!(cms2.query_point(&1), 1);
        assert_eq!(cms2.query_point(&2), 0);
    }

    #[test]
    fn debug() {
        let cms = CountMinSketch::<u64>::with_params(10, 20);
        assert_eq!(format!("{:?}", cms), "CountMinSketch { w: 10, d: 20 }");
    }

    #[test]
    fn extend() {
        let mut cms = CountMinSketch::<u64>::with_params(10, 10);

        cms.extend(vec![1, 1]);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 0);
    }
}
