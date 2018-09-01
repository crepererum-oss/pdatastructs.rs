//! BloomFilter implementation.
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{BuildHasher, Hash};

use fixedbitset::FixedBitSet;

use hash_utils::{HashIterBuilder, MyBuildHasherDefault};

/// A BloomFilter is a set-like data structure, that keeps track of elements it has seen without
/// the need to store them. Looking up values has a certain false positive rate, but a false
/// negative rate of 0%.
///
/// # Examples
/// ```
/// use pdatastructs::filters::bloomfilter::BloomFilter;
///
/// // set up filter
/// let false_positive_rate = 0.02;  // = 2%
/// let expected_elements = 1000;
/// let mut filter = BloomFilter::with_properties(expected_elements, false_positive_rate);
///
/// // add some data
/// filter.add(&"my super long string");
///
/// // later
/// assert!(filter.query(&"my super long string"));
/// assert!(!filter.query(&"another super long string"));
/// ```
///
/// # Applications
/// - when a lot of data should be added to the set and a moderate false positive rate is
///   acceptable, was used for spell checking
/// - as a pre-filter for more expensive lookups, e.g. in combination with a real set, map or
///   database, so the final false positive rate is 0%
///
/// # How It Works
/// The filter is represented by a bit vector of size `m`. Also given are `k` hash functions
/// `h_i(x), for i in 0..k`, every one mapping an input value `x` to an integer `>= 0` and `< m`.
/// Initially, all bits are set to `False`.
///
/// During insertion of value `x`, the `k` bits addressed by `h_i(x), for i in 0..k` are set to
/// `True`.
///
/// During lookup, it is checked if all these bits are set. If so, the value might be in the
/// filter. If only a single bit is not set, it is clear that the value was never added to the
/// filter.
///
/// # See Also
/// - `std::collections::HashSet`: has a false positive rate of 0%, but also needs to store all
///   elements
/// - `pdatastructs::cuckoofilter::CuckooFilter`: better under some circumstances, but more
///   complex data structure
///
/// # References
/// - ["Space/Time Trade-offs in Hash Coding with Allowable Errors", Burton H. Bloom, 1970](http://dmod.eu/deca/ft_gateway.cfm.pdf)
/// - [Wikipedia: Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter)
#[derive(Clone)]
pub struct BloomFilter<B = MyBuildHasherDefault<DefaultHasher>>
where
    B: BuildHasher + Clone + Eq,
{
    bs: FixedBitSet,
    k: usize,
    builder: HashIterBuilder<B>,
}

impl BloomFilter {
    /// Create new, empty BloomFilter with internal parameters.
    ///
    /// - `k` is the number of hash functions
    /// - `m` is the number of bits used to store state
    pub fn with_params(m: usize, k: usize) -> Self {
        let bh = MyBuildHasherDefault::<DefaultHasher>::default();
        Self::with_params_and_hash(m, k, bh)
    }

    /// Create new, empty BloomFilter with given properties.
    ///
    /// - `n` number of unique elements the BloomFilter is expected to hold, must be `> 0`
    /// - `p` false positive rate when querying the BloomFilter after adding `n` unique
    ///   elements, must be `> 0` and `< 1`
    ///
    /// Panics if the parameters are not in range.
    pub fn with_properties(n: usize, p: f64) -> Self {
        let bh = MyBuildHasherDefault::<DefaultHasher>::default();
        Self::with_properties_and_hash(n, p, bh)
    }
}

impl<B> BloomFilter<B>
where
    B: BuildHasher + Clone + Eq,
{
    /// Same as `with_params` but with specific `BuildHasher`.
    pub fn with_params_and_hash(m: usize, k: usize, buildhasher: B) -> Self {
        Self {
            bs: FixedBitSet::with_capacity(m),
            k,
            builder: HashIterBuilder::new(m, k, buildhasher),
        }
    }

    /// Same as `with_properties` but with specific `BuildHasher`.
    pub fn with_properties_and_hash(n: usize, p: f64, buildhasher: B) -> Self {
        assert!(n > 0, "n must be greater than 0");
        assert!(
            (p > 0.) & (p < 1.),
            "p ({}) must be greater than 0 and smaller than 1",
            p
        );

        let k = (-p.log2()) as usize;
        let ln2 = (2f64).ln();
        let m = (-((n as f64) * p.ln()) / (ln2 * ln2)) as usize;

        BloomFilter::with_params_and_hash(m, k, buildhasher)
    }

    /// Get `k` (number of hash functions).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get `m` (number of stored bits).
    pub fn m(&self) -> usize {
        self.bs.len()
    }

    /// Get `BuildHasher`.
    pub fn buildhasher(&self) -> &B {
        self.builder.buildhasher()
    }

    /// Add new element to the BloomFilter.
    ///
    /// If the same element is added multiple times or if an element results in the same hash
    /// signature, this method does not have any effect.
    pub fn add<T>(&mut self, obj: &T)
    where
        T: Hash,
    {
        for pos in self.builder.iter_for(obj) {
            self.bs.set(pos, true);
        }
    }

    /// Guess if the given element was added to the BloomFilter.
    pub fn query<T>(&self, obj: &T) -> bool
    where
        T: Hash,
    {
        for pos in self.builder.iter_for(obj) {
            if !self.bs[pos] {
                return false;
            }
        }
        true
    }

    /// Clear state of the BloomFilter, so that it behaves like a fresh one.
    pub fn clear(&mut self) {
        self.bs.clear()
    }

    /// Check whether the BloomFilter is empty.
    pub fn is_empty(&self) -> bool {
        self.bs.ones().next().is_none()
    }

    /// Add the entire content of another bloomfilter to this BloomFilter.
    ///
    /// The result is the same as adding all elements added to `other` to `self` in the first
    /// place.
    ///
    /// Panics if `k`,`m` or `buildhasher` of the two BloomFilters are not identical.
    pub fn union(&mut self, other: &Self) {
        assert_eq!(
            self.k, other.k,
            "k must be equal (left={}, right={})",
            self.k, other.k
        );
        assert_eq!(
            self.bs.len(),
            other.bs.len(),
            "m must be equal (left={}, right={})",
            self.bs.len(),
            other.bs.len()
        );
        assert!(
            self.buildhasher() == other.buildhasher(),
            "buildhasher must be equal"
        );

        self.bs = &self.bs | &other.bs;
    }

    /// Guess the number of unique elements added to the BloomFilter.
    pub fn guess_n(&self) -> usize {
        let m = self.bs.len() as f64;
        let k = self.k as f64;
        let x = self.bs.ones().count() as f64;

        (-m / k * (1. - x / m).ln()) as usize
    }
}

impl fmt::Debug for BloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BloomFilter {{ m: {}, k: {} }}", self.bs.len(), self.k)
    }
}

impl<T> Extend<T> for BloomFilter
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
    use super::BloomFilter;
    use hash_utils::BuildHasherSeeded;

    #[test]
    fn getter() {
        let bf = BloomFilter::with_params(100, 2);
        assert_eq!(bf.k(), 2);
        assert_eq!(bf.m(), 100);
        bf.buildhasher();
    }

    #[test]
    fn empty() {
        let bf = BloomFilter::with_params(100, 2);
        assert!(!bf.query(&1));
    }

    #[test]
    fn add() {
        let mut bf = BloomFilter::with_params(100, 2);

        bf.add(&1);
        assert!(bf.query(&1));
        assert!(!bf.query(&2));
    }

    #[test]
    fn clear() {
        let mut bf = BloomFilter::with_params(100, 2);

        bf.add(&1);
        bf.clear();
        assert!(!bf.query(&1));
    }

    #[test]
    fn is_empty() {
        let mut bf = BloomFilter::with_params(100, 2);
        assert!(bf.is_empty());

        bf.add(&1);
        assert!(!bf.is_empty());

        bf.clear();
        assert!(bf.is_empty());
    }

    #[test]
    fn clone() {
        let mut bf1 = BloomFilter::with_params(100, 2);
        bf1.add(&1);

        let bf2 = bf1.clone();
        bf1.add(&2);
        assert!(bf2.query(&1));
        assert!(!bf2.query(&2));
    }

    #[test]
    fn union() {
        let mut bf1 = BloomFilter::with_params(100, 2);
        bf1.add(&1);
        assert!(bf1.query(&1));
        assert!(!bf1.query(&2));
        assert!(!bf1.query(&3));

        let mut bf2 = BloomFilter::with_params(100, 2);
        bf2.add(&2);
        assert!(!bf2.query(&1));
        assert!(bf2.query(&2));
        assert!(!bf2.query(&3));

        bf1.union(&bf2);
        assert!(bf1.query(&1));
        assert!(bf1.query(&2));
        assert!(!bf1.query(&3));
    }

    #[test]
    #[should_panic(expected = "k must be equal (left=2, right=3)")]
    fn union_panics_k() {
        let mut bf1 = BloomFilter::with_params(100, 2);
        let bf2 = BloomFilter::with_params(100, 3);
        bf1.union(&bf2);
    }

    #[test]
    #[should_panic(expected = "m must be equal (left=100, right=200)")]
    fn union_panics_m() {
        let mut bf1 = BloomFilter::with_params(100, 2);
        let bf2 = BloomFilter::with_params(200, 2);
        bf1.union(&bf2);
    }

    #[test]
    #[should_panic(expected = "buildhasher must be equal")]
    fn union_panics_buildhasher() {
        let mut bf1 = BloomFilter::with_params_and_hash(100, 2, BuildHasherSeeded::new(0));
        let bf2 = BloomFilter::with_params_and_hash(100, 2, BuildHasherSeeded::new(1));
        bf1.union(&bf2);
    }

    #[test]
    fn with_properties() {
        let bf = BloomFilter::with_properties(1000, 0.1);
        assert_eq!(bf.k(), 3);
        assert_eq!(bf.m(), 4792);
    }

    #[test]
    #[should_panic(expected = "n must be greater than 0")]
    fn with_properties_panics_n0() {
        BloomFilter::with_properties(0, 0.1);
    }

    #[test]
    #[should_panic(expected = "p (0) must be greater than 0 and smaller than 1")]
    fn with_properties_panics_p0() {
        BloomFilter::with_properties(1000, 0.);
    }

    #[test]
    #[should_panic(expected = "p (1) must be greater than 0 and smaller than 1")]
    fn with_properties_panics_p1() {
        BloomFilter::with_properties(1000, 1.);
    }

    #[test]
    fn guess_n() {
        let mut bf = BloomFilter::with_params(100, 2);
        assert_eq!(bf.guess_n(), 0);

        bf.add(&1);
        assert_eq!(bf.guess_n(), 1);

        bf.add(&1);
        assert_eq!(bf.guess_n(), 1);

        bf.add(&2);
        assert_eq!(bf.guess_n(), 2);
    }

    #[test]
    fn debug() {
        let bf = BloomFilter::with_params(100, 2);
        assert_eq!(format!("{:?}", bf), "BloomFilter { m: 100, k: 2 }");
    }

    #[test]
    fn extend() {
        let mut bf = BloomFilter::with_params(100, 2);

        bf.extend(vec![1, 2]);
        assert!(bf.query(&1));
        assert!(bf.query(&2));
        assert!(!bf.query(&3));
    }
}
