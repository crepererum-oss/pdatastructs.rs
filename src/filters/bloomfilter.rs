//! BloomFilter implementation.
use std::collections::hash_map::DefaultHasher;
use std::convert::Infallible;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash};
use std::marker::PhantomData;

use fixedbitset::FixedBitSet;

use crate::filters::Filter;
use crate::hash_utils::HashIterBuilder;

/// A BloomFilter is a set-like data structure, that keeps track of elements it has seen without
/// the need to store them. Looking up values has a certain false positive rate, but a false
/// negative rate of 0%.
///
/// # Examples
/// ```
/// use pdatastructs::filters::Filter;
/// use pdatastructs::filters::bloomfilter::BloomFilter;
///
/// // set up filter
/// let false_positive_rate = 0.02;  // = 2%
/// let expected_elements = 1000;
/// let mut filter = BloomFilter::with_properties(expected_elements, false_positive_rate);
///
/// // add some data
/// filter.insert(&"my super long string").unwrap();
///
/// // later
/// assert!(filter.query(&"my super long string"));
/// assert!(!filter.query(&"another super long string"));
/// ```
///
/// Note that the filter is specific to `T`, so the following will not compile:
///
/// ```compile_fail
/// use pdatastructs::filters::Filter;
/// use pdatastructs::filters::bloomfilter::BloomFilter;
///
/// let false_positive_rate = 0.02;  // = 2%
/// let expected_elements = 1000;
/// let mut filter1 = BloomFilter::<u8>::with_properties(expected_elements, false_positive_rate);
/// let filter2 = BloomFilter::<i8>::with_properties(expected_elements, false_positive_rate);
///
/// filter1.union(&filter2);
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
/// ```text
/// m = 8
/// k = 2
///
/// +---------++---+---+---+---+---+---+---+---+
/// | address || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
/// |    data ||   |   |   |   |   |   |   |   |
/// +---------++---+---+---+---+---+---+---+---+
/// ```
///
/// ## Insertion
/// During insertion of value `x`, the `k` bits addressed by `h_i(x), for i in 0..k` are set to
/// `True`.
///
/// ```text
/// insert(a):
///
///   h0(a) = 1
///   h1(a) = 6
///
/// +---------++---+---+---+---+---+---+---+---+
/// | address || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
/// |    data ||   | X |   |   |   |   | X |   |
/// +---------++---+---+---+---+---+---+---+---+
///
///
/// insert(b):
///
///   h0(b) = 6
///   h1(b) = 4
///
/// +---------++---+---+---+---+---+---+---+---+
/// | address || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
/// |    data ||   | X |   |   | X |   | X |   |
/// +---------++---+---+---+---+---+---+---+---+
/// ```
///
/// ## Query
/// During lookup, it is checked if all these bits are set. If so, the value might be in the
/// filter.
///
/// ```text
/// query(a):
///
///   h0(a) = 1
///   h1(a) = 6
///
/// +---------++---+---+---+---+---+---+---+---+
/// | address || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
/// |    data ||   |(X)|   |   | X |   |(X)|   |
/// +---------++---+---+---+---+---+---+---+---+
///
/// => present
/// ```
///
/// If only a single bit is not set, it is clear that the value was never added to the
/// filter.
///
/// ```text
/// query(c):
///
///   h0(a) = 2
///   h1(a) = 6
///
/// +---------++---+---+---+---+---+---+---+---+
/// | address || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
/// |    data ||   | X |( )|   | X |   |(X)|   |
/// +---------++---+---+---+---+---+---+---+---+
///
/// => absent
/// ```
///
/// What could happen of course is that the filter results in false positives.
///
/// ```text
/// query(d):
///
///   h0(a) = 4
///   h1(a) = 1
///
/// +---------++---+---+---+---+---+---+---+---+
/// | address || 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
/// |    data ||   |(X)|   |   |(X)|   | X |   |
/// +---------++---+---+---+---+---+---+---+---+
///
/// => present
/// ```
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
pub struct BloomFilter<T: ?Sized, B = BuildHasherDefault<DefaultHasher>>
where
    T: Hash,
    B: BuildHasher + Clone + Eq,
{
    bs: FixedBitSet,
    k: usize,
    builder: HashIterBuilder<B>,
    phantom: PhantomData<fn() -> T>,
}

impl<T> BloomFilter<T>
where
    T: Hash + ?Sized,
{
    /// Create new, empty BloomFilter with internal parameters.
    ///
    /// - `k` is the number of hash functions
    /// - `m` is the number of bits used to store state
    pub fn with_params(m: usize, k: usize) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_params_and_hash(m, k, bh)
    }

    /// Create BloomFilter with internal parameters and existing bitmap.
    ///
    /// - `k` is the number of hash functions
    /// - `m` is the number of bits used to store state
    /// - `bitmap` is the bitmap from an existing bloom filter
    pub fn with_existing_filter<I: IntoIterator<Item = u32>>(
        m: usize,
        k: usize,
        bitmap: I,
    ) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_existing_filter_and_hash(m, k, bitmap, bh)
    }

    /// Create new, empty BloomFilter with given properties.
    ///
    /// - `n` number of unique elements the BloomFilter is expected to hold, must be `> 0`
    /// - `p` false positive rate when querying the BloomFilter after adding `n` unique
    ///   elements, must be `> 0` and `< 1`
    ///
    /// Panics if the parameters are not in range.
    pub fn with_properties(n: usize, p: f64) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_properties_and_hash(n, p, bh)
    }
}

impl<T, B> BloomFilter<T, B>
where
    T: Hash + ?Sized,
    B: BuildHasher + Clone + Eq,
{
    /// Same as `with_params` but with specific `BuildHasher`.
    pub fn with_params_and_hash(m: usize, k: usize, buildhasher: B) -> Self {
        Self {
            bs: FixedBitSet::with_capacity(m),
            k,
            builder: HashIterBuilder::new(m, k, buildhasher),
            phantom: PhantomData,
        }
    }

    /// Same as `with_existing_filter` but with specific `BuildHasher`.
    pub fn with_existing_filter_and_hash<I: IntoIterator<Item = u32>>(
        m: usize,
        k: usize,
        bitmap: I,
        buildhasher: B,
    ) -> Self {
        Self {
            bs: FixedBitSet::with_capacity_and_blocks(m, bitmap),
            k,
            builder: HashIterBuilder::new(m, k, buildhasher),
            phantom: PhantomData,
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

        Self::with_params_and_hash(m, k, buildhasher)
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

    /// Get bitmap data.
    pub fn bitmap(&self) -> &[u32] {
        self.bs.as_slice()
    }
}

impl<T, B> Filter<T> for BloomFilter<T, B>
where
    T: Hash + ?Sized,
    B: BuildHasher + Clone + Eq,
{
    type InsertErr = Infallible;

    fn clear(&mut self) {
        self.bs.clear()
    }

    /// Add new element to the BloomFilter.
    ///
    /// If the same element is added multiple times or if an element results in the same hash
    /// signature, this method does not have any effect.
    fn insert(&mut self, obj: &T) -> Result<bool, Self::InsertErr> {
        let mut was_present = true;
        for pos in self.builder.iter_for(obj) {
            was_present &= self.bs.put(pos);
        }

        Ok(!was_present)
    }

    /// Add the entire content of another bloomfilter to this BloomFilter.
    ///
    /// The result is the same as adding all elements added to `other` to `self` in the first
    /// place.
    ///
    /// Panics if `k`,`m` or `buildhasher` of the two BloomFilters are not identical.
    fn union(&mut self, other: &Self) -> Result<(), Self::InsertErr> {
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

        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.bs.ones().next().is_none()
    }

    fn len(&self) -> usize {
        let m = self.bs.len() as f64;
        let k = self.k as f64;
        let x = self.bs.ones().count() as f64;

        (-m / k * (1. - x / m).ln()) as usize
    }

    fn query(&self, obj: &T) -> bool {
        for pos in self.builder.iter_for(obj) {
            if !self.bs[pos] {
                return false;
            }
        }
        true
    }
}

impl<T> fmt::Debug for BloomFilter<T>
where
    T: Hash + ?Sized,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter {{ m: {}, k: {} }}", self.bs.len(), self.k)
    }
}

impl<T> Extend<T> for BloomFilter<T>
where
    T: Hash,
{
    fn extend<S: IntoIterator<Item = T>>(&mut self, iter: S) {
        for elem in iter {
            self.insert(&elem).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BloomFilter;

    use crate::filters::Filter;
    use crate::hash_utils::BuildHasherSeeded;
    use crate::test_util::{assert_send, NotSend};

    #[test]
    fn getter() {
        let bf = BloomFilter::<u64>::with_params(100, 2);
        assert_eq!(bf.k(), 2);
        assert_eq!(bf.m(), 100);
        bf.buildhasher();
    }

    #[test]
    fn empty() {
        let bf = BloomFilter::<u64>::with_params(100, 2);
        assert!(!bf.query(&1));
    }

    #[test]
    fn insert() {
        let mut bf = BloomFilter::with_params(100, 2);

        assert!(bf.insert(&1).unwrap());
        assert!(bf.query(&1));
        assert!(!bf.query(&2));
    }

    #[test]
    fn double_insert() {
        let mut bf = BloomFilter::with_params(100, 2);

        assert!(bf.insert(&1).unwrap());
        assert!(!bf.insert(&1).unwrap());
        assert!(bf.query(&1));
    }

    #[test]
    fn clear() {
        let mut bf = BloomFilter::with_params(100, 2);

        bf.insert(&1).unwrap();
        bf.clear();
        assert!(!bf.query(&1));
        assert!(bf.is_empty());
    }

    #[test]
    fn is_empty() {
        let mut bf = BloomFilter::with_params(100, 2);
        assert!(bf.is_empty());

        bf.insert(&1).unwrap();
        assert!(!bf.is_empty());

        bf.clear();
        assert!(bf.is_empty());
    }

    #[test]
    fn clone() {
        let mut bf1 = BloomFilter::with_params(100, 2);
        bf1.insert(&1).unwrap();

        let bf2 = bf1.clone();
        bf1.insert(&2).unwrap();
        assert!(bf2.query(&1));
        assert!(!bf2.query(&2));
    }

    #[test]
    fn union() {
        let mut bf1 = BloomFilter::with_params(100, 2);
        bf1.insert(&1).unwrap();
        assert!(bf1.query(&1));
        assert!(!bf1.query(&2));
        assert!(!bf1.query(&3));

        let mut bf2 = BloomFilter::with_params(100, 2);
        bf2.insert(&2).unwrap();
        assert!(!bf2.query(&1));
        assert!(bf2.query(&2));
        assert!(!bf2.query(&3));

        bf1.union(&bf2).unwrap();
        assert!(bf1.query(&1));
        assert!(bf1.query(&2));
        assert!(!bf1.query(&3));
    }

    #[test]
    #[should_panic(expected = "k must be equal (left=2, right=3)")]
    fn union_panics_k() {
        let mut bf1 = BloomFilter::<u64>::with_params(100, 2);
        let bf2 = BloomFilter::<u64>::with_params(100, 3);
        bf1.union(&bf2).unwrap();
    }

    #[test]
    #[should_panic(expected = "m must be equal (left=100, right=200)")]
    fn union_panics_m() {
        let mut bf1 = BloomFilter::<u64>::with_params(100, 2);
        let bf2 = BloomFilter::<u64>::with_params(200, 2);
        bf1.union(&bf2).unwrap();
    }

    #[test]
    #[should_panic(expected = "buildhasher must be equal")]
    fn union_panics_buildhasher() {
        let mut bf1 = BloomFilter::<u64, BuildHasherSeeded>::with_params_and_hash(
            100,
            2,
            BuildHasherSeeded::new(0),
        );
        let bf2 = BloomFilter::<u64, BuildHasherSeeded>::with_params_and_hash(
            100,
            2,
            BuildHasherSeeded::new(1),
        );
        bf1.union(&bf2).unwrap();
    }

    #[test]
    fn with_properties() {
        let bf = BloomFilter::<u64>::with_properties(1000, 0.1);
        assert_eq!(bf.k(), 3);
        assert_eq!(bf.m(), 4792);
    }

    #[test]
    #[should_panic(expected = "n must be greater than 0")]
    fn with_properties_panics_n0() {
        BloomFilter::<u64>::with_properties(0, 0.1);
    }

    #[test]
    #[should_panic(expected = "p (0) must be greater than 0 and smaller than 1")]
    fn with_properties_panics_p0() {
        BloomFilter::<u64>::with_properties(1000, 0.);
    }

    #[test]
    #[should_panic(expected = "p (1) must be greater than 0 and smaller than 1")]
    fn with_properties_panics_p1() {
        BloomFilter::<u64>::with_properties(1000, 1.);
    }

    #[test]
    fn len() {
        let mut bf = BloomFilter::with_params(100, 2);
        assert_eq!(bf.len(), 0);

        bf.insert(&1).unwrap();
        assert_eq!(bf.len(), 1);

        bf.insert(&1).unwrap();
        assert_eq!(bf.len(), 1);

        bf.insert(&2).unwrap();
        assert_eq!(bf.len(), 2);
    }

    #[test]
    fn debug() {
        let bf = BloomFilter::<u64>::with_params(100, 2);
        assert_eq!(format!("{:?}", bf), "BloomFilter { m: 100, k: 2 }");
    }

    #[test]
    fn extend() {
        let mut bf = BloomFilter::<u64>::with_params(100, 2);

        bf.extend(vec![1, 2]);
        assert!(bf.query(&1));
        assert!(bf.query(&2));
        assert!(!bf.query(&3));
    }

    #[test]
    fn insert_unsized() {
        let mut bf = BloomFilter::with_params(100, 2);

        assert!(bf.insert("test1").unwrap());
        assert!(bf.query("test1"));
        assert!(!bf.query("test2"));
    }

    #[test]
    fn send() {
        let bf = BloomFilter::<NotSend>::with_params(100, 2);
        assert_send(&bf);
    }

    #[test]
    fn bitmap_save_load() {
        let mut bf = BloomFilter::with_params(100, 2);

        assert!(bf.insert(&1).unwrap());
        assert!(bf.insert(&7).unwrap());
        assert!(bf.insert(&52).unwrap());

        let bitmap = bf.bitmap().to_vec();

        let loaded_bf = BloomFilter::with_existing_filter(100, 2, bitmap);

        assert!(loaded_bf.query(&1));
        assert!(loaded_bf.query(&7));
        assert!(loaded_bf.query(&52));
        assert!(!loaded_bf.query(&15));
    }
}
