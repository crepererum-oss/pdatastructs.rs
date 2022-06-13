//! Hash-related utils.
use std::any::TypeId;
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

/// Builder for iterators for multiple hash function results for the same given objects.
///
/// In other words: for a given object `x`, HashIterBuilder creates an iterator that emits
/// `h_i(x) in [0, m)` for `i in [0, k)` with `m` and `k` being configurable.
///
/// # Examples
/// ```
/// use pdatastructs::hash_utils::HashIterBuilder;
/// use std::collections::hash_map::DefaultHasher;
/// use std::hash::BuildHasherDefault;
///
/// // set up builder
/// let bh = BuildHasherDefault::<DefaultHasher>::default();
/// let max_result = 42;
/// let number_functions = 2;
/// let builder = HashIterBuilder::new(max_result, number_functions, bh);
///
/// // create iterator for a given object
/// let obj = "my string";
/// let iter = builder.iter_for(&obj);
/// for h_i in iter {
///     println!("{}", h_i);
/// }
/// ```
///
/// # Applications
/// - mostly used as helper for BloomFilter and CountMinSketch
///
/// # How It Works
/// Instead of hashing `x` `i` times with different hash functions, only 2 hash functions `h_1` and
/// `h_2` are used. Furthermore, a function `f: [0, k) -> [0, m)` is generated in a pseudo-random,
/// but derministic manner. Then, the result of `i in [0, k)` is:
///
/// ```text
/// (h_1(x) + i * h_2(x) + f(i)) mod m
/// ```
///
/// This is also called "enhanced double hashing".
///
/// # See Also
/// - `std::collections::hash_map::DefaultHasher`: a solid choice to produce some hash results,
///   but slower for the special case needed for BloomFilter etc.
///
/// # References
/// - ["Less Hashing, Same Performance: Building a Better Bloom Filter", Adam Kirsch, Michael
///   Mitzenmacher, 2008](https://www.eecs.harvard.edu/%7Emichaelm/postscripts/rsa2008.pdf)
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashIterBuilder<B>
where
    B: BuildHasher,
{
    m: usize,
    k: usize,
    buildhasher: B,
    f: Vec<u64>,
}

impl<B> HashIterBuilder<B>
where
    B: BuildHasher,
{
    /// Create new `HashIterBuilder` with the following parameters:
    ///
    /// - `m`: the maximal result value of `h_i(x)`
    /// - `k`: number of hash functions to generate
    /// - `buildhasher`: `BuildHasher` used to construct new `Hash` objects, must be stable (i.e. create the same `Hash`
    ///   object on every call)
    pub fn new(m: usize, k: usize, buildhasher: B) -> Self {
        let f = Self::setup_f(m, k, &buildhasher);
        Self {
            m,
            k,
            buildhasher,
            f,
        }
    }

    /// Domain aka maximal result of `h_i(x)`.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of hash functions.
    pub fn k(&self) -> usize {
        self.k
    }

    /// `BuildHasher` that is used to hash objects and to create `f`.
    pub fn buildhasher(&self) -> &B {
        &self.buildhasher
    }

    /// Shift-function to enhance the hashing scheme.
    pub fn f(&self, i: usize) -> u64 {
        self.f[i]
    }

    /// Create an iterator for `k` hash functions for the given object.
    ///
    /// This call is CPU-expensive and may only be used if the resulting iterator will really be
    /// used.
    pub fn iter_for<T>(&self, obj: &T) -> HashIter<'_, B>
    where
        T: Hash + ?Sized,
    {
        let h1 = self.h_i(&obj, 0) % (self.m as u64);
        let h2 = self.h_i(&obj, 1) % (self.m as u64);
        HashIter::new(self, h1, h2)
    }

    /// Set up `f`.
    fn setup_f(m: usize, k: usize, buildhasher: &B) -> Vec<u64> {
        (0..k)
            .map(|i| {
                let mut hasher = buildhasher.build_hasher();
                hasher.write_usize(i + 2); // skip 2 for h1 und h2
                hasher.finish() % (m as u64)
            })
            .collect()
    }

    /// Helper to calculate h_0 and h_1.
    fn h_i<T>(&self, obj: &T, i: usize) -> u64
    where
        T: Hash + ?Sized,
    {
        let mut hasher = self.buildhasher.build_hasher();
        hasher.write_usize(i);
        obj.hash(&mut hasher);
        hasher.finish()
    }
}

/// Iterate over `h_i(x) in [0, k)` for `i in [0, m)`.
#[derive(Debug)]
pub struct HashIter<'a, B>
where
    B: BuildHasher,
{
    builder: &'a HashIterBuilder<B>,
    h1: u64,
    h2: u64,
    i: usize,
}

impl<'a, B> HashIter<'a, B>
where
    B: 'a + BuildHasher,
{
    /// Create new `HashIter` with the following parameters:
    ///
    /// - `builder`: builder
    /// - `h1`: first hash for double hashing
    /// - `h2`: second hash for double hashing
    fn new(builder: &'a HashIterBuilder<B>, h1: u64, h2: u64) -> Self {
        Self {
            builder,
            h1,
            h2,
            i: 0,
        }
    }
}

impl<'a, B> Iterator for HashIter<'a, B>
where
    B: 'a + BuildHasher,
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.i < self.builder.k() {
            let m = self.builder.m() as u64;
            let i = (self.i as u64) % m;
            let f = self.builder.f(self.i);
            let x = (self.h1 + (i * self.h2) + f) % m;

            self.i += 1;

            Some(x as usize)
        } else {
            None
        }
    }
}

/// BuildHasher that takes a seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildHasherSeeded {
    seed: usize,
}

impl BuildHasherSeeded {
    /// Create new BuildHasherSeeded with given seed.
    pub fn new(seed: usize) -> Self {
        Self { seed }
    }
}

impl BuildHasher for BuildHasherSeeded {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        let mut h = DefaultHasher::default();
        h.write_usize(self.seed);
        h
    }
}

/// Hash placeholder that can be used to make certain data structures accept multiple data types.
///
/// # Example
/// ```
/// use pdatastructs::filters::Filter;
/// use pdatastructs::filters::bloomfilter::BloomFilter;
/// use pdatastructs::hash_utils::AnyHash;
///
/// // set up filter
/// let false_positive_rate = 0.02;  // = 2%
/// let expected_elements = 1000;
/// let mut filter = BloomFilter::<AnyHash>::with_properties(expected_elements, false_positive_rate);
///
/// // add different types
/// filter.insert(&AnyHash::new("1"));
/// filter.insert(&AnyHash::new(&1u64));
///
/// // query
/// assert!(filter.query(&AnyHash::new("1")));
/// assert!(filter.query(&AnyHash::new(&1u64)));
/// assert!(!filter.query(&AnyHash::new(&1u32)));
/// ```
///
/// # How It Works
/// During construction, a 64bit hash of the object as well as the `TypeID` of the object type will
/// be preserved. Both will later be fed into the target hasher of the used data structure.
#[derive(Debug)]
pub struct AnyHash<H = DefaultHasher>
where
    H: Hasher + Default,
{
    hash_obj: u64,
    type_id: TypeId,
    phantom: PhantomData<H>,
}

impl<H> AnyHash<H>
where
    H: Hasher + Default,
{
    /// Create new hash placeholder for given object.
    pub fn new<T>(obj: &T) -> Self
    where
        T: Hash + 'static + ?Sized,
    {
        let mut hasher_obj = H::default();
        obj.hash(&mut hasher_obj);
        let hash_obj = hasher_obj.finish();

        Self {
            hash_obj,
            type_id: TypeId::of::<T>(),
            phantom: PhantomData,
        }
    }
}

impl Hash for AnyHash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        self.hash_obj.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::{BuildHasherSeeded, HashIterBuilder};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

    #[test]
    fn hash_iter_builder_getter() {
        let bh1 = BuildHasherDefault::<DefaultHasher>::default();
        let builder = HashIterBuilder::new(42, 2, bh1);

        assert_eq!(builder.m(), 42);
        assert_eq!(builder.k(), 2);

        let bh2 = BuildHasherDefault::<DefaultHasher>::default();
        assert_eq!(builder.buildhasher(), &bh2);
    }

    #[test]
    fn hash_iter_builder_f() {
        let bh1 = BuildHasherDefault::<DefaultHasher>::default();
        let bh2 = BuildHasherDefault::<DefaultHasher>::default();
        let builder1 = HashIterBuilder::new(42, 2, bh1);
        let builder2 = HashIterBuilder::new(42, 2, bh2);

        assert_eq!(builder1.f(0), builder2.f(0));
        assert_eq!(builder1.f(1), builder2.f(1));
    }

    #[test]
    fn hash_iter_builder_eq() {
        let bh1 = BuildHasherSeeded::new(0);
        let bh2 = BuildHasherSeeded::new(0);
        let bh3 = BuildHasherSeeded::new(0);
        let bh4 = BuildHasherSeeded::new(1);
        let builder1 = HashIterBuilder::new(42, 2, bh1);
        let builder2 = HashIterBuilder::new(42, 2, bh2);
        let builder3 = HashIterBuilder::new(42, 3, bh3);
        let builder4 = HashIterBuilder::new(42, 2, bh4);

        assert_eq!(builder1, builder2);
        assert_ne!(builder1, builder3);
        assert_ne!(builder1, builder4);
    }

    #[test]
    fn hash_iter() {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        let obj = 1337;

        let builder = HashIterBuilder::new(42, 2, bh);

        let iter1 = builder.iter_for(&obj);
        let v1: Vec<usize> = iter1.collect();
        assert_eq!(v1.len(), 2);
        assert!(v1[0] < 42);
        assert!(v1[1] < 42);
        assert_ne!(v1[0], v1[1]);

        let iter2 = builder.iter_for(&obj);
        let v2: Vec<usize> = iter2.collect();
        assert_eq!(v1, v2);
    }

    #[test]
    fn build_hasher_seeded() {
        let bh1 = BuildHasherSeeded::new(0);
        let bh2 = BuildHasherSeeded::new(0);
        let bh3 = BuildHasherSeeded::new(1);

        let mut hasher1 = bh1.build_hasher();
        let mut hasher2 = bh2.build_hasher();
        let mut hasher3 = bh3.build_hasher();

        let obj = "foo bar";
        obj.hash(&mut hasher1);
        obj.hash(&mut hasher2);
        obj.hash(&mut hasher3);

        let result1 = hasher1.finish();
        let result2 = hasher2.finish();
        let result3 = hasher3.finish();

        assert_eq!(result1, result2);
        assert_ne!(result1, result3);
    }
}
