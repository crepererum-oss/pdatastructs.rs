//! CuckooFilter implementation.
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;

use rand::Rng;
use succinct::{BitVec, BitVecMut, IntVec, IntVecMut, IntVector};

use crate::filters::Filter;
use crate::helpers::all_zero_intvector;

const MAX_NUM_KICKS: usize = 500; // mentioned in paper

/// Error struct used to signal that a `CuckooFilter` is full, i.e. that a value cannot be inserted
/// because the implementation was unable to find a free bucket.
#[derive(Debug, Clone, Copy)]
pub struct CuckooFilterFull;

/// A CuckooFilter is a set-like data structure, that keeps track of elements it has seen without
/// the need to store them. Looking up values has a certain false positive rate, but a false
/// negative rate of 0%. Also, it is "Practically Better Than Bloom" (see paper).
///
/// # Examples
/// ```
/// use pdatastructs::filters::Filter;
/// use pdatastructs::filters::cuckoofilter::CuckooFilter;
/// use pdatastructs::rand::SeedableRng;
/// use rand_chacha::ChaChaRng;
///
/// // set up filter
/// let false_positive_rate = 0.02;  // = 2%
/// let expected_elements = 1000;
/// let rng = ChaChaRng::from_seed([0; 32]);
/// let mut filter = CuckooFilter::with_properties_8(false_positive_rate, expected_elements, rng);
///
/// // add some data
/// filter.insert(&"my super long string");
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
/// use pdatastructs::filters::cuckoofilter::CuckooFilter;
/// use pdatastructs::rand::SeedableRng;
/// use rand_chacha::ChaChaRng;
///
/// // set up filter
/// let false_positive_rate = 0.02;  // = 2%
/// let expected_elements = 1000;
/// let rng = ChaChaRng::from_seed([0; 32]);
/// let mut filter1 = CuckooFilter::<u8, _>::with_properties_8(false_positive_rate, expected_elements, rng);
/// let filter2 = CuckooFilter::<i8, _>::with_properties_8(false_positive_rate, expected_elements, rng);
///
/// filter1.union(&filter2);
/// ```
///
/// # Applications
/// Same as BloomFilter.
///
/// # How It Works
///
/// ## Setup
/// The filter is created by a table of `b` buckets, each bucket having `s` slots. Every slot can
/// hold `n` bits, all set to 0 in the beginning.
///
/// ```text
/// b = 4
/// s = 2
/// n = 4
///
/// +-----+-----+
/// | 0x0 | 0x0 |
/// +-----+-----+
/// | 0x0 | 0x0 |
/// +-----+-----+
/// | 0x0 | 0x0 |
/// +-----+-----+
/// | 0x0 | 0x0 |
/// +-----+-----+
/// ```
///
/// ## Insertion
/// During the insertion, a fingerprint of `n` bits is calculated. Furthermore, there are 2 hash
/// functions. The first one is calculated by an external hash function (e.g. SipHash), the other
/// is the XOR of the fingerprint and the first hash. That way, you can switch between the 2 hash
/// functions by XORing with the fingerprint.
///
/// The 2 hash functions address 2 candidate buckets. If one has a free slot, the fingerprint will
/// be added there and we are done.
///
/// ```text
/// f(x)   = 0x3
/// h_1(x) = 0x0
/// h_2(x) = (h_1(x) ^ f(x)) & 0x3 = (0x0 ^ 0x3) & 0x3 = 0x3
///
/// +-----+-----+       +-----+-----+
/// |[0x0]| 0x0 |       | 0x3 | 0x0 |
/// +-----+-----+       +-----+-----+
/// | 0x0 | 0x0 |       | 0x0 | 0x0 |
/// +-----+-----+  ==>  +-----+-----+
/// | 0x0 | 0x0 |       | 0x0 | 0x0 |
/// +-----+-----+       +-----+-----+
/// |[0x0]| 0x0 |       | 0x0 | 0x0 |
/// +-----+-----+       +-----+-----+
/// ```
///
/// It can occur that different elements address the same buckets. The fingerprint helps to
/// distinguish between them.
///
/// ```text
/// f(x)   = 0x7
/// h_1(x) = 0x0
/// h_2(x) = (h_1(x) ^ f(x)) & 0x3 = (0x0 ^ 0x7) & 0x3 = 0x3
///
/// +-----+-----+       +-----+-----+
/// |[0x0]| 0x0 |       | 0x3 | 0x7 |
/// +-----+-----+       +-----+-----+
/// | 0x0 | 0x0 |       | 0x0 | 0x0 |
/// +-----+-----+  ==>  +-----+-----+
/// | 0x0 | 0x0 |       | 0x0 | 0x0 |
/// +-----+-----+       +-----+-----+
/// |[0x0]| 0x0 |       | 0x0 | 0x0 |
/// +-----+-----+       +-----+-----+
/// ```
///
/// If none of the 2 candidate buckets has a free slot, the algorithm tries to relocate existing
/// fingerprints by swapping hash functions (remember that `h_i(x) = h_j(x) ^ f(x)`). The element
/// to relocate is chosen randomly.
///
/// ## Lookup
/// Hash functions are calculated as shown above and if one of the buckets contains the
/// fingerprint, the element may be in the filter.
///
/// # Implementation
/// This implementation uses one large bit-vector to pack all fingerprints to be as space-efficient
/// as possible. So even odd fingerprints sizes like 5 result in optimal memory consumption.
///
/// # See Also
/// - `std::collections::HashSet`: has a false positive rate of 0%, but also needs to store all
///   elements
/// - `pdatastructs::bloomfilter::BloomFilter`: simpler data structure, but might have worse false
///   positive rates
///
/// # References
/// - [Probabilistic Filters By Example](https://bdupras.github.io/filter-tutorial/)
/// - ["Cuckoo Filter: Practically Better Than Bloom", Bin Fan, David G. Andersen, Michael
///   Kaminsky, Michael D. Mitzenmacher, 2014](https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf).
#[derive(Clone)]
pub struct CuckooFilter<T, R, B = BuildHasherDefault<DefaultHasher>>
where
    T: Hash + ?Sized,
    R: Rng,
    B: BuildHasher + Clone + Eq,
{
    table: IntVector<u64>,
    n_elements: usize,
    buildhasher: B,
    bucketsize: usize,
    n_buckets: usize,
    l_fingerprint: usize,
    rng: R,
    phantom: PhantomData<fn() -> T>,
}

impl<T, R> CuckooFilter<T, R>
where
    T: Hash + ?Sized,
    R: Rng,
{
    /// Create new CuckooFilter with:
    ///
    /// - `rng`: random number generator used for certain random actions
    /// - `bucketsize`: number of elements per bucket, must be at least 2
    /// - `n_buckets`: number of buckets, must be a power of 2 and at least 2
    /// - `l_fingerprint`: size of the fingerprint in bits
    ///
    /// The BuildHasher is set to the `DefaultHasher`.
    pub fn with_params(rng: R, bucketsize: usize, n_buckets: usize, l_fingerprint: usize) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_params_and_hash(rng, bucketsize, n_buckets, l_fingerprint, bh)
    }

    /// Create CuckooFilter with existing filter table data:
    ///
    /// - `rng`: random number generator used for certain random actions
    /// - `bucketsize`: number of elements per bucket, must be at least 2
    /// - `n_buckets`: number of buckets, must be a power of 2 and at least 2
    /// - `l_fingerprint`: size of the fingerprint in bits
    /// - `n_elements`: number of elements in existing filter
    /// - `table_succinct_blocks`: filter table block data
    ///
    /// The BuildHasher is set to the `DefaultHasher`.
    pub fn with_existing_filter<I: IntoIterator<Item = u64>>(
        rng: R,
        bucketsize: usize,
        n_buckets: usize,
        l_fingerprint: usize,
        n_elements: usize,
        table_succinct_blocks: I,
    ) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_existing_filter_and_hash(
            rng,
            bucketsize,
            n_buckets,
            l_fingerprint,
            n_elements,
            table_succinct_blocks,
            bh,
        )
    }

    /// Construct new `bucketsize=4`-cuckoofilter with properties:
    ///
    /// - `false_positive_rate`: false positive lookup rate
    /// - `expected_elements`: number of expected elements to be added to the filter
    /// - `rng`: random number generator used for certain random actions
    pub fn with_properties_4(false_positive_rate: f64, expected_elements: usize, rng: R) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_properties_and_hash_4(false_positive_rate, expected_elements, rng, bh)
    }

    /// Construct new `bucketsize=8`-cuckoofilter with properties:
    ///
    /// - `false_positive_rate`: false positive lookup rate
    /// - `expected_elements`: number of expected elements to be added to the filter
    /// - `rng`: random number generator used for certain random actions
    pub fn with_properties_8(false_positive_rate: f64, expected_elements: usize, rng: R) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_properties_and_hash_8(false_positive_rate, expected_elements, rng, bh)
    }
}

impl<T, R, B> CuckooFilter<T, R, B>
where
    T: Hash + ?Sized,
    R: Rng,
    B: BuildHasher + Clone + Eq,
{
    /// Create new CuckooFilter with:
    ///
    /// - `rng`: random number generator used for certain random actions
    /// - `bucketsize`: number of elements per bucket, must be at least 2
    /// - `n_buckets`: number of buckets, must be a power of 2 and at least 2
    /// - `l_fingerprint`: size of the fingerprint in bits
    /// - `bh`: BuildHasher that creates Hash objects, used for fingerprint creation and
    ///   fingerprint hashing
    pub fn with_params_and_hash(
        rng: R,
        bucketsize: usize,
        n_buckets: usize,
        l_fingerprint: usize,
        bh: B,
    ) -> Self {
        assert!(
            bucketsize >= 2,
            "bucketsize ({}) must be greater or equal than 2",
            bucketsize
        );
        assert!(
            n_buckets.is_power_of_two() & (n_buckets >= 2),
            "n_buckets ({}) must be a power of 2 and greater or equal than 2",
            n_buckets
        );
        assert!(
            (l_fingerprint > 1) & (l_fingerprint <= 64),
            "l_fingerprint ({}) must be greater than 1 and less or equal than 64",
            l_fingerprint
        );

        let table_size = n_buckets
            .checked_mul(bucketsize)
            .expect("Table size too large");

        Self {
            table: all_zero_intvector(l_fingerprint, table_size),
            n_elements: 0,
            buildhasher: bh,
            bucketsize,
            n_buckets,
            l_fingerprint,
            rng,
            phantom: PhantomData,
        }
    }

    /// Same as `with_existing_filter` but with specific `BuildHasher`.
    pub fn with_existing_filter_and_hash<I: IntoIterator<Item = u64>>(
        rng: R,
        bucketsize: usize,
        n_buckets: usize,
        l_fingerprint: usize,
        n_elements: usize,
        table_succinct_blocks: I,
        bh: B,
    ) -> Self {
        let mut filter = Self::with_params_and_hash(rng, bucketsize, n_buckets, l_fingerprint, bh);
        for (i, block) in table_succinct_blocks.into_iter().enumerate() {
            assert!(
                i < filter.table.block_len(),
                "existing input table block length must not exceed filter table block length"
            );
            filter.table.set_block(i, block);
        }
        filter.n_elements = n_elements;
        filter
    }

    /// Construct new `bucketsize=4`-cuckoofilter with properties:
    ///
    /// - `false_positive_rate`: false positive lookup rate
    /// - `expected_elements`: number of expected elements to be added to the filter
    /// - `rng`: random number generator used for certain random actions
    /// - `bh`: BuildHasher that creates Hash objects, used for fingerprint creation and
    ///   fingerprint hashing
    pub fn with_properties_and_hash_4(
        false_positive_rate: f64,
        expected_elements: usize,
        rng: R,
        bh: B,
    ) -> Self {
        let bucketsize = 4usize;
        let load_factor = 0.95f64;
        Self::with_properties_and_hash_n(
            bucketsize,
            load_factor,
            false_positive_rate,
            expected_elements,
            rng,
            bh,
        )
    }

    /// Construct new `bucketsize=8`-cuckoofilter with properties:
    ///
    /// - `false_positive_rate`: false positive lookup rate
    /// - `expected_elements`: number of expected elements to be added to the filter
    /// - `rng`: random number generator used for certain random actions
    /// - `bh`: BuildHasher that creates Hash objects, used for fingerprint creation and
    ///   fingerprint hashing
    pub fn with_properties_and_hash_8(
        false_positive_rate: f64,
        expected_elements: usize,
        rng: R,
        bh: B,
    ) -> Self {
        let bucketsize = 8usize;
        let load_factor = 0.98f64;
        Self::with_properties_and_hash_n(
            bucketsize,
            load_factor,
            false_positive_rate,
            expected_elements,
            rng,
            bh,
        )
    }

    fn with_properties_and_hash_n(
        bucketsize: usize,
        load_factor: f64,
        false_positive_rate: f64,
        expected_elements: usize,
        rng: R,
        bh: B,
    ) -> Self {
        assert!(
            expected_elements >= 1,
            "expected_elements ({}) must be at least 1",
            expected_elements
        );
        assert!(
            (false_positive_rate > 0.) && (false_positive_rate < 1.),
            "false_positive_rate ({}) must be greater than 0 and smaller than 1",
            false_positive_rate
        );

        let l_fingerprint = (2.0 * (bucketsize as f64) / false_positive_rate)
            .log2()
            .ceil() as usize;
        let costs = (l_fingerprint as f64) / load_factor;
        let n_buckets = ((costs * (expected_elements as f64) / (l_fingerprint as f64)).ceil()
            as usize)
            .next_power_of_two();
        Self::with_params_and_hash(rng, bucketsize, n_buckets, l_fingerprint, bh)
    }

    /// Number of entries stored in a bucket.
    pub fn bucketsize(&self) -> usize {
        self.bucketsize
    }

    /// Number of buckets used by the CuckooFilter.
    pub fn n_buckets(&self) -> usize {
        self.n_buckets
    }

    /// Size of the used fingerprint in bits
    pub fn l_fingerprint(&self) -> usize {
        self.l_fingerprint
    }

    /// Remove element from the filter.
    ///
    /// Returns `true` if element was in the filter, `false` if it was not in which case the operation did not modify
    /// the filter.
    pub fn delete(&mut self, t: &T) -> bool {
        let (f, i1, i2) = self.start(t);

        if self.remove_from_bucket(i1, f) {
            self.n_elements -= 1;
            return true;
        }
        if self.remove_from_bucket(i2, f) {
            self.n_elements -= 1;
            return true;
        }
        false
    }

    fn start(&self, t: &T) -> (u64, usize, usize) {
        let f = self.fingerprint(t);
        let i1 = self.hash(t);
        let i2 = i1 ^ self.hash(&f);
        (f, i1, i2)
    }

    fn fingerprint(&self, t: &T) -> u64 {
        let mut hasher = self.buildhasher.build_hasher();
        hasher.write_usize(0); // IV
        t.hash(&mut hasher);

        // don't produce 0, since this is used as "free"-slot value
        let x_mod = if self.l_fingerprint == 64 {
            u64::max_value()
        } else {
            (1u64 << self.l_fingerprint) - 1
        };
        1 + (hasher.finish() % x_mod)
    }

    fn hash<U>(&self, obj: &U) -> usize
    where
        U: Hash + ?Sized,
    {
        let mut hasher = self.buildhasher.build_hasher();
        hasher.write_usize(1); // IV
        obj.hash(&mut hasher);
        (hasher.finish() & (self.n_buckets as u64 - 1)) as usize
    }

    fn write_to_bucket(&mut self, i: usize, f: u64) -> bool {
        let offset = i * self.bucketsize;
        for x in offset..(offset + self.bucketsize) {
            if self.table.get(x as u64) == 0 {
                self.table.set(x as u64, f);
                return true;
            }
        }
        false
    }

    fn has_in_bucket(&self, i: usize, f: u64) -> bool {
        let offset = i * self.bucketsize;
        for x in offset..(offset + self.bucketsize) {
            if self.table.get(x as u64) == f {
                return true;
            }
        }
        false
    }

    fn remove_from_bucket(&mut self, i: usize, f: u64) -> bool {
        let offset = i * self.bucketsize;
        for x in offset..(offset + self.bucketsize) {
            if self.table.get(x as u64) == f {
                self.table.set(x as u64, 0);
                return true;
            }
        }
        false
    }

    fn insert_internal(
        &mut self,
        mut f: u64,
        i1: usize,
        i2: usize,
        log: &mut Vec<(usize, u64)>,
    ) -> Result<bool, CuckooFilterFull> {
        if self.write_to_bucket(i1, f) {
            self.n_elements += 1;
            return Ok(true);
        }
        if self.write_to_bucket(i2, f) {
            self.n_elements += 1;
            return Ok(false);
        }

        // cannot write to obvious buckets => relocate
        let mut i = if self.rng.gen::<bool>() { i1 } else { i2 };

        for _ in 0..MAX_NUM_KICKS {
            let e: usize = self.rng.gen_range(0..self.bucketsize);
            let offset = i * self.bucketsize;
            let x = offset + e;

            // swap table[x] and f
            let tmp = self.table.get(x as u64);
            log.push((x, tmp));
            self.table.set(x as u64, f);
            f = tmp;

            i ^= self.hash(&f);
            if self.write_to_bucket(i, f) {
                self.n_elements += 1;
                return Ok(true);
            }
        }

        // no space left => fail
        Err(CuckooFilterFull)
    }

    fn restore_state(&mut self, log: &[(usize, u64)]) {
        for (pos, data) in log.iter().rev().cloned() {
            self.table.set(pos as u64, data);
        }
    }

    /// Clear and load filter table with individual filter table elements
    /// and existing element count.
    pub fn load_table<I: IntoIterator<Item = u64>>(&mut self, table: I, n_elements: usize) {
        self.clear();
        for (i, value) in table.into_iter().enumerate() {
            let i = i as u64;
            assert!(
                i < self.table.len(),
                "input table length must not exceed filter table length"
            );
            self.table.set(i, value);
        }
        self.n_elements = n_elements;
    }

    /// Return the individual filter table elements.
    pub fn table(&self) -> Vec<u64> {
        self.table.iter().collect()
    }

    /// Return the filter table succinct block data.
    pub fn table_succinct_blocks(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.table.block_len());
        for i in 0..self.table.block_len() {
            result.push(self.table.get_block(i));
        }
        result
    }
}

impl<T, R, B> Filter<T> for CuckooFilter<T, R, B>
where
    T: Hash + ?Sized,
    R: Rng,
    B: BuildHasher + Clone + Eq,
{
    type InsertErr = CuckooFilterFull;

    fn clear(&mut self) {
        self.n_elements = 0;
        self.table = IntVector::with_fill(self.table.element_bits(), self.table.len(), 0);
    }

    /// Insert new element into filter.
    ///
    /// The method may return an error if it was unable to find a free bucket. This means the
    /// filter is full and you should not add any additional elements to it. When this happens,
    /// `len` was not increased and the filter content was not altered.
    ///
    /// Inserting the same element multiple times is supported, but keep in mind that after
    /// `n_buckets * 2` times, the filter will return `Err(CuckooFilterFull)`. Also, this function
    /// will always report `Ok(true)` in case of success, even if the same element is inserted
    /// twice.
    fn insert(&mut self, obj: &T) -> Result<bool, Self::InsertErr> {
        let (f, i1, i2) = self.start(obj);
        let mut log: Vec<(usize, u64)> = vec![];
        let result = self.insert_internal(f, i1, i2, &mut log);
        if result.is_err() {
            self.restore_state(&log);
        }
        result
    }

    fn union(&mut self, other: &Self) -> Result<(), Self::InsertErr> {
        assert_eq!(
            self.bucketsize, other.bucketsize,
            "bucketsize must be equal (left={}, right={})",
            self.bucketsize, other.bucketsize
        );
        assert_eq!(
            self.n_buckets, other.n_buckets,
            "n_buckets must be equal (left={}, right={})",
            self.n_buckets, other.n_buckets
        );
        assert_eq!(
            self.l_fingerprint, other.l_fingerprint,
            "l_fingerprint must be equal (left={}, right={})",
            self.l_fingerprint, other.l_fingerprint
        );
        assert!(
            self.buildhasher == other.buildhasher,
            "buildhasher must be equal",
        );

        let mut log: Vec<(usize, u64)> = vec![];
        let n_elements_backup = self.n_elements;
        let mut i1: usize = 0;
        for (counter, f) in other.table.iter().enumerate() {
            // calculate current bucket
            if (counter > 0) && (counter % other.bucketsize == 0) {
                i1 += 1;
            }

            // check if slot is used
            if f != 0 {
                let i2 = i1 ^ other.hash(&f);
                if let Err(err) = self.insert_internal(f, i1, i2, &mut log) {
                    self.restore_state(&log);
                    self.n_elements = n_elements_backup;
                    return Err(err);
                }
            }
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.n_elements == 0
    }

    /// Return exact number of elements in the filter.
    fn len(&self) -> usize {
        self.n_elements
    }

    fn query(&self, obj: &T) -> bool {
        let (f, i1, i2) = self.start(obj);

        if self.has_in_bucket(i1, f) {
            return true;
        }
        if self.has_in_bucket(i2, f) {
            return true;
        }
        false
    }
}

impl<T, R, B> fmt::Debug for CuckooFilter<T, R, B>
where
    T: Hash + ?Sized,
    R: Rng,
    B: BuildHasher + Clone + Eq,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CuckooFilter {{ bucketsize: {}, n_buckets: {} }}",
            self.bucketsize, self.n_buckets
        )
    }
}

#[cfg(test)]
mod tests {
    use super::CuckooFilter;
    use crate::{
        filters::Filter,
        hash_utils::BuildHasherSeeded,
        test_util::{assert_send, NotSend},
    };
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    #[test]
    #[should_panic(expected = "bucketsize (0) must be greater or equal than 2")]
    fn new_panics_bucketsize_0() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 0, 16, 8);
    }

    #[test]
    #[should_panic(expected = "bucketsize (1) must be greater or equal than 2")]
    fn new_panics_bucketsize_1() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 1, 16, 8);
    }

    #[test]
    #[should_panic(expected = "n_buckets (0) must be a power of 2 and greater or equal than 2")]
    fn new_panics_n_buckets_0() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 0, 8);
    }

    #[test]
    #[should_panic(expected = "n_buckets (1) must be a power of 2 and greater or equal than 2")]
    fn new_panics_n_buckets_1() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 1, 8);
    }

    #[test]
    #[should_panic(expected = "n_buckets (5) must be a power of 2 and greater or equal than 2")]
    fn new_panics_n_buckets_5() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 5, 8);
    }

    #[test]
    #[should_panic(expected = "l_fingerprint (0) must be greater than 1 and less or equal than 64")]
    fn new_panics_l_fingerprint_0() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 0);
    }

    #[test]
    #[should_panic(expected = "l_fingerprint (1) must be greater than 1 and less or equal than 64")]
    fn new_panics_l_fingerprint_1() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 1);
    }

    #[test]
    #[should_panic(
        expected = "l_fingerprint (65) must be greater than 1 and less or equal than 64"
    )]
    fn new_panics_l_fingerprint_65() {
        CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 65);
    }

    #[test]
    #[should_panic(expected = "Table size too large")]
    fn new_panics_table_size_overflow_1() {
        CuckooFilter::<u64, ChaChaRng>::with_params(
            ChaChaRng::from_seed([0; 32]),
            usize::max_value(),
            2,
            2,
        );
    }

    #[test]
    #[should_panic(expected = "Table size too large")]
    fn new_panics_table_size_overflow_2() {
        CuckooFilter::<u64, ChaChaRng>::with_params(
            ChaChaRng::from_seed([0; 32]),
            2,
            (((usize::max_value() as u128) + 1) / 2) as usize,
            2,
        );
    }

    #[test]
    #[should_panic(expected = "Table size too large")]
    fn new_panics_table_size_overflow_3() {
        CuckooFilter::<u64, ChaChaRng>::with_params(
            ChaChaRng::from_seed([0; 32]),
            2,
            (((usize::max_value() as u128) + 1) / 8) as usize,
            64,
        );
    }

    #[test]
    fn getter() {
        let cf =
            CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert_eq!(cf.bucketsize(), 2);
        assert_eq!(cf.n_buckets(), 16);
        assert_eq!(cf.l_fingerprint(), 8);
    }

    #[test]
    fn is_empty() {
        let cf =
            CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert!(cf.is_empty());
        assert_eq!(cf.len(), 0);
    }

    #[test]
    fn insert() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert!(cf.insert(&13).unwrap());
        assert!(!cf.is_empty());
        assert_eq!(cf.len(), 1);
        assert!(cf.query(&13));
        assert!(!cf.query(&42));
    }

    #[test]
    fn double_insert() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert!(cf.insert(&13).unwrap());
        assert!(cf.insert(&13).unwrap());
        assert!(cf.query(&13));
    }

    #[test]
    fn delete() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        cf.insert(&13).unwrap();
        cf.insert(&42).unwrap();
        assert!(cf.query(&13));
        assert!(cf.query(&42));
        assert_eq!(cf.len(), 2);

        assert!(cf.delete(&13));
        assert!(!cf.query(&13));
        assert!(cf.query(&42));
        assert_eq!(cf.len(), 1);
    }

    #[test]
    fn clear() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);

        cf.insert(&1).unwrap();
        cf.clear();
        assert!(!cf.query(&1));
        assert!(cf.is_empty());
    }

    #[test]
    fn full() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 2, 8);

        for i in 0..4 {
            cf.insert(&i).unwrap();
        }
        assert_eq!(cf.len(), 4);
        for i in 0..4 {
            assert!(cf.query(&i));
        }

        assert!(cf.insert(&5).is_err());
        assert_eq!(cf.len(), 4);
        assert!(!cf.query(&5)); // rollback was executed
    }

    #[test]
    fn debug() {
        let cf =
            CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert_eq!(
            format!("{:?}", cf),
            "CuckooFilter { bucketsize: 2, n_buckets: 16 }"
        );
    }

    #[test]
    fn clone() {
        let mut cf1 = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        cf1.insert(&13).unwrap();
        assert!(cf1.query(&13));

        let cf2 = cf1.clone();
        cf1.insert(&42).unwrap();
        assert!(cf2.query(&13));
        assert!(!cf2.query(&42));
    }

    #[test]
    fn with_properties_4() {
        let cf = CuckooFilter::<u64, ChaChaRng>::with_properties_4(
            0.02,
            1000,
            ChaChaRng::from_seed([0; 32]),
        );
        assert_eq!(cf.bucketsize(), 4);
        assert_eq!(cf.n_buckets(), 2048);
        assert_eq!(cf.l_fingerprint(), 9);
    }

    #[test]
    fn with_properties_8() {
        let cf = CuckooFilter::<u64, ChaChaRng>::with_properties_8(
            0.02,
            1000,
            ChaChaRng::from_seed([0; 32]),
        );
        assert_eq!(cf.bucketsize(), 8);
        assert_eq!(cf.n_buckets(), 1024);
        assert_eq!(cf.l_fingerprint(), 10);
    }

    #[test]
    #[should_panic(expected = "expected_elements (0) must be at least 1")]
    fn with_properties_4_panics_expected_elements_0() {
        CuckooFilter::<u64, ChaChaRng>::with_properties_4(0.02, 0, ChaChaRng::from_seed([0; 32]));
    }

    #[test]
    #[should_panic(expected = "false_positive_rate (0) must be greater than 0 and smaller than 1")]
    fn with_properties_4_panics_false_positive_rate_0() {
        CuckooFilter::<u64, ChaChaRng>::with_properties_4(0., 1000, ChaChaRng::from_seed([0; 32]));
    }

    #[test]
    #[should_panic(expected = "false_positive_rate (1) must be greater than 0 and smaller than 1")]
    fn with_properties_4_panics_false_positive_rate_1() {
        CuckooFilter::<u64, ChaChaRng>::with_properties_4(1., 1000, ChaChaRng::from_seed([0; 32]));
    }

    #[test]
    fn union() {
        let mut cf1 =
            CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        let mut cf2 = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);

        cf1.insert(&13).unwrap();
        cf1.insert(&42).unwrap();

        cf2.insert(&130).unwrap();
        cf2.insert(&420).unwrap();

        cf1.union(&cf2).unwrap();

        assert!(cf1.query(&13));
        assert!(cf1.query(&42));
        assert!(cf1.query(&130));
        assert!(cf1.query(&420));

        assert!(!cf2.query(&13));
        assert!(!cf2.query(&42));
        assert!(cf2.query(&130));
        assert!(cf2.query(&420));
    }

    #[test]
    #[should_panic(expected = "bucketsize must be equal (left=2, right=3)")]
    fn union_panics_bucketsize() {
        let mut cf1 =
            CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        let cf2 = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 3, 16, 8);
        cf1.union(&cf2).unwrap();
    }

    #[test]
    #[should_panic(expected = "n_buckets must be equal (left=16, right=32)")]
    fn union_panics_n_buckets() {
        let mut cf1 =
            CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        let cf2 = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 32, 8);
        cf1.union(&cf2).unwrap();
    }

    #[test]
    #[should_panic(expected = "l_fingerprint must be equal (left=8, right=16)")]
    fn union_panics_l_fingerprint() {
        let mut cf1 =
            CuckooFilter::<u64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        let cf2 = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 16);
        cf1.union(&cf2).unwrap();
    }

    #[test]
    #[should_panic(expected = "buildhasher must be equal")]
    fn union_panics_buildhasher() {
        let mut cf1 = CuckooFilter::<u64, ChaChaRng, BuildHasherSeeded>::with_params_and_hash(
            ChaChaRng::from_seed([0; 32]),
            2,
            16,
            8,
            BuildHasherSeeded::new(0),
        );
        let cf2 = CuckooFilter::with_params_and_hash(
            ChaChaRng::from_seed([0; 32]),
            2,
            16,
            8,
            BuildHasherSeeded::new(1),
        );
        cf1.union(&cf2).unwrap();
    }

    #[test]
    fn union_full() {
        let mut cf1 =
            CuckooFilter::<i64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        let mut cf2 =
            CuckooFilter::<i64, ChaChaRng>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);

        // fill up cf1
        let mut obj = 0;
        loop {
            if cf1.insert(&obj).is_err() {
                break;
            }
            obj += 1;
        }
        assert!(cf1.query(&0));

        // add some payload to cf2
        let n_cf2 = 10;
        for i in 0..n_cf2 {
            cf2.insert(&-i).unwrap();
        }
        assert_eq!(cf2.len(), n_cf2 as usize);
        assert!(!cf2.query(&1));

        // union with failure, state must not be altered
        assert!(cf2.union(&cf1).is_err());
        assert_eq!(cf2.len(), n_cf2 as usize);
        assert!(!cf2.query(&1));
    }

    #[test]
    fn insert_unsized() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert!(cf.insert("test1").unwrap());
        assert!(!cf.is_empty());
        assert_eq!(cf.len(), 1);
        assert!(cf.query("test1"));
        assert!(!cf.query("test2"));
    }

    #[test]
    fn send() {
        let cf = CuckooFilter::<NotSend, _>::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert_send(&cf);
    }

    #[test]
    fn succinct_table_save_load() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert!(cf.insert(&10).unwrap());
        assert!(cf.insert(&51).unwrap());
        assert_eq!(cf.len(), 2);

        let loaded_cf = CuckooFilter::with_existing_filter(
            ChaChaRng::from_seed([0; 32]),
            2,
            16,
            8,
            cf.len(),
            cf.table_succinct_blocks(),
        );

        assert!(loaded_cf.query(&10));
        assert!(loaded_cf.query(&51));
        assert!(!loaded_cf.query(&33));
        assert_eq!(loaded_cf.len(), 2);
    }

    #[test]
    fn table_save_load() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert!(cf.insert(&10).unwrap());
        assert!(cf.insert(&51).unwrap());
        assert_eq!(cf.len(), 2);

        let mut loaded_cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        loaded_cf.load_table(cf.table(), cf.len());

        assert!(loaded_cf.query(&10));
        assert!(loaded_cf.query(&51));
        assert!(!loaded_cf.query(&33));
        assert_eq!(loaded_cf.len(), 2);
    }
}
