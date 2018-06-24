//! `CuckooFilter` implementation.
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};

use fixedbitset::FixedBitSet;
use rand::Rng;

use hash_utils::MyBuildHasherDefault;

const MAX_NUM_KICKS: usize = 500; // mentioned in paper

/// Error struct used to signal that a `CuckooFilter` is full, i.e. that a value cannot be inserted
/// because the implementation was unable to find a free bucket.
#[derive(Debug)]
pub struct CuckooFilterFull;

/// [`CuckooFilter`](https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf).
#[derive(Clone)]
pub struct CuckooFilter<R, B = MyBuildHasherDefault<DefaultHasher>>
where
    R: Rng,
    B: BuildHasher + Clone + Eq,
{
    table: FixedBitSet,
    n_elements: usize,
    buildhasher: B,
    bucketsize: usize,
    n_buckets: usize,
    l_fingerprint: usize,
    rng: R,
}

impl<R> CuckooFilter<R>
where
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
        let bh = MyBuildHasherDefault::<DefaultHasher>::default();
        Self::with_params_and_hash(rng, bucketsize, n_buckets, l_fingerprint, bh)
    }
}

impl<R, B> CuckooFilter<R, B>
where
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
            .expect("Table size too large")
            .checked_mul(l_fingerprint)
            .expect("Table size too large");

        Self {
            table: FixedBitSet::with_capacity(table_size),
            n_elements: 0,
            buildhasher: bh,
            bucketsize: bucketsize,
            n_buckets: n_buckets,
            l_fingerprint: l_fingerprint,
            rng: rng,
        }
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

    /// Check if CuckooFilter is empty, i.e. contains no elements.
    pub fn empty(&self) -> bool {
        self.n_elements == 0
    }

    /// Return number of elements in the filter.
    pub fn len(&self) -> usize {
        self.n_elements
    }

    /// Insert new element into filter.
    ///
    /// The method may return an error if it was unable to find a free bucket. This means the
    /// filter is full and you should not add any additional elements to it. When this happens,
    /// `len` was not increased and the filter content was not altered.
    ///
    /// Inserting the same element multiple times is supported, but keep in mind that after
    /// `n_buckets * 2` times, the filter will return `Err(CuckooFilterFull)`.
    pub fn insert<T>(&mut self, t: &T) -> Result<(), CuckooFilterFull>
    where
        T: Hash,
    {
        let (mut f, i1, i2) = self.start(t);

        if self.write_to_bucket(i1, f) {
            self.n_elements += 1;
            return Ok(());
        }
        if self.write_to_bucket(i2, f) {
            self.n_elements += 1;
            return Ok(());
        }

        // cannot write to obvious buckets => relocate
        let table_backup = self.table.clone(); // may be required for rollback
        let mut i = if self.rng.gen::<bool>() { i1 } else { i2 };

        for _ in 0..MAX_NUM_KICKS {
            let e = self.rng.gen_range::<usize>(0, self.bucketsize);
            let offset = i * self.bucketsize;
            let x = offset + e;

            // swap table[x] and f
            let tmp = self.table_read(x);
            self.table_write(x, f);
            f = tmp;

            i = i ^ self.hash(&f);
            if self.write_to_bucket(i, f) {
                self.n_elements += 1;
                return Ok(());
            }
        }

        // no space left => fail
        self.table = table_backup; // rollback transaction
        Err(CuckooFilterFull)
    }

    /// Check if given element was inserted into the filter.
    pub fn lookup<T>(&self, t: &T) -> bool
    where
        T: Hash,
    {
        let (f, i1, i2) = self.start(t);

        if self.has_in_bucket(i1, f) {
            return true;
        }
        if self.has_in_bucket(i2, f) {
            return true;
        }
        false
    }

    /// Remove element from the filter.
    ///
    /// Returns `true` if element was in the filter, `false` if it was not in which case the operation did not modify
    /// the filter.
    pub fn delete<T>(&mut self, t: &T) -> bool
    where
        T: Hash,
    {
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

    fn start<T>(&self, t: &T) -> (u64, usize, usize)
    where
        T: Hash,
    {
        let f = self.fingerprint(t);
        let i1 = self.hash(t);
        let i2 = i1 ^ self.hash(&f);
        (f, i1, i2)
    }

    fn fingerprint<T>(&self, t: &T) -> u64
    where
        T: Hash,
    {
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

    fn hash<T>(&self, t: &T) -> usize
    where
        T: Hash,
    {
        let mut hasher = self.buildhasher.build_hasher();
        hasher.write_usize(1); // IV
        t.hash(&mut hasher);
        (hasher.finish() & (self.n_buckets as u64 - 1)) as usize
    }

    fn write_to_bucket(&mut self, i: usize, f: u64) -> bool {
        let offset = i * self.bucketsize;
        for x in offset..(offset + self.bucketsize) {
            if self.table_read(x) == 0 {
                self.table_write(x, f);
                return true;
            }
        }
        false
    }

    fn has_in_bucket(&self, i: usize, f: u64) -> bool {
        let offset = i * self.bucketsize;
        for x in offset..(offset + self.bucketsize) {
            if self.table_read(x) == f {
                return true;
            }
        }
        false
    }

    fn remove_from_bucket(&mut self, i: usize, f: u64) -> bool {
        let offset = i * self.bucketsize;
        for x in offset..(offset + self.bucketsize) {
            if self.table_read(x) == f {
                self.table_write(x, 0);
                return true;
            }
        }
        false
    }

    fn table_read(&self, i: usize) -> u64 {
        let offset = i * self.l_fingerprint;
        let mut result = 0u64;
        for b in 0..self.l_fingerprint {
            result |= (self.table[offset + b] as u64) << (self.l_fingerprint - b - 1);
        }
        result
    }

    fn table_write(&mut self, i: usize, f: u64) {
        let offset = i * self.l_fingerprint;
        for b in 0..self.l_fingerprint {
            self.table
                .set(offset + b, (f >> (self.l_fingerprint - b - 1) & 1u64) != 0);
        }
    }
}

impl<R, B> fmt::Debug for CuckooFilter<R, B>
where
    R: Rng,
    B: BuildHasher + Clone + Eq,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    use rand::{ChaChaRng, SeedableRng};

    #[test]
    #[should_panic(expected = "bucketsize (0) must be greater or equal than 2")]
    fn new_panics_bucketsize_0() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 0, 16, 8);
    }

    #[test]
    #[should_panic(expected = "bucketsize (1) must be greater or equal than 2")]
    fn new_panics_bucketsize_1() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 1, 16, 8);
    }

    #[test]
    #[should_panic(expected = "n_buckets (0) must be a power of 2 and greater or equal than 2")]
    fn new_panics_n_buckets_0() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 0, 8);
    }

    #[test]
    #[should_panic(expected = "n_buckets (1) must be a power of 2 and greater or equal than 2")]
    fn new_panics_n_buckets_1() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 1, 8);
    }

    #[test]
    #[should_panic(expected = "n_buckets (5) must be a power of 2 and greater or equal than 2")]
    fn new_panics_n_buckets_5() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 5, 8);
    }

    #[test]
    #[should_panic(expected = "l_fingerprint (0) must be greater than 1 and less or equal than 64")]
    fn new_panics_l_fingerprint_0() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 0);
    }

    #[test]
    #[should_panic(expected = "l_fingerprint (1) must be greater than 1 and less or equal than 64")]
    fn new_panics_l_fingerprint_1() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 1);
    }

    #[test]
    #[should_panic(
        expected = "l_fingerprint (65) must be greater than 1 and less or equal than 64"
    )]
    fn new_panics_l_fingerprint_65() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 65);
    }

    #[test]
    #[should_panic(expected = "Table size too large")]
    fn new_panics_table_size_overflow_1() {
        CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), usize::max_value(), 2, 2);
    }

    #[test]
    #[should_panic(expected = "Table size too large")]
    fn new_panics_table_size_overflow_2() {
        CuckooFilter::with_params(
            ChaChaRng::from_seed([0; 32]),
            2,
            (((usize::max_value() as u128) + 1) / 2) as usize,
            2,
        );
    }

    #[test]
    #[should_panic(expected = "Table size too large")]
    fn new_panics_table_size_overflow_3() {
        CuckooFilter::with_params(
            ChaChaRng::from_seed([0; 32]),
            2,
            (((usize::max_value() as u128) + 1) / 8) as usize,
            64,
        );
    }

    #[test]
    fn getter() {
        let cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert_eq!(cf.bucketsize(), 2);
        assert_eq!(cf.n_buckets(), 16);
        assert_eq!(cf.l_fingerprint(), 8);
    }

    #[test]
    fn empty() {
        let cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert!(cf.empty());
        assert_eq!(cf.len(), 0);
    }

    #[test]
    fn insert() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        cf.insert(&13).unwrap();
        assert!(!cf.empty());
        assert_eq!(cf.len(), 1);
        assert!(cf.lookup(&13));
        assert!(!cf.lookup(&42));
    }

    #[test]
    fn delete() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        cf.insert(&13).unwrap();
        cf.insert(&42).unwrap();
        assert!(cf.lookup(&13));
        assert!(cf.lookup(&42));
        assert_eq!(cf.len(), 2);

        assert!(cf.delete(&13));
        assert!(!cf.lookup(&13));
        assert!(cf.lookup(&42));
        assert_eq!(cf.len(), 1);
    }

    #[test]
    fn full() {
        let mut cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 2, 8);

        for i in 0..4 {
            cf.insert(&i).unwrap();
        }
        assert_eq!(cf.len(), 4);
        for i in 0..4 {
            assert!(cf.lookup(&i));
        }

        assert!(cf.insert(&5).is_err());
        assert_eq!(cf.len(), 4);
        assert!(!cf.lookup(&5)); // rollback was executed
    }

    #[test]
    fn debug() {
        let cf = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        assert_eq!(
            format!("{:?}", cf),
            "CuckooFilter { bucketsize: 2, n_buckets: 16 }"
        );
    }

    #[test]
    fn clone() {
        let mut cf1 = CuckooFilter::with_params(ChaChaRng::from_seed([0; 32]), 2, 16, 8);
        cf1.insert(&13).unwrap();
        assert!(cf1.lookup(&13));

        let cf2 = cf1.clone();
        cf1.insert(&42).unwrap();
        assert!(cf2.lookup(&13));
        assert!(!cf2.lookup(&42));
    }
}
