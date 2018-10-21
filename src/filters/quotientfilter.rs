//! QuotientFilter implementation.
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::mem::size_of;

use fixedbitset::FixedBitSet;
use succinct::{IntVec, IntVecMut, IntVector};

use filters::Filter;

/// Error that signals that the QuotientFilter is full.
#[derive(Debug)]
pub struct QuotientFilterFull;

/// TODO
/// Run: List of fingerprints with same quotient.
/// Cluster: set of runs
///
/// https://en.wikipedia.org/wiki/Quotient_filter
/// http://static.usenix.org/events/hotstorage11/tech/final_files/Bender.pdf
/// https://www.vldb.org/pvldb/vol5/p1627_michaelabender_vldb2012.pdf
#[derive(Debug)]
pub struct QuotientFilter<B = BuildHasherDefault<DefaultHasher>>
where
    B: BuildHasher + Clone + Eq,
{
    is_occupied: FixedBitSet,
    is_continuation: FixedBitSet,
    is_shifted: FixedBitSet,
    remainders: IntVector,
    bits_quotient: usize,
    buildhasher: B,
    n_elements: usize,
}

impl QuotientFilter {
    /// Create new quotient filter with:
    ///
    /// - `bits_quotient`: number of bits used for a quotient, aka `2^bits_quotient` slots will be
    ///   allocated
    /// - `bits_remainder`: number of bits used for the remainder, so every slot will require
    ///   `bits_remainder + 3` bits of storage
    ///
    /// and a default hasher.
    pub fn with_params(bits_quotient: usize, bits_remainder: usize) -> Self {
        let buildhasher = BuildHasherDefault::<DefaultHasher>::default();
        QuotientFilter::with_params_and_hash(bits_quotient, bits_remainder, buildhasher)
    }
}

impl<B> QuotientFilter<B>
where
    B: BuildHasher + Clone + Eq,
{
    /// Create new quotient filter with:
    ///
    /// - `bits_quotient`: number of bits used for a quotient, aka `2^bits_quotient` slots will be
    ///   allocated
    /// - `bits_remainder`: number of bits used for the remainder, so every slot will require
    ///   `bits_remainder + 3` bits of storage
    /// - `buildhasher`: hash implementation
    pub fn with_params_and_hash(
        bits_quotient: usize,
        bits_remainder: usize,
        buildhasher: B,
    ) -> Self {
        assert!(
            (bits_remainder > 0) && (bits_remainder <= size_of::<usize>() * 8),
            "bits_remainder ({}) must be greater than 0 and smaller or equal than {}",
            bits_remainder,
            size_of::<usize>() * 8,
        );
        assert!(
            bits_quotient > 0,
            "bits_quotient ({}) must be greater than 0",
            bits_quotient,
        );
        assert!(
            bits_remainder + bits_quotient <= 64,
            "bits_remainder ({}) + bits_quotient ({}) must be smaller or equal than 64",
            bits_remainder,
            bits_quotient,
        );

        let len = 1 << bits_quotient;
        Self {
            is_occupied: FixedBitSet::with_capacity(len),
            is_continuation: FixedBitSet::with_capacity(len),
            is_shifted: FixedBitSet::with_capacity(len),
            remainders: IntVector::with_fill(bits_remainder, len as u64, 0),
            bits_quotient,
            buildhasher,
            n_elements: 0,
        }
    }

    /// Number of bits used for addressing slots.
    pub fn bits_quotient(&self) -> usize {
        self.bits_quotient
    }

    /// Number of bits stored as fingeprint information.
    pub fn bits_remainder(&self) -> usize {
        self.remainders.element_bits()
    }

    fn calc_quotient_remainder<T>(&self, obj: &T) -> (usize, usize)
    where
        T: Hash,
    {
        let bits_remainder = self.bits_remainder();
        let mut hasher = self.buildhasher.build_hasher();
        obj.hash(&mut hasher);
        let fingerprint = hasher.finish();
        let bits_trash = 64 - bits_remainder - self.bits_quotient;
        let trash = if bits_trash > 0 {
            (fingerprint >> (64 - bits_trash)) << (64 - bits_trash)
        } else {
            0
        };
        let fingerprint_clean = fingerprint - trash;
        let quotient = fingerprint_clean >> bits_remainder;
        let remainder = fingerprint_clean - (quotient << bits_remainder);
        (quotient as usize, remainder as usize)
    }

    fn decr(&self, pos: &mut usize) {
        *pos = if *pos == 0 {
            self.is_occupied.len() - 1
        } else {
            *pos - 1
        };
    }

    fn incr(&self, pos: &mut usize) {
        *pos = if *pos == self.is_occupied.len() - 1 {
            0
        } else {
            *pos + 1
        }
    }

    fn scan(
        &self,
        quotient: usize,
        remainder: usize,
        on_insert: bool,
    ) -> (bool, usize, bool, usize) {
        let run_exists = self.is_occupied[quotient];
        if (!run_exists) && (!on_insert) {
            // fast-path for query, since we don't need to find the correct position for the
            // insertion process
            return (run_exists, quotient, run_exists, quotient);
        }

        // walk back to find the beginning of the cluster
        let mut b = quotient;
        while self.is_shifted[b] {
            self.decr(&mut b);
        }

        // walk forward to find the actual start of the run
        let mut s = b;
        while b != quotient {
            // invariant: `s` poins to first slot of bucket `b`

            // skip all elements in the current run
            loop {
                self.incr(&mut s);
                if !self.is_continuation[s] {
                    break;
                }
            }

            // find the next occupied bucket
            loop {
                self.incr(&mut b);
                if self.is_occupied[b] || ((b == quotient) && on_insert) {
                    break;
                }
            }
        }
        // `s` now points to the first remainder in bucket at `quotient`

        // search of remainder within the run
        let start_of_run = s;
        if run_exists {
            loop {
                let r = self.remainders.get(s as u64);
                if r == remainder {
                    return (run_exists, s, run_exists, start_of_run);
                }
                if r > remainder {
                    // remainders are sorted within run
                    break;
                }
                self.incr(&mut s);
                if !self.is_continuation[s] {
                    break;
                }
            }
        }
        (false, s, run_exists, start_of_run)
    }
}

impl<B> Filter for QuotientFilter<B>
where
    B: BuildHasher + Clone + Eq,
{
    type InsertErr = QuotientFilterFull;

    fn clear(&mut self) {
        self.is_occupied.clear();
        self.is_continuation.clear();
        self.is_shifted.clear();
        self.remainders =
            IntVector::with_fill(self.remainders.element_bits(), self.remainders.len(), 0);
        self.n_elements = 0;
    }

    fn insert<T>(&mut self, t: &T) -> Result<(), Self::InsertErr>
    where
        T: Hash,
    {
        let (quotient, remainder) = self.calc_quotient_remainder(t);
        let (present, mut position, run_exists, start_of_run) =
            self.scan(quotient, remainder, true);

        // early exit if the element is already present
        if present {
            return Ok(());
        }
        // we need to insert the element into the filter

        // error out if there is no space left
        if self.n_elements == self.is_occupied.len() {
            return Err(QuotientFilterFull);
        }

        // set up swap chain
        let mut current_is_continuation =
            self.is_continuation[position] || (run_exists && (position == start_of_run));
        let mut current_remainder = self.remainders.get(position as u64);
        let mut current_used = self.is_occupied[position] || self.is_shifted[position];

        // set current state
        self.remainders.set(position as u64, remainder);
        if position != start_of_run {
            // might be an append operation, ensure is_continuation and is_shifted are set
            self.is_continuation.set(position, true);
        }
        if position != quotient {
            // not at canonical slot
            self.is_shifted.set(position, true);
        }

        // run swap chain until nothing to do
        let start = position;
        while current_used {
            self.incr(&mut position);
            let next_is_continuation = self.is_continuation[position];
            let next_remainder = self.remainders.get(position as u64);
            let next_used = self.is_occupied[position] || self.is_shifted[position];

            self.is_shifted.set(position, true);
            self.is_continuation.set(position, current_is_continuation);
            self.remainders.set(position as u64, current_remainder);

            current_is_continuation = next_is_continuation;
            current_remainder = next_remainder;
            current_used = next_used;

            if position == start {
                panic!("infinite loop detected");
            }
        }

        // mark canonical slot as occupied
        self.is_occupied.set(quotient, true);

        // done
        self.n_elements += 1;
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.n_elements == 0
    }

    fn len(&self) -> usize {
        self.n_elements
    }

    fn query<T>(&self, obj: &T) -> bool
    where
        T: Hash,
    {
        let (quotient, remainder) = self.calc_quotient_remainder(obj);
        let (present, _position, _run_exists, _start_of_run) =
            self.scan(quotient, remainder, false);
        present
    }
}

#[cfg(test)]
mod tests {
    use super::QuotientFilter;
    use filters::Filter;

    #[test]
    #[should_panic(expected = "bits_quotient (0) must be greater than 0")]
    fn new_bits_quotient_0() {
        QuotientFilter::with_params(0, 16);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (0) must be greater than 0 and smaller or equal than 32"
    )]
    fn new_bits_remainder_0() {
        QuotientFilter::with_params(3, 0);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (0) must be greater than 0 and smaller or equal than 64"
    )]
    fn new_bits_remainder_0() {
        QuotientFilter::with_params(3, 0);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (33) must be greater than 0 and smaller or equal than 32"
    )]
    fn new_bits_remainder_too_large() {
        QuotientFilter::with_params(3, 33);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (65) must be greater than 0 and smaller or equal than 64"
    )]
    fn new_bits_remainder_too_large() {
        QuotientFilter::with_params(3, 65);
    }

    #[test]
    #[should_panic(
        expected = "bits_remainder (5) + bits_quotient (60) must be smaller or equal than 64"
    )]
    fn new_too_many_bits() {
        QuotientFilter::with_params(60, 5);
    }

    #[test]
    fn new() {
        let qf = QuotientFilter::with_params(3, 16);
        assert!(qf.is_empty());
        assert_eq!(qf.len(), 0);
        assert!(!qf.query(&13));
        assert_eq!(qf.bits_quotient(), 3);
        assert_eq!(qf.bits_remainder(), 16);
    }

    #[test]
    fn insert() {
        let mut qf = QuotientFilter::with_params(3, 16);
        qf.insert(&13).unwrap();
        assert!(!qf.is_empty());
        assert_eq!(qf.len(), 1);
        assert!(qf.query(&13));
        assert!(!qf.query(&42));
    }

    #[test]
    fn double_insert() {
        let mut qf = QuotientFilter::with_params(3, 16);
        qf.insert(&13).unwrap();
        qf.insert(&13).unwrap();
        assert!(!qf.is_empty());
        assert_eq!(qf.len(), 1);
        assert!(qf.query(&13));
        assert!(!qf.query(&42));
    }

    #[test]
    fn full() {
        let mut qf = QuotientFilter::with_params(3, 16);
        for i in 0..8 {
            qf.insert(&i).unwrap();
            for j in 0..i {
                assert!(qf.query(&j), "Cannot find {} after inserting {}", j, i);
            }
        }
        assert!(qf.insert(&1000).is_err());
    }

    #[test]
    fn clear() {
        let mut qf = QuotientFilter::with_params(3, 16);
        qf.insert(&13).unwrap();
        qf.clear();
        assert!(qf.is_empty());
        assert_eq!(qf.len(), 0);
        assert!(!qf.query(&13));
        assert_eq!(qf.bits_quotient(), 3);
        assert_eq!(qf.bits_remainder(), 16);
    }
}
