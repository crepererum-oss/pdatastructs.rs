//! QuotientFilter implementation.
use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

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
    is_occupied: Vec<bool>,     // TODO: shrink
    is_continuation: Vec<bool>, // TODO: shrink
    is_shifted: Vec<bool>,      // TODO: shrink
    remainders: Vec<u64>,       // TODO: shrink
    bits_quotient: usize,
    bits_remainder: usize,
    buildhasher: B,
    n_elements: usize,
}

impl QuotientFilter {
    /// TODO
    pub fn with_params(bits_quotient: usize, bits_remainder: usize) -> Self {
        let buildhasher = BuildHasherDefault::<DefaultHasher>::default();
        QuotientFilter::with_params_and_hash(bits_quotient, bits_remainder, buildhasher)
    }
}

impl<B> QuotientFilter<B>
where
    B: BuildHasher + Clone + Eq,
{
    /// TODO
    pub fn with_params_and_hash(
        bits_quotient: usize,
        bits_remainder: usize,
        buildhasher: B,
    ) -> Self {
        assert!(
            bits_remainder > 0,
            "bits_remainder ({}) be larger than 0",
            bits_remainder,
        );
        assert!(
            bits_quotient > 0,
            "bits_quotient ({}) be larger than 0",
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
            is_occupied: vec![false; len],
            is_continuation: vec![false; len],
            is_shifted: vec![false; len],
            remainders: vec![0; len],
            bits_quotient,
            bits_remainder,
            buildhasher,
            n_elements: 0,
        }
    }

    fn calc_quotient_remainder<T>(&self, obj: &T) -> (usize, u64)
    where
        T: Hash,
    {
        let mut hasher = self.buildhasher.build_hasher();
        obj.hash(&mut hasher);
        let fingerprint = hasher.finish();
        let bits_trash = 64 - self.bits_remainder - self.bits_quotient;
        let trash = if bits_trash > 0 {
            (fingerprint >> (64 - bits_trash)) << (64 - bits_trash)
        } else {
            0
        };
        let fingerprint_clean = fingerprint - trash;
        let quotient = fingerprint_clean >> self.bits_remainder;
        let remainder = fingerprint_clean - (quotient << self.bits_remainder);
        (quotient as usize, remainder)
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

    fn scan(&self, quotient: usize, remainder: u64, on_insert: bool) -> (bool, usize, bool, usize) {
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
                if self.remainders[s] == remainder {
                    return (run_exists, s, run_exists, start_of_run);
                }
                if self.remainders[s] > remainder {
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
        unimplemented!()
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
        let mut current_remainder = self.remainders[position];
        let mut current_used = self.is_occupied[position] || self.is_shifted[position];

        // set current state
        self.remainders[position] = remainder;
        if position != start_of_run {
            // might be an append operation, ensure is_continuation and is_shifted are set
            self.is_continuation[position] = true;
        }
        if position != quotient {
            // not at canonical slot
            self.is_shifted[position] = true;
        }

        // run swap chain until nothing to do
        let start = position;
        while current_used {
            self.incr(&mut position);
            let next_is_continuation = self.is_continuation[position];
            let next_remainder = self.remainders[position];
            let next_used = self.is_occupied[position] || self.is_shifted[position];

            self.is_shifted[position] = true;
            self.is_continuation[position] = current_is_continuation;
            self.remainders[position] = current_remainder;

            current_is_continuation = next_is_continuation;
            current_remainder = next_remainder;
            current_used = next_used;

            if position == start {
                panic!("infinite loop detected");
            }
        }

        // mark canonical slot as occupied
        self.is_occupied[quotient] = true;

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
    fn new() {
        let qf = QuotientFilter::with_params(3, 16);
        assert!(qf.is_empty());
        assert_eq!(qf.len(), 0);
        assert!(!qf.query(&13));
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
}
