//! QuotientFilter implementation.
use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::size_of;

use fixedbitset::FixedBitSet;
use succinct::{IntVec, IntVecMut, IntVector};

use crate::filters::Filter;
use crate::helpers::all_zero_intvector;

/// Error that signals that the QuotientFilter is full.
#[derive(Debug)]
pub struct QuotientFilterFull;

/// Internal results for scanning the quotientfilter.
struct ScanResult {
    /// Indicates if the requested element is already present in the filter.
    present: bool,

    /// Position where the search ended.
    position: usize,

    /// Start position of run where the search ended.
    start_of_run: Option<usize>,
}

impl ScanResult {
    fn has_run(&self) -> bool {
        self.start_of_run.is_some()
    }

    fn at_start_of_run(&self) -> bool {
        match self.start_of_run {
            Some(start) => start == self.position,
            None => false,
        }
    }
}

/// A QuotientFilter is a set-like data structure, that keeps track of elements it has seen without
/// the need to store them. Looking up values has a certain false positive rate, but a false
/// negative rate of 0%.
///
/// # Examples
/// ```
/// use pdatastructs::filters::Filter;
/// use pdatastructs::filters::quotientfilter::QuotientFilter;
///
/// // set up filter
/// let bits_quotient = 16;
/// let bits_remainder = 5;
/// let mut filter = QuotientFilter::with_params(bits_quotient, bits_remainder);
///
/// // add some data
/// filter.insert(&"my super long string").unwrap();
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
///
/// ## Setup
/// There are `2^bits_quotient` slots, initial empty. For every slot, we store `bits_remainder` as
/// fingerprint information, a `is_continuation` bit, a `is_occupied` bit and a `is_shifted` bit.
/// All bits are initially set to false.
///
/// ```text
/// bits_quotient  = 3
/// bits_remainder = 4
///
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// |  position       ||   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | is_occupied     ||     |     |     |     |     |     |     |     |
/// | is_continuation ||     |     |     |     |     |     |     |     |
/// | is_shifted      ||     |     |     |     |     |     |     |     |
/// | remainder       || 0x0 | 0x0 | 0x0 | 0x0 | 0x0 | 0x0 | 0x0 | 0x0 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// ```
///
/// ## Insertion
/// On insertion, elements are hashed to 64 bits. From these, `bits_quotient` are used as a
/// quotient and `bits_remainder` are used as remainder, the remaining bits are dropped.
///
/// The quotient represents the canonical position in which the remainder should be inserted. If is
/// is free, we use that position, set the `is_occupied` bit and are done.
///
/// ```text
/// x           = "foo"
/// h(x)        = 0x0123456789abcda5
/// h(x) & 0x7f = 0x25
/// remainder   = 0x5
/// quotient    = 0x2
///
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | position        ||   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | is_occupied     ||     |     |   X |     |     |     |     |     |
/// | is_continuation ||     |     |     |     |     |     |     |     |
/// | is_shifted      ||     |     |     |     |     |     |     |     |
/// | remainder       || 0x0 | 0x0 | 0x2 | 0x0 | 0x0 | 0x0 | 0x0 | 0x0 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// ```
///
/// If not, linear probing is applied. If an element with the same quotient is already in the
/// filter, the so called "run" of it will be extended. For extensions, the `is_continuation` bit
/// is set as well as the `is_shifted` bit because the stored remainder is not in its canonical
/// position:
///
/// ```text
/// x           = "bar"
/// h(x)        = 0xad8caa00248af32e
/// h(x) & 0x7f = 0x2e
/// remainder   = 0xe
/// quotient    = 0x2
///
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | position        ||   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | is_occupied     ||     |     |   X |     |     |     |     |     |
/// | is_continuation ||     |     |     |   X |     |     |     |     |
/// | is_shifted      ||     |     |     |   X |     |     |     |     |
/// | remainder       || 0x0 | 0x0 | 0x2 | 0xe | 0x0 | 0x0 | 0x0 | 0x0 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | run             ||            [=========]                        |
/// +-----------------++-----------------------------------------------|
/// ```
///
/// While doing so, the order of remainders within the run is preserved:
///
/// ```text
/// x           = "elephant"
/// h(x)        = 0x34235511eeadbc26
/// h(x) & 0x7f = 0x26
/// remainder   = 0x6
/// quotient    = 0x2
///
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | position        ||   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | is_occupied     ||     |     |   X |     |     |     |     |     |
/// | is_continuation ||     |     |     |   X |   X |     |     |     |
/// | is_shifted      ||     |     |     |   X |   X |     |     |     |
/// | remainder       || 0x0 | 0x0 | 0x2 | 0x6 | 0xe | 0x0 | 0x0 | 0x0 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | run             ||            [===============]                  |
/// +-----------------++-----------------------------------------------|
/// ```
///
/// If a new quotient is inserted but the corresponding run cannot start at the canonical position,
/// the entire run will be shifted. A sequence of runs is also called "cluster". Even though the
/// run is shifted, the original position will still be marked as occupied:
///
/// ```text
/// x           = "banana"
/// h(x)        = 0xdfdfdfdfdfdfdf31
/// h(x) & 0x7f = 0x31
/// remainder   = 0x1
/// quotient    = 0x3
///
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | position        ||   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | is_occupied     ||     |     |   X |   X |     |     |     |     |
/// | is_continuation ||     |     |     |   X |   X |     |     |     |
/// | is_shifted      ||     |     |     |   X |   X |   X |     |     |
/// | remainder       || 0x0 | 0x0 | 0x2 | 0x6 | 0xe | 0x1 | 0x0 | 0x0 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | run             ||            [===============] [===]            |
/// | cluster         ||            [=====================]            |
/// +-----------------++-----------------------------------------------|
/// ```
///
/// Remainders may duplicate over multiple runs:
///
/// ```text
/// x           = "apple"
/// h(x)        = 0x0000000000000072
/// h(x) & 0x7f = 0x72
/// remainder   = 0x2
/// quotient    = 0x7
///
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | position        ||   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | is_occupied     ||     |     |   X |   X |     |     |     |   X |
/// | is_continuation ||     |     |     |   X |   X |     |     |     |
/// | is_shifted      ||     |     |     |   X |   X |   X |     |     |
/// | remainder       || 0x0 | 0x0 | 0x2 | 0x6 | 0xe | 0x1 | 0x0 | 0x2 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | run             ||            [===============] [===]       [===]|
/// | cluster         ||            [=====================]       [===]|
/// +-----------------++-----------------------------------------------|
/// ```
///
/// The entire array works like a ring-buffer and operations can over- and underflow:
///
/// ```text
/// x           = "last"
/// h(x)        = 0x11355343431323f3
/// h(x) & 0x7f = 0x73
/// remainder   = 0x3
/// quotient    = 0x7
///
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | position        ||   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | is_occupied     ||     |     |   X |   X |     |     |     |   X |
/// | is_continuation ||   X |     |     |   X |   X |     |     |     |
/// | is_shifted      ||   X |     |     |   X |   X |   X |     |     |
/// | remainder       || 0x3 | 0x0 | 0x2 | 0x6 | 0xe | 0x1 | 0x0 | 0x2 |
/// +-----------------++-----+-----+-----+-----+-----+-----+-----+-----+
/// | run             ||====]       [===============] [===]       [====|
/// | cluster         ||====]       [=====================]       [====|
/// +-----------------++-----------------------------------------------|
/// ```
///
/// ## Lookup
/// The lookup basically follows the insertion procedure.
///
///
/// # See Also
/// - `std::collections::HashSet`: has a false positive rate of 0%, but also needs to store all
///   elements
///
/// # References
/// - ["Don’t Thrash: How to Cache your Hash on Flash" (short version), Michael A. Bender and others, 2012](http://static.usenix.org/events/hotstorage11/tech/final_files/Bender.pdf)
/// - ["Don’t Thrash: How to Cache your Hash on Flash" (long version), Michael A. Bender and others, 2012](https://www.vldb.org/pvldb/vol5/p1627_michaelabender_vldb2012.pdf)
/// - [Wikipedia: Quotient Filter](https://en.wikipedia.org/wiki/Quotient_filter)
#[derive(Clone, Debug)]
pub struct QuotientFilter<T, B = BuildHasherDefault<DefaultHasher>>
where
    T: Hash,
    B: BuildHasher + Clone + Eq,
{
    is_occupied: FixedBitSet,
    is_continuation: FixedBitSet,
    is_shifted: FixedBitSet,
    remainders: IntVector,
    bits_quotient: usize,
    buildhasher: B,
    n_elements: usize,
    phantom: PhantomData<T>,
}

impl<T> QuotientFilter<T>
where
    T: Hash,
{
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

impl<T, B> QuotientFilter<T, B>
where
    T: Hash,
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
            remainders: all_zero_intvector(bits_remainder, len),
            bits_quotient,
            buildhasher,
            n_elements: 0,
            phantom: PhantomData,
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

    fn calc_quotient_remainder(&self, obj: &T) -> (usize, usize) {
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

    fn scan(&self, quotient: usize, remainder: usize, on_insert: bool) -> ScanResult {
        let run_exists = self.is_occupied[quotient];
        if (!run_exists) && (!on_insert) {
            // fast-path for query, since we don't need to find the correct position for the
            // insertion process
            return ScanResult {
                present: false,
                position: quotient,
                start_of_run: None,
            };
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
        if run_exists {
            let start_of_run = s;
            loop {
                let r = self.remainders.get(s as u64);
                if r == remainder {
                    return ScanResult {
                        present: true,
                        position: s,
                        start_of_run: Some(start_of_run),
                    };
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
            ScanResult {
                present: false,
                position: s,
                start_of_run: Some(start_of_run),
            }
        } else {
            ScanResult {
                present: false,
                position: s,
                start_of_run: None,
            }
        }
    }

    fn insert_internal(
        &mut self,
        quotient: usize,
        remainder: usize,
    ) -> Result<bool, QuotientFilterFull> {
        let scan_result = self.scan(quotient, remainder, true);

        // early exit if the element is already present
        if scan_result.present {
            return Ok(false);
        }
        // we need to insert the element into the filter

        // error out if there is no space left
        if self.n_elements == self.is_occupied.len() {
            return Err(QuotientFilterFull);
        }

        // set up swap chain
        let mut current_is_continuation =
            self.is_continuation[scan_result.position] || scan_result.at_start_of_run();
        let mut current_remainder = self.remainders.get(scan_result.position as u64);
        let mut current_used =
            self.is_occupied[scan_result.position] || self.is_shifted[scan_result.position];

        // set current state
        self.remainders.set(scan_result.position as u64, remainder);
        // if scan_result.position != scan_result.start_of_run.unwrap_or(scan_result.position) {
        if scan_result.has_run() && (!scan_result.at_start_of_run()) {
            // might be an append operation, ensure is_continuation and is_shifted are set
            self.is_continuation.set(scan_result.position, true);
        }
        if scan_result.position != quotient {
            // not at canonical slot
            self.is_shifted.set(scan_result.position, true);
        }

        // run swap chain until nothing to do
        let start = scan_result.position;
        let mut position = scan_result.position;
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
        Ok(true)
    }
}

impl<T, B> Filter<T> for QuotientFilter<T, B>
where
    T: Hash,
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

    fn insert(&mut self, obj: &T) -> Result<bool, Self::InsertErr> {
        let (quotient, remainder) = self.calc_quotient_remainder(obj);
        self.insert_internal(quotient, remainder)
    }

    fn union(&mut self, other: &Self) -> Result<(), Self::InsertErr> {
        assert_eq!(
            self.bits_quotient, other.bits_quotient,
            "bits_quotient must be equal (left={}, right={})",
            self.bits_quotient, other.bits_quotient
        );
        assert_eq!(
            self.bits_remainder(),
            other.bits_remainder(),
            "bits_remainder must be equal (left={}, right={})",
            self.bits_remainder(),
            other.bits_remainder()
        );
        assert!(
            self.buildhasher == other.buildhasher,
            "buildhasher must be equal",
        );

        // create backup of the entire state
        let is_occupied_backup = self.is_occupied.clone();
        let is_continuation_backup = self.is_continuation.clone();
        let is_shifted_backup = self.is_shifted.clone();
        let remainders_backup = self.remainders.clone();
        let n_elements_backup = self.n_elements;

        for i in 0..other.is_occupied.len() {
            if other.is_occupied[i] && !other.is_shifted[i] {
                // found cluster start
                let mut quotient = i;
                match self.insert_internal(quotient, other.remainders.get(i as u64)) {
                    Err(err) => {
                        self.is_occupied = is_occupied_backup;
                        self.is_continuation = is_continuation_backup;
                        self.is_shifted = is_shifted_backup;
                        self.remainders = remainders_backup;
                        self.n_elements = n_elements_backup;
                        return Err(err);
                    }
                    Ok(_) => {}
                }

                let mut next_quotients = VecDeque::new();

                let mut j = i;
                self.incr(&mut j);
                while (j != i) && other.is_shifted[j] {
                    if other.is_occupied[j] {
                        // this cluster contains another run, so remember the quotient
                        next_quotients.push_back(j);
                    }
                    if !other.is_continuation[j] {
                        // this is the start of another run, get the quotient
                        quotient = next_quotients.pop_front().unwrap();
                    }
                    match self.insert_internal(quotient, other.remainders.get(j as u64)) {
                        Err(err) => {
                            self.is_occupied = is_occupied_backup;
                            self.is_continuation = is_continuation_backup;
                            self.is_shifted = is_shifted_backup;
                            self.remainders = remainders_backup;
                            self.n_elements = n_elements_backup;
                            return Err(err);
                        }
                        Ok(_) => {}
                    }

                    self.incr(&mut j)
                }
            }
        }

        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.n_elements == 0
    }

    fn len(&self) -> usize {
        self.n_elements
    }

    fn query(&self, obj: &T) -> bool {
        let (quotient, remainder) = self.calc_quotient_remainder(obj);
        self.scan(quotient, remainder, false).present
    }
}

#[cfg(test)]
mod tests {
    use super::QuotientFilter;
    use crate::filters::Filter;
    use crate::hash_utils::BuildHasherSeeded;

    #[test]
    #[should_panic(expected = "bits_quotient (0) must be greater than 0")]
    fn new_bits_quotient_0() {
        QuotientFilter::<u64>::with_params(0, 16);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (0) must be greater than 0 and smaller or equal than 32"
    )]
    fn new_bits_remainder_0() {
        QuotientFilter::<u64>::with_params(3, 0);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (0) must be greater than 0 and smaller or equal than 64"
    )]
    fn new_bits_remainder_0() {
        QuotientFilter::<u64>::with_params(3, 0);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (33) must be greater than 0 and smaller or equal than 32"
    )]
    fn new_bits_remainder_too_large() {
        QuotientFilter::<u64>::with_params(3, 33);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(
        expected = "bits_remainder (65) must be greater than 0 and smaller or equal than 64"
    )]
    fn new_bits_remainder_too_large() {
        QuotientFilter::<u64>::with_params(3, 65);
    }

    #[test]
    #[should_panic(
        expected = "bits_remainder (5) + bits_quotient (60) must be smaller or equal than 64"
    )]
    fn new_too_many_bits() {
        QuotientFilter::<u64>::with_params(60, 5);
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
        assert!(qf.insert(&13).unwrap());
        assert!(!qf.is_empty());
        assert_eq!(qf.len(), 1);
        assert!(qf.query(&13));
        assert!(!qf.query(&42));
    }

    #[test]
    fn double_insert() {
        let mut qf = QuotientFilter::with_params(3, 16);
        assert!(qf.insert(&13).unwrap());
        assert!(!qf.insert(&13).unwrap());
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

    #[test]
    fn clone() {
        let mut qf1 = QuotientFilter::with_params(3, 16);
        qf1.insert(&13).unwrap();

        let mut qf2 = qf1.clone();
        qf2.insert(&42).unwrap();

        assert_eq!(qf1.len(), 1);
        assert!(qf1.query(&13));
        assert!(!qf1.query(&42));

        assert_eq!(qf2.len(), 2);
        assert!(qf2.query(&13));
        assert!(qf2.query(&42));
    }

    #[test]
    fn union() {
        let mut qf1 = QuotientFilter::with_params(3, 16);
        let mut qf2 = QuotientFilter::with_params(3, 16);

        qf1.insert(&13).unwrap();
        qf1.insert(&42).unwrap();

        qf2.insert(&130).unwrap();
        qf2.insert(&420).unwrap();

        qf1.union(&qf2).unwrap();

        assert!(qf1.query(&13));
        assert!(qf1.query(&42));
        assert!(qf1.query(&130));
        assert!(qf1.query(&420));

        assert!(!qf2.query(&13));
        assert!(!qf2.query(&42));
        assert!(qf2.query(&130));
        assert!(qf2.query(&420));
    }

    #[test]
    #[should_panic(expected = "bits_quotient must be equal (left=3, right=4)")]
    fn union_panics_bits_quotient() {
        let mut qf1 = QuotientFilter::<i32>::with_params(3, 16);
        let qf2 = QuotientFilter::with_params(4, 16);
        qf1.union(&qf2).unwrap();
    }

    #[test]
    #[should_panic(expected = "bits_remainder must be equal (left=16, right=32)")]
    fn union_panics_bits_remainder() {
        let mut qf1 = QuotientFilter::<i32>::with_params(3, 16);
        let qf2 = QuotientFilter::with_params(3, 32);
        qf1.union(&qf2).unwrap();
    }

    #[test]
    #[should_panic(expected = "buildhasher must be equal")]
    fn union_panics_buildhasher() {
        let mut qf1 = QuotientFilter::<i32, BuildHasherSeeded>::with_params_and_hash(
            3,
            16,
            BuildHasherSeeded::new(0),
        );
        let qf2 = QuotientFilter::with_params_and_hash(3, 16, BuildHasherSeeded::new(1));
        qf1.union(&qf2).unwrap();
    }

    #[test]
    fn union_full() {
        let mut qf1 = QuotientFilter::with_params(3, 16);
        let mut qf2 = QuotientFilter::with_params(3, 16);

        // fill up cf1
        let mut obj = 0;
        loop {
            if qf1.insert(&obj).is_err() {
                break;
            }
            obj += 1;
        }
        assert!(qf1.query(&0));

        // add some payload to cf2
        let n_qf2 = 3;
        for i in 0..n_qf2 {
            qf2.insert(&-i).unwrap();
        }
        assert_eq!(qf2.len(), n_qf2 as usize);
        assert!(!qf2.query(&1));

        // union with failure, state must not be altered
        assert!(qf2.union(&qf1).is_err());
        assert_eq!(qf2.len(), n_qf2 as usize);
        assert!(!qf2.query(&1));
    }
}
