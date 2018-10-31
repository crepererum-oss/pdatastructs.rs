//! HyperLogLog implementation.
use bytecount;
use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;

use hyperloglog_data::{
    BIAS_DATA_OFFSET, BIAS_DATA_VEC, RAW_ESTIMATE_DATA_OFFSET, RAW_ESTIMATE_DATA_VEC,
    THRESHOLD_DATA_OFFSET, THRESHOLD_DATA_VEC,
};

/// A HyperLogLog is a data structure to count unique elements on a data stream.
///
/// # Examples
/// ```
/// use pdatastructs::hyperloglog::HyperLogLog;
///
/// // set up filter
/// let address_bits = 4;  // so we store 2^4 = 16 registers in total
/// let mut hll = HyperLogLog::new(address_bits);
///
/// // add some data
/// hll.add(&"my super long string");
/// hll.add(&"another super long string");
/// hll.add(&"my super long string");  // again
///
/// // later
/// assert_eq!(hll.count(), 2);
/// ```
///
/// # Applications
/// - an approximative `COUNT(DISTINCT x)` in SQL
/// - count distinct elements in a data stream
///
/// # How It Works
/// The HyperLogLog consists of `2^b` 8bit counters. Each counter is initialized to 0.
///
/// During insertion, a hash `h(x)` is calculated. The first `b` bits of the hash function are used
/// to address a register, the other bits are used to create a number `p` which essentially counts
/// the number of leading 0-bits (or in other words: the leftmost 1-bit). The addressed register is
/// then updated to the maximum of its current value and `p`.
///
/// The calculation of the count is based on `1 / Sum_0^{2^b} (2^-register_i)` with a bunch of
/// factors a corrections applied (see paper or source code).
///
/// # Implementation
/// - The registers always allocate 8 bits and are not compressed.
/// - No sparse representation is used at any point.
/// - A 64 bit hash function is used (like in HyperLogLog++ paper) instead of the 32 bit hash
///   function (like in the original HyperLogLog paper).
/// - Bias correction is applied and the data is currently just taken from the HyperLogLog++ paper
///   appendix.
///
/// # See Also
/// - `std::collections::HashSet`: can be used to get the exact count but requires you to store
///   each and every element
/// - `pdatastructs::bloomfilter::BloomFilter` and `pdatastructs::cuckoofilter::CuckooFilter`: when
///   you also want to check if a single element is in the observed set
///
/// # References
/// - ["HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm", Philippe
///   Flajolet, Éric Fusy, Olivier Gandouet, Frédéric Meunier, 2007](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.142.9475)
/// - ["HyperLogLog in Practice: Algorithmic Engineering of a State of The Art Cardinality Estimation Algorithm", Stefan
///   Heule, Marc Nunkesser, Alexander Hall, 2013](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40671.pdf)
/// - ["Appendix to HyperLogLog in Practice: Algorithmic Engineering of a State of the Art
///   Cardinality Estimation Algorithm", Stefan Heule, Marc Nunkesser, Alexander Hall, 2016](https://goo.gl/iU8Ig)
/// - [Wikipedia: HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog)
#[derive(Clone)]
pub struct HyperLogLog<T, B = BuildHasherDefault<DefaultHasher>>
where
    T: Hash,
    B: BuildHasher + Clone + Eq,
{
    registers: Vec<u8>,
    b: usize,
    buildhasher: B,
    phantom: PhantomData<T>,
}

impl<T> HyperLogLog<T>
where
    T: Hash,
{
    /// Creates a new, empty HyperLogLog.
    ///
    /// - `b` number of bits used for register selection, number of registers within the
    ///   HyperLogLog will be `2^b`. `b` must be in `[4, 16]`
    ///
    /// Panics when `b` is out of bounds.
    pub fn new(b: usize) -> Self {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        Self::with_hash(b, bh)
    }
}

impl<T, B> HyperLogLog<T, B>
where
    T: Hash,
    B: BuildHasher + Clone + Eq,
{
    /// Same as `new` but with a specific `BuildHasher`.
    pub fn with_hash(b: usize, buildhasher: B) -> Self {
        assert!(
            (b >= 4) & (b <= 16),
            "b ({}) must be larger or equal than 4 and smaller or equal than 16",
            b
        );

        let m = (1 as usize) << b;
        let registers = vec![0; m];
        Self {
            registers,
            b,
            buildhasher,
            phantom: PhantomData,
        }
    }

    /// Get number of bits used for register selection.
    pub fn b(&self) -> usize {
        self.b
    }

    /// Get number of registers.
    pub fn m(&self) -> usize {
        self.registers.len()
    }

    /// Get `BuildHasher`.
    pub fn buildhasher(&self) -> &B {
        &self.buildhasher
    }

    /// Get relative error for this HyperLogLog configuration.
    pub fn relative_error(&self) -> f64 {
        (3f64 * 2f64.ln() - 1f64).sqrt() / (self.m() as f64).sqrt()
    }

    /// Adds an element to the HyperLogLog.
    pub fn add(&mut self, obj: &T) {
        let mut hasher = self.buildhasher.build_hasher();
        obj.hash(&mut hasher);
        let h: u64 = hasher.finish();

        // split h into:
        //  - w = 64 - b upper bits
        //  - j = b lower bits
        let w = h >> self.b;
        let j = h - (w << self.b); // no 1 as in the paper since register indices are 0-based

        // p = leftmost bit (1-based count)
        let p = w.leading_zeros() + 1 - (self.b as u32);

        let m_old = self.registers[j as usize];
        self.registers[j as usize] = cmp::max(m_old, p as u8);
    }

    fn am(&self) -> f64 {
        let m = self.registers.len();

        if m >= 128 {
            0.7213 / (1. + 1.079 / (m as f64))
        } else if m >= 64 {
            0.709
        } else if m >= 32 {
            0.697
        } else {
            0.673
        }
    }

    fn estimate_bias(&self, e: f64) -> f64 {
        // binary search first nearest neighbor
        let lookup_array = RAW_ESTIMATE_DATA_VEC[self.b - RAW_ESTIMATE_DATA_OFFSET];
        let mut idx_left = match lookup_array.binary_search_by(|v| v.partial_cmp(&e).unwrap()) {
            Ok(i) => Some(i),  // exact match
            Err(i) => Some(i), // no match, i points to left neighbor
        };

        let mut idx_right = match idx_left {
            Some(i) => if i < lookup_array.len() - 1 {
                Some(i + 1)
            } else {
                None
            },
            _ => None,
        };

        // collect k nearest neighbors
        let k = 6;
        assert!(lookup_array.len() >= k);
        let mut neighbors = vec![];
        for _ in 0..k {
            let (right_instead_left, idx) = match (idx_left, idx_right) {
                (Some(i_left), Some(i_right)) => {
                    // 2 candidates, find better one
                    let delta_left = (lookup_array[i_left] - e).abs();
                    let delta_right = (lookup_array[i_right] - e).abs();
                    if delta_right < delta_left {
                        (true, i_right)
                    } else {
                        (false, i_left)
                    }
                }
                (Some(i_left), None) => {
                    // just left one is there, use it
                    (false, i_left)
                }
                (None, Some(i_right)) => {
                    // just right one is there, use it
                    (true, i_right)
                }
                _ => panic!("neighborhood search failed, this is bug!"),
            };
            neighbors.push(idx);
            if right_instead_left {
                idx_right = if idx < lookup_array.len() - 1 {
                    Some(idx + 1)
                } else {
                    None
                };
            } else {
                idx_left = if idx > 0 { Some(idx - 1) } else { None };
            }
        }

        // calculate mean of neighbors
        let bias_data = BIAS_DATA_VEC[self.b - BIAS_DATA_OFFSET];
        neighbors.iter().map(|&i| bias_data[i]).sum::<f64>() / (k as f64)
    }

    fn linear_counting(&self, v: usize) -> f64 {
        let m = self.registers.len() as f64;

        m * (m / (v as f64)).ln()
    }

    fn threshold(&self) -> usize {
        THRESHOLD_DATA_VEC[self.b - THRESHOLD_DATA_OFFSET]
    }

    /// Guess the number of unique elements seen by the HyperLogLog.
    pub fn count(&self) -> usize {
        let m = self.registers.len() as f64;

        let z = 1f64 / self
            .registers
            .iter()
            .map(|&x| 2f64.powi(-(i32::from(x))))
            .sum::<f64>();

        let e = self.am() * m * m * z;

        let e_star = if e <= (5. * m) {
            e - self.estimate_bias(e)
        } else {
            e
        };

        let v = bytecount::count(&self.registers, 0);
        let h = if v != 0 {
            self.linear_counting(v)
        } else {
            e_star
        };

        if h <= (self.threshold() as f64) {
            h as usize
        } else {
            e_star as usize
        }
    }

    /// Merge w/ another HyperLogLog.
    ///
    /// This HyperLogLog will then have the same state as if all elements seen by `other` where
    /// directly added to `self`.
    ///
    /// Panics when `b` or `buildhasher` parameter of `self` and `other` do not match.
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(
            self.b, other.b,
            "b must be equal (left={}, right={})",
            self.b, other.b
        );
        assert!(
            self.buildhasher == other.buildhasher,
            "buildhasher must be equal"
        );

        self.registers = self
            .registers
            .iter()
            .zip(other.registers.iter())
            .map(|x| cmp::max(x.0, x.1))
            .cloned()
            .collect();
    }

    /// Empties the HyperLogLog.
    pub fn clear(&mut self) {
        self.registers = vec![0; self.registers.len()];
    }

    /// Checks whether the HyperLogLog has never seen an element.
    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&x| x == 0)
    }
}

impl<T> fmt::Debug for HyperLogLog<T>
where
    T: Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HyperLogLog {{ b: {} }}", self.b)
    }
}

impl<T> Extend<T> for HyperLogLog<T>
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
    use super::HyperLogLog;
    use hash_utils::BuildHasherSeeded;

    #[test]
    #[should_panic(expected = "b (3) must be larger or equal than 4 and smaller or equal than 16")]
    fn new_panics_b3() {
        HyperLogLog::<u64>::new(3);
    }

    #[test]
    #[should_panic(expected = "b (17) must be larger or equal than 4 and smaller or equal than 16")]
    fn new_panics_b17() {
        HyperLogLog::<u64>::new(17);
    }

    #[test]
    fn getter() {
        let hll = HyperLogLog::<u64>::new(8);
        assert_eq!(hll.b(), 8);
        assert_eq!(hll.m(), 1 << 8);
        hll.buildhasher();
    }

    #[test]
    fn relative_error() {
        let hll = HyperLogLog::<u64>::new(4);
        assert!((hll.relative_error() - 0.2597).abs() < 0.001);
    }

    #[test]
    fn empty() {
        let hll = HyperLogLog::<u64>::new(8);
        assert_eq!(hll.count(), 0);
        assert!(hll.is_empty());
    }

    #[test]
    fn add_b4_n1k() {
        let mut hll = HyperLogLog::new(4);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 571);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b8_n1k() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 964);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b12_n1k() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 984);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b16_n1k() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 998);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b8_n10k() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..10000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 10196);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b12_n10k() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..10000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 10050);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b16_n10k() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..10000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 10055);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b16_n100k() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..100000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 100656);
        assert!(!hll.is_empty());
    }

    #[test]
    fn add_b16_n1m() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..1000000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 1000226);
        assert!(!hll.is_empty());
    }

    #[test]
    fn clear() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..1000 {
            hll.add(&i);
        }
        hll.clear();
        assert_eq!(hll.count(), 0);
        assert!(hll.is_empty());
    }

    #[test]
    fn clone() {
        let mut hll1 = HyperLogLog::new(12);
        for i in 0..500 {
            hll1.add(&i);
        }
        let c1a = hll1.count();

        let hll2 = hll1.clone();
        assert_eq!(hll2.count(), c1a);

        for i in 501..1000 {
            hll1.add(&i);
        }
        let c1b = hll1.count();
        assert_ne!(c1b, c1a);
        assert_eq!(hll2.count(), c1a);
    }

    #[test]
    fn merge() {
        let mut hll1 = HyperLogLog::new(12);
        let mut hll2 = HyperLogLog::new(12);
        let mut hll = HyperLogLog::new(12);
        for i in 0..500 {
            hll.add(&i);
            hll1.add(&i);
        }
        for i in 501..1000 {
            hll.add(&i);
            hll2.add(&i);
        }
        assert_ne!(hll.count(), hll1.count());
        assert_ne!(hll.count(), hll2.count());

        hll1.merge(&hll2);
        assert_eq!(hll.count(), hll1.count());
    }

    #[test]
    #[should_panic(expected = "b must be equal (left=5, right=12)")]
    fn merge_panics_p() {
        let mut hll1 = HyperLogLog::<u64>::new(5);
        let hll2 = HyperLogLog::<u64>::new(12);
        hll1.merge(&hll2);
    }

    #[test]
    #[should_panic(expected = "buildhasher must be equal")]
    fn merge_panics_buildhasher() {
        let mut hll1 =
            HyperLogLog::<u64, BuildHasherSeeded>::with_hash(12, BuildHasherSeeded::new(0));
        let hll2 = HyperLogLog::<u64, BuildHasherSeeded>::with_hash(12, BuildHasherSeeded::new(1));
        hll1.merge(&hll2);
    }

    #[test]
    fn debug() {
        let hll = HyperLogLog::<u64>::new(12);
        assert_eq!(format!("{:?}", hll), "HyperLogLog { b: 12 }");
    }

    #[test]
    fn extend() {
        let mut hll = HyperLogLog::new(4);
        hll.extend(0..1000);
        assert_eq!(hll.count(), 571);
        assert!(!hll.is_empty());
    }
}
