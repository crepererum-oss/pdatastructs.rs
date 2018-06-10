//! `HyperLogLog` implementation.
use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};

use hash_utils::MyBuildHasherDefault;

/// A simple implementation of a [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog)
#[derive(Clone)]
pub struct HyperLogLog<B = MyBuildHasherDefault<DefaultHasher>>
where
    B: BuildHasher + Clone + Eq,
{
    registers: Vec<u8>,
    b: usize,
    buildhasher: B,
}

impl HyperLogLog {
    /// Creates a new, empty HyperLogLog.
    ///
    /// - `b` number of bits used for register selection, number of registers within the
    ///   HyperLogLog will be `2^b`. `b` must be in `[4, 16]`
    ///
    /// Panics when `b` is out of bounds.
    pub fn new(b: usize) -> HyperLogLog {
        let bh = MyBuildHasherDefault::<DefaultHasher>::default();
        Self::with_hash(b, bh)
    }
}

impl<B> HyperLogLog<B>
where
    B: BuildHasher + Clone + Eq,
{
    /// Same as `new` but with a specific `BuildHasher`.
    pub fn with_hash(b: usize, buildhasher: B) -> HyperLogLog<B> {
        assert!(
            (b >= 4) & (b <= 16),
            "b ({}) must be larger or equal than 4 and smaller or equal than 16",
            b
        );

        let m = (1 as usize) << b;
        let registers = vec![0; m];
        HyperLogLog {
            registers: registers,
            b: b,
            buildhasher: buildhasher,
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

    /// Adds an element to the HyperLogLog.
    pub fn add<T>(&mut self, obj: &T)
    where
        T: Hash,
    {
        let mut hasher = self.buildhasher.build_hasher();
        obj.hash(&mut hasher);
        let h: u64 = hasher.finish();

        let w = h >> self.b;
        let j = h - (w << self.b);
        let p = w.leading_zeros() + 1 - (self.b as u32);

        let m_old = self.registers[j as usize];
        self.registers[j as usize] = cmp::max(m_old, p as u8);
    }

    /// Guess the number of unique elements seen by the HyperLogLog.
    pub fn count(&self) -> usize {
        let m = self.registers.len() as f64;

        let z = 1f64 / self.registers
            .iter()
            .map(|&x| 2f64.powi(-(x as i32)))
            .sum::<f64>();

        let am = if m >= 128. {
            0.7213 / (1. + 1.079 / m)
        } else if m >= 64. {
            0.709
        } else if m >= 32. {
            0.697
        } else {
            0.673
        };

        let e = am * m * m * z;

        let e_star = if e <= 5. / 2. * m {
            // small range correction
            let v = self.registers.iter().filter(|&&x| x == 0).count();
            if v != 0 {
                m * (m / (v as f64)).ln()
            } else {
                e
            }
        } else if e <= 1. / 30. * 2f64.powi(32) {
            // intermediate range => no correction
            e
        } else {
            // large range correction
            -2f64.powi(32) * (1. - e / 2f64.powi(32)).ln()
        };

        e_star as usize
    }

    /// Merge w/ another HyperLogLog.
    ///
    /// This HyperLogLog will then have the same state as if all elements seen by `other` where
    /// directly added to `self`.
    ///
    /// Panics when `b` or `buildhasher` parameter of `self` and `other` do not match.
    pub fn merge(&mut self, other: &HyperLogLog<B>) {
        assert_eq!(
            self.b, other.b,
            "b must be equal (left={}, right={})",
            self.b, other.b
        );
        assert!(
            self.buildhasher == other.buildhasher,
            "buildhasher must be equal"
        );

        self.registers = self.registers
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

impl fmt::Debug for HyperLogLog {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HyperLogLog {{ b: {} }}", self.b)
    }
}

impl<T> Extend<T> for HyperLogLog
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
        HyperLogLog::new(3);
    }

    #[test]
    #[should_panic(expected = "b (17) must be larger or equal than 4 and smaller or equal than 16")]
    fn new_panics_b17() {
        HyperLogLog::new(17);
    }

    #[test]
    fn getter() {
        let hll = HyperLogLog::new(8);
        assert_eq!(hll.b(), 8);
        assert_eq!(hll.m(), 1 << 8);
        hll.buildhasher();
    }

    #[test]
    fn empty() {
        let hll = HyperLogLog::new(8);
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
        assert_eq!(hll.count(), 966);
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
        assert_eq!(hll.count(), 10303);
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
        assert_eq!(hll.count(), 100551);
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
        let mut hll1 = HyperLogLog::new(5);
        let hll2 = HyperLogLog::new(12);
        hll1.merge(&hll2);
    }

    #[test]
    #[should_panic(expected = "buildhasher must be equal")]
    fn merge_panics_buildhasher() {
        let mut hll1 = HyperLogLog::with_hash(12, BuildHasherSeeded::new(0));
        let hll2 = HyperLogLog::with_hash(12, BuildHasherSeeded::new(1));
        hll1.merge(&hll2);
    }

    #[test]
    fn debug() {
        let hll = HyperLogLog::new(12);
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
