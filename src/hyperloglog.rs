use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};


/// A simple implementation of a [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog)
pub struct HyperLogLog {
    registers: Vec<u8>,
    b: usize,
}


impl HyperLogLog {
    /// Creates a new, empty HyperLogLog.
    ///
    /// - `b` number of bits used for register selection, number of registers within the
    ///   HyperLogLog will be `2^b`. `b` must be in `[4, 16]`
    ///
    /// Panics when `b` is out of bounds.
    pub fn new(b: usize) -> HyperLogLog {
        assert!(b >= 4);
        assert!(b <= 16);

        let m = (1 as usize) << b;
        let registers = vec![0; m];
        HyperLogLog {
            registers: registers,
            b: b,
        }
    }

    /// Adds an element to the HyperLogLog.
    pub fn add<T>(&mut self, obj: &T)
    where
        T: Hash,
    {
        let mut hasher = DefaultHasher::new();
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

        let z = 1f64 /
            self.registers
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

    /// Empties the HyperLogLog.
    pub fn clear(&mut self) {
        self.registers = vec![0; self.registers.len()];
    }

    /// Checks whether the HyperLogLog has never seen an element.
    pub fn is_empty(&self) -> bool {
        self.registers.iter().all(|&x| x == 0)
    }
}


#[cfg(test)]
mod tests {
    use super::HyperLogLog;

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
}
