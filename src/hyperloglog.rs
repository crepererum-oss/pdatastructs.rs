use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};


pub struct HyperLogLog {
    registers: Vec<u64>,
    b: usize,
}


impl HyperLogLog {
    pub fn new(b: usize) -> HyperLogLog {
        assert!(b >= 4);
        assert!(b <= 16);

        let m = (1 as usize) << b;
        let registers = (0..m).map(|_| 0).collect();
        HyperLogLog {
            registers: registers,
            b: b,
        }
    }

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
        self.registers[j as usize] = cmp::max(m_old, p as u64);
    }

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
}


#[cfg(test)]
mod tests {
    use super::HyperLogLog;

    #[test]
    fn empty() {
        let hll = HyperLogLog::new(8);
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn add_b4_n1k() {
        let mut hll = HyperLogLog::new(4);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 571);
    }

    #[test]
    fn add_b8_n1k() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 966);
    }

    #[test]
    fn add_b12_n1k() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 984);
    }

    #[test]
    fn add_b16_n1k() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..1000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 998);
    }

    #[test]
    fn add_b8_n10k() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..10000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 10196);
    }

    #[test]
    fn add_b12_n10k() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..10000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 10303);
    }

    #[test]
    fn add_b16_n10k() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..10000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 10055);
    }

    #[test]
    fn add_b16_n100k() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..100000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 100551);
    }

    #[test]
    fn add_b16_n1m() {
        let mut hll = HyperLogLog::new(16);
        for i in 0..1000000 {
            hll.add(&i);
        }
        assert_eq!(hll.count(), 1000226);
    }
}
