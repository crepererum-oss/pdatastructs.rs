use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use bit_vec::BitVec;


struct HashIter<'a, T>
where
    T: 'a,
{
    m: usize,
    k: usize,
    i: usize,
    obj: &'a T,
}


impl<'a, T> HashIter<'a, T>
where
    T: 'a,
{
    fn new(m: usize, k: usize, obj: &'a T) -> HashIter<'a, T> {
        HashIter {
            m: m,
            k: k,
            i: 0,
            obj: obj,
        }
    }
}


impl<'a, T> Iterator for HashIter<'a, T>
where
    T: 'a + Hash,
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.i < self.k {
            let mut hasher = DefaultHasher::new();
            hasher.write_usize(self.i);
            self.obj.hash(&mut hasher);
            let x = (hasher.finish() as usize) % self.m;

            self.i += 1;

            Some(x)
        } else {
            None
        }
    }
}


pub struct BloomFilter {
    bv: BitVec,
    k: usize,
}


impl BloomFilter {
    pub fn with_params(m: usize, k: usize) -> BloomFilter {
        BloomFilter {
            bv: BitVec::from_elem(m, false),
            k: k,
        }
    }

    pub fn with_properties(n: usize, p: f64) -> BloomFilter {
        assert!(n > 0);
        assert!(p > 0.);
        assert!(p < 1.);

        let k = (-p.log2()) as usize;
        let ln2 = (2f64).ln();
        let m = (-((n as f64) * p.ln()) / (ln2 * ln2)) as usize;

        BloomFilter::with_params(m, k)
    }

    pub fn get_k(&self) -> usize {
        self.k
    }

    pub fn get_m(&self) -> usize {
        self.bv.len()
    }

    pub fn add<T>(&mut self, obj: &T)
    where
        T: Hash,
    {
        for pos in HashIter::new(self.bv.len(), self.k, obj) {
            self.bv.set(pos, true);
        }
    }

    pub fn query<T>(&self, obj: &T) -> bool
    where
        T: Hash,
    {
        for pos in HashIter::new(self.bv.len(), self.k, obj) {
            if !self.bv.get(pos).unwrap() {
                return false;
            }
        }
        true
    }

    pub fn clear(&mut self) {
        self.bv.clear()
    }

    pub fn union(&mut self, other: &BloomFilter) {
        assert!(self.k == other.k);
        assert!(self.bv.len() == other.bv.len());
        self.bv.union(&other.bv);
    }

    pub fn guess_n(&self) -> usize {
        let m = self.bv.len() as f64;
        let k = self.k as f64;
        let x = self.bv.iter().filter(|x| *x).count() as f64;

        (-m / k * (1. - x / m).ln()) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::BloomFilter;

    #[test]
    fn getter() {
        let bf = BloomFilter::with_params(100, 2);
        assert!(bf.get_k() == 2);
        assert!(bf.get_m() == 100);
    }

    #[test]
    fn empty() {
        let bf = BloomFilter::with_params(100, 2);
        assert!(!bf.query(&1));
    }

    #[test]
    fn add() {
        let mut bf = BloomFilter::with_params(100, 2);

        bf.add(&1);
        assert!(bf.query(&1));
        assert!(!bf.query(&2));
    }

    #[test]
    fn clear() {
        let mut bf = BloomFilter::with_params(100, 2);

        bf.add(&1);
        bf.clear();
        assert!(!bf.query(&1));
    }

    #[test]
    fn union() {
        let mut bf1 = BloomFilter::with_params(100, 2);
        bf1.add(&1);
        assert!(bf1.query(&1));
        assert!(!bf1.query(&2));
        assert!(!bf1.query(&3));

        let mut bf2 = BloomFilter::with_params(100, 2);
        bf2.add(&2);
        assert!(!bf2.query(&1));
        assert!(bf2.query(&2));
        assert!(!bf2.query(&3));

        bf1.union(&bf2);
        assert!(bf1.query(&1));
        assert!(bf1.query(&2));
        assert!(!bf1.query(&3));
    }

    #[test]
    fn with_properties() {
        let bf = BloomFilter::with_properties(1000, 0.1);
        assert!(bf.get_k() == 3);
        assert!(bf.get_m() == 4792);
    }

    #[test]
    fn guess_n() {
        let mut bf = BloomFilter::with_params(100, 2);
        assert!(bf.guess_n() == 0);

        bf.add(&1);
        assert!(bf.guess_n() == 1);

        bf.add(&1);
        assert!(bf.guess_n() == 1);

        bf.add(&2);
        assert!(bf.guess_n() == 2);
    }
}
