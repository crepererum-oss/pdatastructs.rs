use std::hash::Hash;

use utils::HashIter;


pub struct CountMinSketch {
    table: Vec<usize>,
    w: usize,
    d: usize,
}


impl CountMinSketch {
    pub fn with_params(w: usize, d: usize) -> CountMinSketch {
        let table = vec![0; w.checked_mul(d).unwrap()];
        CountMinSketch {
            table: table,
            w: w,
            d: d,
        }
    }

    pub fn add<T>(&mut self, obj: &T)
    where
        T: Hash,
    {
        self.add_n(&obj, 1)
    }

    pub fn add_n<T>(&mut self, obj: &T, n: usize)
    where
        T: Hash,
    {
        for (i, pos) in HashIter::new(self.w, self.d, obj).enumerate() {
            let x = i * self.w + pos;
            self.table[x] = self.table[x].checked_add(n).unwrap();
        }
    }

    pub fn query_point<T>(&self, obj: &T) -> usize
    where
        T: Hash,
    {
        HashIter::new(self.w, self.d, obj)
            .enumerate()
            .map(|(i, pos)| i * self.w + pos)
            .map(|x| self.table[x])
            .min()
            .unwrap()
    }
}


#[cfg(test)]
mod tests {
    use super::CountMinSketch;

    #[test]
    fn empty() {
        let cms = CountMinSketch::with_params(10, 10);
        assert_eq!(cms.query_point(&1), 0);
    }

    #[test]
    fn add_1() {
        let mut cms = CountMinSketch::with_params(10, 10);

        cms.add(&1);
        assert_eq!(cms.query_point(&1), 1);
        assert_eq!(cms.query_point(&2), 0);
    }

    #[test]
    fn add_2() {
        let mut cms = CountMinSketch::with_params(10, 10);

        cms.add(&1);
        cms.add(&1);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 0);
    }

    #[test]
    fn add_2_1() {
        let mut cms = CountMinSketch::with_params(10, 10);

        cms.add(&1);
        cms.add(&2);
        cms.add(&1);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 1);
        assert_eq!(cms.query_point(&3), 0);
    }
}
