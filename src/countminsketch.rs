use std::collections::hash_map::DefaultHasher;
use std::hash::{BuildHasherDefault, Hash};

use utils::HashIter;


pub trait Counter: Copy + Ord + Sized {
    fn checked_add(self, other: Self) -> Option<Self>;

    fn zero() -> Self;

    fn one() -> Self;
}


macro_rules! impl_counter {
    ($t:ty) => {
        impl Counter for $t {
            #[inline]
            fn checked_add(self, other: Self) -> Option<Self> {
                self.checked_add(other)
            }

            #[inline]
            fn zero() -> Self {
                0
            }

            #[inline]
            fn one() -> Self {
                1
            }
        }
    }
}


impl_counter!(usize);
impl_counter!(u64);
impl_counter!(u32);
impl_counter!(u16);
impl_counter!(u8);


pub struct CountMinSketch<C>
where
    C: Counter,
{
    table: Vec<C>,
    w: usize,
    d: usize,
}


impl<C> CountMinSketch<C>
where
    C: Counter,
{
    pub fn with_params(w: usize, d: usize) -> CountMinSketch<C> {
        let table = vec![C::zero(); w.checked_mul(d).unwrap()];
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
        self.add_n(&obj, C::one())
    }

    pub fn add_n<T>(&mut self, obj: &T, n: C)
    where
        T: Hash,
    {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        for (i, pos) in HashIter::new(self.w, self.d, obj, bh).enumerate() {
            let x = i * self.w + pos;
            self.table[x] = self.table[x].checked_add(n).unwrap();
        }
    }

    pub fn query_point<T>(&self, obj: &T) -> C
    where
        T: Hash,
    {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        HashIter::new(self.w, self.d, obj, bh)
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
        let cms = CountMinSketch::<usize>::with_params(10, 10);
        assert_eq!(cms.query_point(&1), 0);
    }

    #[test]
    fn add_1() {
        let mut cms = CountMinSketch::<usize>::with_params(10, 10);

        cms.add(&1);
        assert_eq!(cms.query_point(&1), 1);
        assert_eq!(cms.query_point(&2), 0);
    }

    #[test]
    fn add_2() {
        let mut cms = CountMinSketch::<usize>::with_params(10, 10);

        cms.add(&1);
        cms.add(&1);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 0);
    }

    #[test]
    fn add_2_1() {
        let mut cms = CountMinSketch::<usize>::with_params(10, 10);

        cms.add(&1);
        cms.add(&2);
        cms.add(&1);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 1);
        assert_eq!(cms.query_point(&3), 0);
    }
}
