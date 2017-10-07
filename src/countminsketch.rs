use std::collections::hash_map::DefaultHasher;
use std::f64;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash};

use utils::HashIter;


/// Abstract, but safe counter.
pub trait Counter: Copy + fmt::Debug + Ord + Sized {
    /// Add self to another counter.
    ///
    /// Returns `Some(Self)` when the addition was successfull (i.e. no overflow occured) and
    /// `None` in case of an error.
    fn checked_add(self, other: Self) -> Option<Self>;

    /// Return a counter representing zero.
    fn zero() -> Self;

    /// Return a counter representing one.
    fn one() -> Self;

    /// Checks whether the counter is zero.
    fn is_zero(&self) -> bool;
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

            #[inline]
            fn is_zero(&self) -> bool {
                *self == 0
            }
        }
    }
}


impl_counter!(usize);
impl_counter!(u64);
impl_counter!(u32);
impl_counter!(u16);
impl_counter!(u8);


/// Simple implementation of a
/// [Count-min sketch](https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch)
///
/// The type parameter `C` sets the type of the counter in the internal table and can be used to
/// reduce memory consumption when low counts are expected.
#[derive(Clone)]
pub struct CountMinSketch<C = usize>
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
    /// Create new CountMinSketch based on table size.
    ///
    /// - `w` sets the number of columns
    /// - `d` sets the number of rows
    pub fn with_params(w: usize, d: usize) -> CountMinSketch<C> {
        let table = vec![C::zero(); w.checked_mul(d).unwrap()];
        CountMinSketch {
            table: table,
            w: w,
            d: d,
        }
    }

    /// Create new CountMinSketch with the following properties for a point query:
    ///
    /// - `a` is the real count of observed objects
    /// - `a'` is the guessed count of observed objects
    /// - `N` is the total count in the internal table
    /// - `a <= a'` always holds
    /// - `a' <= a + epsilon * N` holds with `p > 1 - delta`
    ///
    /// The following conditions must hold:
    ///
    /// - `epsilon > 0`
    /// - `delta > 0` and `delta < 1`
    ///
    /// Panics when the input conditions do not hold.
    pub fn with_point_query_properties(epsilon: f64, delta: f64) -> CountMinSketch<C> {
        assert!(epsilon > 0.);
        assert!(delta > 0.);
        assert!(delta < 1.);

        let w = (f64::consts::E / epsilon).ceil() as usize;
        let d = (1. / delta).ln().ceil() as usize;
        CountMinSketch::<C>::with_params(w, d)
    }

    /// Get number of columns of internal counter table.
    pub fn w(&self) -> usize {
        self.w
    }

    /// Get number of rows of internal counter table.
    pub fn d(&self) -> usize {
        self.d
    }

    /// Check whether the CountMinSketch is empty (i.e. no elements seen yet).
    pub fn is_empty(&self) -> bool {
        self.table.iter().all(|x| x.is_zero())
    }

    /// Add one to the counter of the given element.
    pub fn add<T>(&mut self, obj: &T)
    where
        T: Hash,
    {
        self.add_n(&obj, C::one())
    }

    /// Add `n` to the counter of the given element.
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

    /// Runs a point query, i.e. a query for the count of a single object.
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

    /// Merge self with another CountMinSketch.
    ///
    /// After this operation `self` will be in the same state as when it would have seen all
    /// elements from `self` and `other`.
    ///
    /// Panics when `d` and `w` from `self` and `other` differ.
    pub fn merge(&mut self, other: &CountMinSketch<C>) {
        assert_eq!(self.d, other.d);
        assert_eq!(self.w, other.w);

        self.table = self.table
            .iter()
            .zip(other.table.iter())
            .map(|x| x.0.checked_add(*x.1).unwrap())
            .collect();
    }

    /// Clear internal counters to a fresh state (i.e. no objects seen).
    pub fn clear(&mut self) {
        self.table = vec![C::zero(); self.w.checked_mul(self.d).unwrap()];
    }
}


impl fmt::Debug for CountMinSketch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CountMinSketch {{ w: {}, d: {} }}", self.w, self.d)
    }
}


#[cfg(test)]
mod tests {
    use super::CountMinSketch;

    #[test]
    fn getter() {
        let cms = CountMinSketch::<usize>::with_params(10, 20);
        assert_eq!(cms.w(), 10);
        assert_eq!(cms.d(), 20);
    }

    #[test]
    fn properties() {
        let cms = CountMinSketch::<usize>::with_point_query_properties(0.01, 0.1);
        assert_eq!(cms.w(), 272);
        assert_eq!(cms.d(), 3);
    }

    #[test]
    fn empty() {
        let cms = CountMinSketch::<usize>::with_params(10, 10);
        assert_eq!(cms.query_point(&1), 0);
        assert!(cms.is_empty());
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
    fn add_2_1a() {
        let mut cms = CountMinSketch::<usize>::with_params(10, 10);

        cms.add(&1);
        cms.add(&2);
        cms.add(&1);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 1);
        assert_eq!(cms.query_point(&3), 0);
    }

    #[test]
    fn add_2_1b() {
        let mut cms = CountMinSketch::<usize>::with_params(10, 10);

        cms.add_n(&1, 2);
        cms.add(&2);
        assert_eq!(cms.query_point(&1), 2);
        assert_eq!(cms.query_point(&2), 1);
        assert_eq!(cms.query_point(&3), 0);
    }

    #[test]
    fn merge() {
        let mut cms1 = CountMinSketch::<usize>::with_params(10, 10);
        let mut cms2 = CountMinSketch::<usize>::with_params(10, 10);

        cms1.add_n(&1, 1);
        cms1.add_n(&2, 2);
        assert_eq!(cms1.query_point(&1), 1);
        assert_eq!(cms1.query_point(&2), 2);
        assert_eq!(cms1.query_point(&3), 0);
        assert_eq!(cms1.query_point(&4), 0);

        cms2.add_n(&2, 20);
        cms2.add_n(&3, 30);
        assert_eq!(cms2.query_point(&1), 0);
        assert_eq!(cms2.query_point(&2), 20);
        assert_eq!(cms2.query_point(&3), 30);
        assert_eq!(cms2.query_point(&4), 0);

        cms1.merge(&cms2);
        assert_eq!(cms1.query_point(&1), 1);
        assert_eq!(cms1.query_point(&2), 22);
        assert_eq!(cms1.query_point(&3), 30);
        assert_eq!(cms1.query_point(&4), 0);
    }

    #[test]
    fn clear() {
        let mut cms = CountMinSketch::<usize>::with_params(10, 10);

        cms.add(&1);
        assert_eq!(cms.query_point(&1), 1);
        assert!(!cms.is_empty());

        cms.clear();
        assert_eq!(cms.query_point(&1), 0);
        assert!(cms.is_empty());
    }

    #[test]
    fn clone() {
        let mut cms1 = CountMinSketch::<usize>::with_params(10, 10);

        cms1.add(&1);
        assert_eq!(cms1.query_point(&1), 1);
        assert_eq!(cms1.query_point(&2), 0);

        let cms2 = cms1.clone();
        assert_eq!(cms2.query_point(&1), 1);
        assert_eq!(cms2.query_point(&2), 0);

        cms1.add(&1);
        assert_eq!(cms1.query_point(&1), 2);
        assert_eq!(cms1.query_point(&2), 0);
        assert_eq!(cms2.query_point(&1), 1);
        assert_eq!(cms2.query_point(&2), 0);
    }

    #[test]
    fn debug() {
        let cms = CountMinSketch::<usize>::with_params(10, 20);
        assert_eq!(format!("{:?}", cms), "CountMinSketch { w: 10, d: 20 }");
    }
}
