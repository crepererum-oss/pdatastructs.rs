//! Implementation of `Filter` for certain non-probabilistic data structures. This can be helpful
//! for debugging and performance comparisons.
use std::collections::HashSet;
use std::convert::Infallible;
use std::hash::{BuildHasher, Hash};

use crate::filters::Filter;

impl<T, S> Filter<T> for HashSet<T, S>
where
    T: Clone + Eq + Hash,
    S: BuildHasher,
{
    type InsertErr = Infallible;

    fn clear(&mut self) {
        self.clear();
    }

    fn insert(&mut self, obj: &T) -> Result<bool, Self::InsertErr> {
        Ok(self.insert(obj.clone()))
    }

    fn union(&mut self, other: &Self) -> Result<(), Self::InsertErr> {
        self.extend(other.iter().cloned());
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn query(&self, obj: &T) -> bool {
        self.contains(obj)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::filters::Filter;

    #[test]
    fn hashset() {
        type H = HashSet<u64>;

        let mut set1: H = HashSet::new();
        assert!(<H as Filter<u64>>::is_empty(&set1));
        assert_eq!(<H as Filter<u64>>::len(&set1), 0);
        assert!(!<H as Filter<u64>>::query(&set1, &42));

        assert!(<H as Filter<u64>>::insert(&mut set1, &42).unwrap());
        assert!(!<H as Filter<u64>>::insert(&mut set1, &42).unwrap());
        assert!(!<H as Filter<u64>>::is_empty(&set1));
        assert_eq!(<H as Filter<u64>>::len(&set1), 1);
        assert!(<H as Filter<u64>>::query(&set1, &42));
        assert!(!<H as Filter<u64>>::query(&set1, &13));

        let mut set2: H = HashSet::new();
        set2.insert(1337);
        <H as Filter<u64>>::union(&mut set1, &set2).unwrap();
        assert!(<H as Filter<u64>>::query(&set1, &42));
        assert!(!<H as Filter<u64>>::query(&set1, &13));
        assert!(<H as Filter<u64>>::query(&set1, &1337));
    }
}
