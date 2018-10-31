//! Implementation of `Filter` for certain non-probabilistic data structures. This can be helpful
//! for debugging and performance comparisons.
use std::collections::HashSet;
use std::hash::Hash;

use void::Void;

use filters::Filter;

impl<T> Filter<T> for HashSet<T>
where
    T: Clone + Eq + Hash,
{
    type InsertErr = Void;

    fn clear(&mut self) {
        self.clear();
    }

    fn insert(&mut self, obj: &T) -> Result<(), Self::InsertErr> {
        self.insert(obj.clone());
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

    use void::Void;

    use filters::Filter;

    #[test]
    fn hashset() {
        let set: &mut Filter<u64, InsertErr = Void> = &mut HashSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(!set.query(&42));

        set.insert(&42).unwrap();
        assert!(!set.is_empty());
        assert_eq!(set.len(), 1);
        assert!(set.query(&42));
        assert!(!set.query(&13));
    }
}
