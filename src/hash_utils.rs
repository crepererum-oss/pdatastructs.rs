//! Hash-related utils.
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker;

/// `Iterator` that creates `h_i(x)` for a given value.
///
/// This can be used for algorithms and data structures that require you to implement multiple hash
/// functions.
///
/// This is implemented by creating a new `Hash` for each result and seed it with an index value,
/// hashing the actual payload and then finalizing the `Hash`.
#[derive(Debug)]
pub struct HashIter<'a, 'b, T, B>
where
    T: 'a + Hash,
    B: 'b + BuildHasher,
{
    m: usize,
    k: usize,
    i: usize,
    obj: &'a T,
    buildhasher: &'b B,
}

impl<'a, 'b, T, B> HashIter<'a, 'b, T, B>
where
    T: 'a + Hash,
    B: 'b + BuildHasher,
{
    /// Create new `HashIter` with the following parameters:
    ///
    /// - `m`: the maximal result value of `h_i(x)`
    /// - `k`: number of hash functions to generate
    /// - `obj`: the object that should be hashed, i.e. `x` in `h_i(x)`
    /// - `buildhasher`: `BuildHasher` used to construct new `Hash` objects, must be stable (i.e.
    ///   create the same `Hash` object on every call)
    pub fn new(m: usize, k: usize, obj: &'a T, buildhasher: &'b B) -> HashIter<'a, 'b, T, B> {
        HashIter {
            m: m,
            k: k,
            i: 0,
            obj: obj,
            buildhasher: buildhasher,
        }
    }
}

impl<'a, 'b, T, B> Iterator for HashIter<'a, 'b, T, B>
where
    T: 'a + Hash,
    B: 'b + BuildHasher,
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.i < self.k {
            let mut hasher = self.buildhasher.build_hasher();
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

/// Like `BuildHasherDefault` but implements `Eq`.
pub struct MyBuildHasherDefault<H>(marker::PhantomData<H>);

impl<H> fmt::Debug for MyBuildHasherDefault<H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("BuildHasherDefault")
    }
}

impl<H: Default + Hasher> BuildHasher for MyBuildHasherDefault<H> {
    type Hasher = H;

    fn build_hasher(&self) -> H {
        H::default()
    }
}

impl<H> Clone for MyBuildHasherDefault<H> {
    fn clone(&self) -> MyBuildHasherDefault<H> {
        MyBuildHasherDefault(marker::PhantomData)
    }
}

impl<H> Default for MyBuildHasherDefault<H> {
    fn default() -> MyBuildHasherDefault<H> {
        MyBuildHasherDefault(marker::PhantomData)
    }
}

impl<H> PartialEq for MyBuildHasherDefault<H> {
    fn eq(&self, _other: &MyBuildHasherDefault<H>) -> bool {
        true
    }
}
impl<H> Eq for MyBuildHasherDefault<H> {}

/// BuildHasher that takes a seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildHasherSeeded {
    seed: usize,
}

impl BuildHasherSeeded {
    /// Create new BuildHasherSeeded with given seed.
    pub fn new(seed: usize) -> Self {
        Self { seed: seed }
    }
}

impl BuildHasher for BuildHasherSeeded {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        let mut h = DefaultHasher::default();
        h.write_usize(self.seed);
        h
    }
}

#[cfg(test)]
mod tests {
    use super::{HashIter, MyBuildHasherDefault};
    use std::collections::hash_map::DefaultHasher;

    #[test]
    fn hash_iter() {
        let bh = MyBuildHasherDefault::<DefaultHasher>::default();
        let obj = 1337;

        let iter1 = HashIter::new(42, 2, &obj, &bh);
        let v1: Vec<usize> = iter1.collect();
        assert_eq!(v1.len(), 2);
        assert!(v1[0] < 42);
        assert!(v1[1] < 42);
        assert_ne!(v1[0], v1[1]);

        let iter2 = HashIter::new(42, 2, &obj, &bh);
        let v2: Vec<usize> = iter2.collect();
        assert_eq!(v1, v2);
    }
}
