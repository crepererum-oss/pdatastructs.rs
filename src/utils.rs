use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker;

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
