use std::hash::{BuildHasher, Hash, Hasher};


pub struct HashIter<'a, T, B>
where
    T: 'a,
    B: BuildHasher,
{
    m: usize,
    k: usize,
    i: usize,
    obj: &'a T,
    buildhasher: B,
}


impl<'a, T, B> HashIter<'a, T, B>
where
    T: 'a,
    B: BuildHasher,
{
    pub fn new(m: usize, k: usize, obj: &'a T, buildhasher: B) -> HashIter<'a, T, B> {
        HashIter {
            m: m,
            k: k,
            i: 0,
            obj: obj,
            buildhasher: buildhasher,
        }
    }
}


impl<'a, T, B> Iterator for HashIter<'a, T, B>
where
    T: 'a + Hash,
    B: BuildHasher,
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
