use std::hash::{Hash, Hasher};


pub struct HashIter<'a, T, H>
where
    T: 'a,
    H: Clone + Hasher,
{
    m: usize,
    k: usize,
    i: usize,
    obj: &'a T,
    hasher: H,
}


impl<'a, T, H> HashIter<'a, T, H>
where
    T: 'a,
    H: Clone + Hasher,
{
    pub fn new(m: usize, k: usize, obj: &'a T, hasher: H) -> HashIter<'a, T, H> {
        HashIter {
            m: m,
            k: k,
            i: 0,
            obj: obj,
            hasher: hasher,
        }
    }
}


impl<'a, T, H> Iterator for HashIter<'a, T, H>
where
    T: 'a + Hash,
    H: Clone + Hasher,
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.i < self.k {
            let mut hasher = self.hasher.clone();
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
