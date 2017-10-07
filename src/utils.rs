use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};


pub struct HashIter<'a, T>
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
    pub fn new(m: usize, k: usize, obj: &'a T) -> HashIter<'a, T> {
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
