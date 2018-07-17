//! `TopK` implementation.
use countminsketch::CountMinSketch;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::hash::Hash;
use std::rc::Rc;

#[derive(Debug)]
struct TreeEntry<T>
where
    T: Eq + Ord,
{
    obj: Rc<T>,
    n: usize,
}

impl<T> Clone for TreeEntry<T>
where
    T: Eq + Ord,
{
    fn clone(&self) -> Self {
        TreeEntry {
            obj: Rc::clone(&self.obj),
            n: self.n,
        }
    }
}

impl<T> PartialEq for TreeEntry<T>
where
    T: Eq + Ord,
{
    fn eq(&self, other: &TreeEntry<T>) -> bool {
        self.obj == other.obj
    }
}

impl<T> Eq for TreeEntry<T>
where
    T: Eq + Ord,
{
}

impl<T> PartialOrd for TreeEntry<T>
where
    T: Eq + Ord,
{
    fn partial_cmp(&self, other: &TreeEntry<T>) -> Option<Ordering> {
        match self.n.cmp(&other.n) {
            Ordering::Greater => Some(Ordering::Greater),
            Ordering::Less => Some(Ordering::Less),
            Ordering::Equal => Some(self.obj.cmp(&other.obj)),
        }
    }
}

impl<T> Ord for TreeEntry<T>
where
    T: Eq + Ord,
{
    fn cmp(&self, other: &TreeEntry<T>) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Top-K implementation.
///
/// This data structure keeps the `k` most frequent data points of a stream.
#[derive(Clone)]
pub struct TopK<T>
where
    T: Clone + Eq + Hash + Ord,
{
    cms: CountMinSketch,
    obj2count: HashMap<Rc<T>, usize>,
    tree: BTreeSet<TreeEntry<T>>,
    k: usize,
}

impl<T> TopK<T>
where
    T: Clone + Eq + Hash + Ord,
{
    /// Create new Top-K data structure.
    ///
    /// - `k`: number of elements to remember
    /// - `cms`: CountMinSketch that is used for guessing the frequency of elements not currently
    ///   hold. Refer to its documentation about parameter selection.
    ///
    /// Panics if `k == 0`.
    pub fn new(k: usize, cms: CountMinSketch) -> Self {
        assert!(k > 0, "k must be greater than 0");

        Self {
            cms,
            obj2count: HashMap::new(),
            tree: BTreeSet::new(),
            k,
        }
    }

    /// Number of data points to remember.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Observe a data point.
    pub fn add(&mut self, obj: T) {
        // create Rc so we can use obj in multiple data structs
        let rc = Rc::new(obj);

        // always increase CountMinSketch counts
        self.cms.add(&rc);

        // check if entry is in top K
        let size = self.obj2count.len();
        let mut update_obj2count: Option<(usize, Rc<T>)> = None;
        match self.obj2count.entry(Rc::clone(&rc)) {
            Entry::Occupied(mut o) => {
                // it is => increase exact counter
                let mut n = o.get_mut();
                *n += 1;

                // update tree data
                let mut entry = TreeEntry {
                    obj: Rc::clone(&rc),
                    n: *n - 1,
                };
                self.tree.remove(&entry);
                entry.n += 1;
                self.tree.insert(entry);
            }
            Entry::Vacant(v) => {
                // it's not => check capicity
                if size < self.k {
                    // space left => add to top k
                    v.insert(1);
                    self.tree.insert(TreeEntry {
                        obj: Rc::clone(&rc),
                        n: 1,
                    });
                } else {
                    // not enough space => check if it would be a top k element
                    let count = self.cms.query_point(&rc);
                    let min: TreeEntry<T> = (*self.tree.iter().next().unwrap()).clone();
                    if count > min.n {
                        // => kick out minimal element of top k
                        self.tree.remove(&min);
                        self.tree.insert(TreeEntry {
                            obj: Rc::clone(&rc),
                            n: count,
                        });

                        // delay obj2count changes, because it is currently borrowed
                        update_obj2count = Some((count, min.obj));
                    }
                }
            }
        }

        // trick the borrow checker
        if let Some((count, obj)) = update_obj2count {
            self.obj2count.insert(rc, count);
            self.obj2count.remove(&obj);
        }
    }

    /// Iterates over collected top-k values.
    ///
    /// The result may contain less than `k` values if less than `k` unique data points where
    /// observed.
    pub fn iter<'a>(&'a self) -> impl 'a + Iterator<Item = T> {
        self.tree.iter().map(|x| (*x.obj).clone())
    }

    /// Check whether the sampler is empty (i.e. no data points observer so far)
    pub fn is_empty(&self) -> bool {
        self.obj2count.is_empty()
    }

    /// Clear sampler, so that it behaves like it never observed any data points.
    pub fn clear(&mut self) {
        self.cms.clear();
        self.tree.clear();
        self.obj2count.clear();
    }
}

impl<T> fmt::Debug for TopK<T>
where
    T: Clone + Eq + Hash + Ord,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TopK {{ k: {} }}", self.k)
    }
}

impl<T> Extend<T> for TopK<T>
where
    T: Clone + Eq + Hash + Ord,
{
    fn extend<S: IntoIterator<Item = T>>(&mut self, iter: S) {
        for elem in iter {
            self.add(elem);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TopK;
    use countminsketch::CountMinSketch;

    #[test]
    #[should_panic(expected = "k must be greater than 0")]
    fn new_panics_k0() {
        let cms = CountMinSketch::with_params(10, 20);
        TopK::<usize>::new(0, cms);
    }

    #[test]
    fn getter() {
        let cms = CountMinSketch::with_params(10, 20);
        let tk: TopK<usize> = TopK::new(2, cms);

        assert_eq!(tk.k(), 2);
    }

    #[test]
    fn add_1() {
        let cms = CountMinSketch::with_params(10, 20);
        let mut tk = TopK::new(2, cms);

        tk.add(1);
        assert_eq!(tk.iter().collect::<Vec<u32>>(), vec![1]);
    }

    #[test]
    fn add_2_same() {
        let cms = CountMinSketch::with_params(10, 20);
        let mut tk = TopK::new(2, cms);

        tk.add(1);
        tk.add(1);
        assert_eq!(tk.iter().collect::<Vec<u32>>(), vec![1]);
    }

    #[test]
    fn add_2_different() {
        let cms = CountMinSketch::with_params(10, 20);
        let mut tk = TopK::new(2, cms);

        tk.add(1);
        tk.add(2);
        assert_eq!(tk.iter().collect::<Vec<u32>>(), vec![1, 2]);
    }

    #[test]
    fn add_n() {
        let cms = CountMinSketch::with_params(10, 20);
        let mut tk = TopK::new(2, cms);

        for i in 0..5 {
            tk.add(i);
        }
        for _ in 0..100 {
            tk.add(99);
        }
        for _ in 0..100 {
            tk.add(100);
        }
        for i in 0..5 {
            tk.add(i);
        }
        assert_eq!(tk.iter().collect::<Vec<u32>>(), vec![99, 100]);
    }

    #[test]
    fn is_empty() {
        let cms = CountMinSketch::with_params(10, 20);
        let mut tk = TopK::new(2, cms);
        assert_eq!(tk.is_empty(), true);

        tk.add(0);
        assert_eq!(tk.is_empty(), false);
    }

    #[test]
    fn clear() {
        let cms = CountMinSketch::with_params(10, 20);
        let mut tk = TopK::new(2, cms);
        tk.add(0);

        tk.clear();
        assert_eq!(tk.is_empty(), true);

        tk.add(1);
        assert_eq!(tk.iter().collect::<Vec<u32>>(), vec![1]);
    }

    #[test]
    fn clone() {
        let cms = CountMinSketch::with_params(10, 20);
        let mut tk1 = TopK::new(2, cms);
        tk1.add(0);

        let mut tk2 = tk1.clone();
        tk2.add(1);

        assert_eq!(tk1.iter().collect::<Vec<u32>>(), vec![0]);
        assert_eq!(tk2.iter().collect::<Vec<u32>>(), vec![0, 1]);
    }

    #[test]
    fn debug() {
        let cms = CountMinSketch::with_params(10, 20);
        let tk: TopK<usize> = TopK::new(2, cms);
        assert_eq!(format!("{:?}", tk), "TopK { k: 2 }");
    }

    #[test]
    fn extend() {
        let cms = CountMinSketch::with_params(10, 10);
        let mut tk = TopK::new(2, cms);

        tk.extend(vec![0, 1]);
        assert_eq!(tk.iter().collect::<Vec<u32>>(), vec![0, 1]);
    }
}
