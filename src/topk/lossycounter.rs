//! LossyCounter implementation.

use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;

#[derive(Clone, Debug)]
struct KnownEntry {
    f: usize,
    delta: usize,
}

/// A LossyCounter is a data structure that memorizes the most frequent elements in data stream.
///
/// # Examples
/// ```
/// use pdatastructs::topk::lossycounter::LossyCounter;
///
/// // set up filter
/// let epsilon = 0.01;  // error threshold
/// let mut lc = LossyCounter::with_epsilon(epsilon);
///
/// // add some data
/// for i in 0..1000 {
///     let x = i % 10;
///     let y = if x < 4 {
///         0
///     } else if x < 7 {
///         1
///     } else {
///         x
///     };
///     lc.add(y);
/// }
///
/// // later
/// let threshold = 0.2;
/// let mut elements: Vec<u64> = lc.query(threshold).collect();
/// elements.sort();  // normalize order for testing purposes
/// assert_eq!(elements, vec![0, 1]);
/// ```
///
/// # Applications
/// - getting a fixed number of most common example of a large data set
/// - detect hotspots / very common events
///
/// # How It Works
/// The LossyCounter is initialized with an relative error bound `epsilon`. Depending on that, a
/// window size will be set. A window is the number of elements after which the counter will be
/// pruned.
///
/// ## Insertion
/// If the element is unknown (i.e. it was never seen before or was already pruned), it will be
/// stored with a frequency of 1. Also, the current window index will be stored alongside.
///
/// ```text
/// window size    = 4
///
///
/// insert(a):
///              n = 1
/// current window = 1
///
/// +---------+-------+-----------+
/// | element | delta | frequency |
/// +=========+=======+===========+
/// |       a |     0 |         1 |  <-- inserted
/// +---------+-------+-----------+
///
///
/// insert(b):
///              n = 2
/// current window = 1
///
/// +---------+-------+-----------+
/// | element | delta | frequency |
/// +=========+=======+===========+
/// |       a |     0 |         1 |
/// |       b |     0 |         1 |  <-- inserted
/// +---------+-------+-----------+
/// ```
///
/// If the element is know, the counter will be increased by 1.
///
/// ```text
/// window size    = 4
///
///
/// insert(a):
///              n = 3
/// current window = 1
///
/// +---------+-------+-----------+
/// | element | delta | frequency |
/// +=========+=======+===========+
/// |       a |     0 |         2 |  <-- incremented
/// |       b |     0 |         1 |
/// +---------+-------+-----------+
///
///
/// insert(a):
///              n = 4
/// current window = 1
///
/// +---------+-------+-----------+
/// | element | delta | frequency |
/// +=========+=======+===========+
/// |       a |     0 |         3 |  <-- incremented
/// |       b |     0 |         1 |
/// +---------+-------+-----------+
/// ```
///
/// If the window ends, pruning takes place. Elements will only be kept if their recorded frequency
/// plus the window index stored with them (i.e. the index they where recorded first) is at least
/// as large as the current window index. So elements that where not seen for a while will slowly
/// be kicked out of the counter, hence the name "Lossy Counter". In other words: an element will
/// only stay in the counter if it was seen more often than the number of windows since it was
/// inserted (i.e. number of windows plus 1).
///
/// ```text
/// window size    = 4
///
///
/// prune:
///              n = 4
/// current window = 1
///
/// +---------+-------+-----------+
/// | element | delta | frequency |
/// +=========+=======+===========+
/// |       a |     0 |         2 |
/// |       b |     0 |         1 |  <-- remove
/// +---------+-------+-----------+
///
/// +---------+-------+-----------+
/// | element | delta | frequency |
/// +=========+=======+===========+
/// |       a |     0 |         2 |
/// +---------+-------+-----------+
/// ```
///
/// ## Query
/// Given a relative threshold, all elements will be returned for which the following holds:
///
/// ```text
/// f >= (threshold - epsilon) * n
/// ```
///
/// # See Also
/// - `std::collections::HashSet`: has a false positive rate of 0%, but also needs to store all
///   elements
/// - `pdatastructs::topk::cmsheap::CMSHeap`: uses a different approach to solve the same problem
///
/// # References
/// - ["Approximate Frequency Counts over Data Streams", Gurmeet S. Manku, Rajeev Motwani, 2002](https://www.vldb.org/conf/2002/S10P03.pdf)
/// - [Wikipedia: Lossy Count Algorithm](https://en.wikipedia.org/wiki/Lossy_Count_Algorithm)
/// - ["Top K Frequent Items Algorithm", Zhipeng Jiang, 2017](https://zpjiang.me/2017/11/13/top-k-elementes-system-design/)
#[derive(Clone, Debug)]
pub struct LossyCounter<T>
where
    T: Clone + Eq + Hash,
{
    epsilon: f64,
    known: HashMap<T, KnownEntry>,
    n: usize,
    width: usize,
}

impl<T> LossyCounter<T>
where
    T: Clone + Eq + Hash,
{
    /// Creates new lossy counter so that for reported frequencies `f_e` it holds:
    ///
    /// ```text
    /// f <= f_e <= epsilon * n
    /// ```
    ///
    /// where `f` is the true frequency and `n` the number of data points seen so far.
    pub fn with_epsilon(epsilon: f64) -> Self {
        assert!(
            (epsilon > 0.) & (epsilon < 1.),
            "epsilon ({}) must be greater than 0 and smaller than 1",
            epsilon
        );

        Self {
            epsilon,
            known: HashMap::new(),
            n: 0,
            width: (1. / epsilon).ceil() as usize,
        }
    }

    /// Creates new lossy counter width given window width.
    pub fn with_width(width: usize) -> Self {
        assert!(width > 0, "width must be greater than 0");

        Self {
            epsilon: (1. / (width as f64)),
            known: HashMap::new(),
            n: 0,
            width,
        }
    }

    /// Epsilon error.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Window width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of data points seen so far.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Add new element to the counter.
    ///
    /// Reports, if the element was considered new (`true`) or not (`false`).
    pub fn add(&mut self, t: T) -> bool {
        // pre-inc, since "N denotes the current length of the stream"
        self.n += 1;

        let at_window_end = self.n % self.width == 0;
        let b_current = self.n / self.width + if at_window_end { 0 } else { 1 };

        // add new data
        let was_new = match self.known.entry(t) {
            Entry::Occupied(mut o) => {
                let mut value = o.get_mut();
                value.f += 1;
                false
            }
            Entry::Vacant(v) => {
                v.insert(KnownEntry {
                    f: 1,
                    delta: b_current - 1,
                });
                true
            }
        };

        // pruning
        if at_window_end {
            self.known = self
                .known
                .drain()
                .filter(|(_k, v)| v.f + v.delta > b_current)
                .collect();
        }

        was_new
    }

    /// Query elements for which, with their guessed frequency `f`, it holds:
    ///
    /// ```text
    /// f >= (threshold - epsilon) * n
    /// f > epsilon * n
    /// ```
    ///
    /// The order of the elements is undefined.
    pub fn query<'a>(&'a self, threshold: f64) -> impl 'a + Iterator<Item = T> {
        let bound = (((threshold - self.epsilon) * (self.n as f64)).ceil()).max(0.) as usize;
        self.known
            .iter()
            .filter(move |(_k, v)| v.f >= bound)
            .map(|(k, _v)| k)
            .cloned()
    }

    /// Clear state of the counter.
    pub fn clear(&mut self) {
        self.known = HashMap::new();
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::LossyCounter;

    #[test]
    fn new_with_epsilon() {
        let counter = LossyCounter::<u64>::with_epsilon(0.5);
        assert_eq!(counter.epsilon(), 0.5);
        assert_eq!(counter.width(), 2);
        assert_eq!(counter.n(), 0);
    }

    #[test]
    fn new_with_width() {
        let counter = LossyCounter::<u64>::with_width(2);
        assert_eq!(counter.width(), 2);
        assert_eq!(counter.epsilon(), 0.5);
        assert_eq!(counter.n(), 0);
    }

    #[test]
    #[should_panic(expected = "epsilon (0) must be greater than 0 and smaller than 1")]
    fn new_panics_epsilon_0() {
        LossyCounter::<u64>::with_epsilon(0.);
    }

    #[test]
    #[should_panic(expected = "epsilon (1) must be greater than 0 and smaller than 1")]
    fn new_panics_epsilon_1() {
        LossyCounter::<u64>::with_epsilon(1.);
    }

    #[test]
    #[should_panic(expected = "width must be greater than 0")]
    fn new_panics_width_0() {
        LossyCounter::<u64>::with_width(0);
    }

    #[test]
    fn add() {
        let mut counter = LossyCounter::<u64>::with_epsilon(0.5);
        assert!(counter.add(13));
        assert_eq!(counter.n(), 1);
    }

    #[test]
    fn double_add() {
        let mut counter = LossyCounter::<u64>::with_epsilon(0.5);
        assert!(counter.add(13));
        assert!(!counter.add(13));
        assert_eq!(counter.n(), 2);
    }

    #[test]
    fn query_all() {
        let mut counter = LossyCounter::<u64>::with_epsilon(0.2);
        assert!(counter.add(13));
        assert!(counter.add(42));

        let mut data: Vec<u64> = counter.query(0.).collect();
        data.sort();
        assert_eq!(data, vec![13, 42]);
    }

    #[test]
    fn query_part() {
        let mut counter = LossyCounter::<u64>::with_epsilon(0.001);
        assert!(counter.add(13));
        assert!(!counter.add(13));
        assert!(counter.add(42));

        let mut data: Vec<u64> = counter.query(0.6).collect();
        data.sort();
        assert_eq!(data, vec![13]);
    }

    #[test]
    fn query_large() {
        let mut counter = LossyCounter::<u64>::with_epsilon(0.01);
        for i in 0..1000 {
            let j = i % 10;
            if j <= 6 {
                counter.add(i);
            } else {
                counter.add(j);
            }
        }

        let mut data: Vec<u64> = counter.query(0.02).collect();
        data.sort();
        assert_eq!(data, vec![7, 8, 9]);
    }

    #[test]
    fn clone() {
        let mut counter1 = LossyCounter::<u64>::with_epsilon(0.2);
        assert!(counter1.add(13));

        let mut counter2 = counter1.clone();
        assert!(counter2.add(42));

        assert_eq!(counter1.n(), 1);
        assert_eq!(counter2.n(), 2);

        let mut data1: Vec<u64> = counter1.query(0.).collect();
        data1.sort();
        assert_eq!(data1, vec![13]);

        let mut data2: Vec<u64> = counter2.query(0.).collect();
        data2.sort();
        assert_eq!(data2, vec![13, 42]);
    }

    #[test]
    fn clear() {
        let mut counter = LossyCounter::<u64>::with_epsilon(0.2);
        assert!(counter.add(13));
        assert_eq!(counter.n(), 1);

        counter.clear();
        assert_eq!(counter.n(), 0);
        let data: Vec<u64> = counter.query(0.).collect();
        assert!(data.is_empty());

        assert!(counter.add(13));
        assert_eq!(counter.n(), 1);
        let data: Vec<u64> = counter.query(0.).collect();
        assert_eq!(data, vec![13]);
    }

    #[test]
    fn pruning() {
        let mut counter = LossyCounter::<u64>::with_epsilon(0.5);

        assert!(counter.add(13));
        assert_eq!(counter.n(), 1);
        let data: Vec<u64> = counter.query(0.).collect();
        assert_eq!(data, vec![13]);

        assert!(counter.add(42));
        assert_eq!(counter.n(), 2);
        let data: Vec<u64> = counter.query(0.).collect();
        assert!(data.is_empty());
    }
}
