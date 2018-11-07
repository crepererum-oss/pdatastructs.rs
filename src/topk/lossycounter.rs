//! LossyCounter implementation.

use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;

#[derive(Debug)]
struct KnownEntry {
    f: usize,
    delta: usize,
}

/// TODO
///
/// # References
/// - ["Approximate Frequency Counts over Data Streams", Gurmeet S. Manku, Rajeev Motwani, 2002](https://www.vldb.org/conf/2002/S10P03.pdf)
/// - [Wikipedia: Lossy Count Algorithm](https://en.wikipedia.org/wiki/Lossy_Count_Algorithm)
/// - ["Top K Frequent Items Algorithm", Zhipeng Jiang, 2017](https://zpjiang.me/2017/11/13/top-k-elementes-system-design/)
#[derive(Debug)]
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
    pub fn with_properties(epsilon: f64) -> Self {
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

    /// Epsilon error.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
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

        let b_current = self.n / self.width + 1;

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
        if self.n % self.width == 0 {
            // FIXME: nll
            let tmp: HashMap<T, KnownEntry> = self
                .known
                .drain()
                .filter(|(_k, v)| v.f + v.delta > b_current)
                .collect();
            self.known = tmp;
        }

        was_new
    }

    /// Query elements for which, with their guessed frequency `f`, it holds:
    ///
    /// ```text
    /// f >= (threshold - epsilon) * n
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
}

#[cfg(test)]
mod tests {
    use super::LossyCounter;

    #[test]
    fn new() {
        let counter = LossyCounter::<u64>::with_properties(0.5);
        assert_eq!(counter.epsilon(), 0.5);
        assert_eq!(counter.n(), 0);
    }

    #[test]
    #[should_panic(expected = "epsilon (0) must be greater than 0 and smaller than 1")]
    fn new_panics_epsilon_0() {
        LossyCounter::<u64>::with_properties(0.);
    }

    #[test]
    #[should_panic(expected = "epsilon (1) must be greater than 0 and smaller than 1")]
    fn new_panics_epsilon_1() {
        LossyCounter::<u64>::with_properties(1.);
    }

    #[test]
    fn add() {
        let mut counter = LossyCounter::<u64>::with_properties(0.5);
        assert!(counter.add(13));
        assert_eq!(counter.n(), 1);
    }

    #[test]
    fn double_add() {
        let mut counter = LossyCounter::<u64>::with_properties(0.5);
        assert!(counter.add(13));
        assert!(!counter.add(13));
        assert_eq!(counter.n(), 2);
    }

    #[test]
    fn query_all() {
        let mut counter = LossyCounter::<u64>::with_properties(0.2);
        assert!(counter.add(13));
        assert!(counter.add(42));

        let mut data: Vec<u64> = counter.query(0.).collect();
        data.sort();
        assert_eq!(data, vec![13, 42]);
    }

    #[test]
    fn query_part() {
        let mut counter = LossyCounter::<u64>::with_properties(0.001);
        assert!(counter.add(13));
        assert!(!counter.add(13));
        assert!(counter.add(42));

        let mut data: Vec<u64> = counter.query(0.6).collect();
        data.sort();
        assert_eq!(data, vec![13]);
    }

    #[test]
    fn query_large() {
        let mut counter = LossyCounter::<u64>::with_properties(0.01);
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
}
