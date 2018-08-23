//! ReservoirSampling implementation.
use rand::Rng;
use std::fmt;

/// A ReservoirSampling is a data-structure that samples a fixed number of elements from a data
/// stream, without the need to memorize any additional elements.
///
/// # Examples
/// ```
/// use pdatastructs::reservoirsampling::ReservoirSampling;
/// use pdatastructs::rand::{ChaChaRng, SeedableRng};
///
/// // set up sampler for 10 elements
/// let rng = ChaChaRng::from_seed([0; 32]);
/// let mut sampler = ReservoirSampling::new(10, rng);
///
/// // add some data
/// for i in 0..1000 {
///     let x = i % 2;
///     sampler.add(x);
/// }
///
/// // later
/// assert_eq!(sampler.reservoir().iter().sum::<i64>(), 5);
/// ```
///
/// # Applications
/// - Getting a set of representative samples of a large data set.
/// - Collecting distribution information of a stream of data, which can then recover approximative
///   histograms or CDFs (Cumulative Distribution Function).
///
/// # How It Works
///
/// ## Setup
/// The sampler is initialized with an empty reservoir vector and a potential capacity `k`.
///
/// ## Insertion
/// If the sampler did see less than `k` elements, the element is appended to the reservoir vector,
/// so no sampling takes place (full take).
///
/// After the reservoir is filled up with `k` elements, the "normal reservoir sampling" phase
/// starts until `t = k * 4` elements were processed. In this phase, every incoming element
/// replaces may replace a random, existing element in the reservoir, iff a random number `x in [1,
/// i]` (i being the number of seen elements) is smaller or equal to `k`.
///
/// In the final phase, not every new incoming element may processed. Instead, some elements are
/// skipped. The number of skipped elements is chosen randomly but with respect to the number of
/// seen elements. Every non-skipped element replaces a random, existing element of the reservoir.
///
/// ## Read-Out
/// The reservoir vector is just returned, no further calculation is required.
///
/// # See Also
/// - `std::vec::Vec`: can be used to get an exact sampling but needs to store all elements of the
///   incoming data stream
///
/// # References
/// - ["Very Fast Reservoir Sampling", Erik Erlandson, 2015](https://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/)
/// - [Wikipedia: Reservoir Sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)
#[derive(Clone)]
pub struct ReservoirSampling<T, R>
where
    R: Rng,
{
    k: usize,
    rng: R,
    reservoir: Vec<T>,
    i: usize,
    skip_until: usize,
}

impl<T, R> ReservoirSampling<T, R>
where
    R: Rng,
{
    /// Create new reservoir sampler that keeps `k` samples and uses `rng` for its random
    /// decisions.
    pub fn new(k: usize, rng: R) -> Self {
        assert!(k > 0, "k must be greater than 0");

        Self {
            k,
            rng,
            reservoir: vec![],
            i: 0,
            skip_until: 0,
        }
    }

    /// Number of samples that should be kept.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Read-only copy of the reservoir. Contains at most `k` entries.
    pub fn reservoir(&self) -> &Vec<T> {
        &self.reservoir
    }

    /// Number of data points seen.
    pub fn i(&self) -> usize {
        self.i
    }

    /// Observe new data point.
    pub fn add(&mut self, obj: T) {
        let t = self.k * 4; // TODO: make this a parameter

        if self.i < self.k {
            // initial fill-up
            self.reservoir.push(obj)
        } else if self.i < t {
            // normal reservoir sampling
            let j = self.rng.gen_range::<usize>(0, self.i);
            if j < self.k {
                self.reservoir[j] = obj;
            }
        } else if self.i >= self.skip_until {
            // fast skipping approximation
            let j = self.rng.gen_range::<usize>(0, self.k);
            self.reservoir[j] = obj;

            // calculate next skip
            let p = (self.k as f64) / ((self.i + 1) as f64);
            let u = 1. - self.rng.gen_range::<f64>(0., 1.); // (0.0, 1.0]
            let g = (u.ln() / (1. - p).ln()).floor() as usize;
            self.skip_until = self.i + g;
        }

        self.i += 1;
    }

    /// Checks if reservoir is empty (i.e. no data points where observed)
    pub fn is_empty(&self) -> bool {
        self.i == 0
    }

    /// Clear sampler state. It now behaves like a fresh one (i.e. no data points seen so far).
    pub fn clear(&mut self) {
        self.i = 0;
        self.skip_until = 0;
        self.reservoir.clear();
    }
}

impl<T, R> fmt::Debug for ReservoirSampling<T, R>
where
    R: Rng,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ReservoirSampling {{ k: {} }}", self.k)
    }
}

impl<T, R> Extend<T> for ReservoirSampling<T, R>
where
    R: Rng,
{
    fn extend<S: IntoIterator<Item = T>>(&mut self, iter: S) {
        for elem in iter {
            self.add(elem);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReservoirSampling;
    use rand::{ChaChaRng, SeedableRng};

    #[test]
    #[should_panic(expected = "k must be greater than 0")]
    fn new_panics_k0() {
        ReservoirSampling::<u64, ChaChaRng>::new(0, ChaChaRng::from_seed([0; 32]));
    }

    #[test]
    fn getter() {
        let rs = ReservoirSampling::<u64, ChaChaRng>::new(10, ChaChaRng::from_seed([0; 32]));
        assert_eq!(rs.k(), 10);
        assert_eq!(rs.i(), 0);
        assert_eq!(rs.reservoir(), &vec![]);
    }

    #[test]
    fn empty() {
        let rs = ReservoirSampling::<u64, ChaChaRng>::new(10, ChaChaRng::from_seed([0; 32]));
        assert!(rs.is_empty());
        assert!(rs.reservoir().is_empty());
    }

    #[test]
    fn add_k() {
        let mut rs = ReservoirSampling::<u64, ChaChaRng>::new(10, ChaChaRng::from_seed([0; 32]));
        for i in 0..10 {
            rs.add(i);
        }
        assert_eq!(rs.i(), 10);
        assert_eq!(rs.reservoir(), &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn add_parts() {
        let mut rs = ReservoirSampling::<u64, ChaChaRng>::new(100, ChaChaRng::from_seed([0; 32]));
        for _ in 0..1500 {
            rs.add(0);
        }
        for _ in 0..7500 {
            rs.add(1);
        }
        for _ in 0..1000 {
            rs.add(0);
        }
        assert!((rs.reservoir().iter().sum::<u64>() as i64 - 75).abs() < 5);
    }

    #[test]
    fn debug() {
        let rs = ReservoirSampling::<u64, ChaChaRng>::new(10, ChaChaRng::from_seed([0; 32]));
        assert_eq!(format!("{:?}", rs), "ReservoirSampling { k: 10 }");
    }

    #[test]
    fn clear() {
        let mut rs = ReservoirSampling::<u64, ChaChaRng>::new(10, ChaChaRng::from_seed([0; 32]));
        for i in 0..10 {
            rs.add(i);
        }
        rs.clear();
        for i in 10..20 {
            rs.add(i);
        }
        assert_eq!(rs.i(), 10);
        assert_eq!(
            rs.reservoir(),
            &vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );
    }

    #[test]
    fn extend() {
        let mut rs = ReservoirSampling::<u64, ChaChaRng>::new(10, ChaChaRng::from_seed([0; 32]));
        rs.extend(0..10);
        assert_eq!(rs.i(), 10);
        assert_eq!(rs.reservoir(), &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn clone() {
        let mut rs1 = ReservoirSampling::<u64, ChaChaRng>::new(10, ChaChaRng::from_seed([0; 32]));
        for i in 0..10 {
            rs1.add(i);
        }
        let mut rs2 = rs1.clone();
        for _ in 0..1000 {
            rs2.add(99);
        }
        assert_eq!(rs1.i(), 10);
        assert_eq!(rs2.i(), 1010);
        assert_eq!(rs1.reservoir(), &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            rs2.reservoir(),
            &vec![99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
        );
    }

    #[test]
    fn k1() {
        let mut rs = ReservoirSampling::<u64, ChaChaRng>::new(1, ChaChaRng::from_seed([0; 32]));
        for _ in 0..10 {
            rs.add(1);
        }
        assert_eq!(rs.reservoir(), &vec![1]);
    }
}
