//! TODO
use std::cell::RefCell;
use std::f64;

#[derive(Clone, Debug)]
struct Centroid {
    sum: f64,
    count: f64,
}

impl Centroid {
    fn fuse(&self, other: &Centroid) -> Self {
        Self {
            count: self.count + other.count,
            sum: self.sum + other.sum,
        }
    }

    fn mean(&self) -> f64 {
        self.sum / self.count
    }
}

fn k1(q: f64, delta: f64) -> f64 {
    delta / (2. * f64::consts::PI) * (2. * q - 1.).asin()
}

fn k1_inv(k: f64, delta: f64) -> f64 {
    ((k * 2. * f64::consts::PI / delta).sin() + 1.) / 2.
}

/// Inner data structure for tdigest to enable inner mutablity.
#[derive(Debug)]
struct TDigestInner {
    centroids: Vec<Centroid>,
    compression_factor: f64,
    backlog: Vec<Centroid>,
    max_backlog_size: usize,
}

impl TDigestInner {
    fn new(compression_factor: f64, max_backlog_size: usize) -> Self {
        Self {
            centroids: vec![],
            compression_factor,
            backlog: vec![],
            max_backlog_size,
        }
    }

    fn is_empty(&self) -> bool {
        self.centroids.is_empty() && self.backlog.is_empty()
    }

    fn insert_weighted(&mut self, x: f64, w: f64) {
        self.backlog.push(Centroid { count: w, sum: x });

        if self.backlog.len() > self.max_backlog_size {
            self.compress();
        }
    }

    fn compress(&mut self) {
        // early return in case nothing to be done
        if self.backlog.is_empty() {
            return;
        }

        // TODO: use sort_by_cached_key once stable
        let mut x: Vec<(f64, Centroid)> = self
            .centroids
            .drain(..)
            .chain(self.backlog.drain(..))
            .map(|c| (c.mean(), c))
            .collect();
        x.sort_by(|t1, t2| t1.0.partial_cmp(&t2.0).unwrap());
        let mut x: Vec<Centroid> = x.drain(..).map(|t| t.1).collect();

        let s: f64 = x.iter().map(|c| c.count).sum();

        let mut q_0 = 0.;
        let mut q_limit = k1_inv(
            k1(q_0, self.compression_factor) + 1.,
            self.compression_factor,
        );

        let mut result = vec![];
        let mut current = x[0].clone();
        for next in x.drain(1..) {
            let q = q_0 + (current.count + next.count) / s;
            if q <= q_limit {
                current = current.fuse(&next);
            } else {
                q_0 += current.count / s;
                q_limit = k1_inv(
                    k1(q_0, self.compression_factor) + 1.,
                    self.compression_factor,
                );
                result.push(current);
                current = next;
            }
        }
        result.push(current);

        self.centroids = result;
    }

    fn quantile(&self, q: f64) -> f64 {
        // empty case
        if self.centroids.is_empty() {
            return f64::NAN;
        }

        let s: f64 = self.centroids.iter().map(|c| c.count).sum();
        let limit = s * q;

        // left tail?
        let c_first = &self.centroids[0];
        if limit < c_first.count {
            // TODO: interpolate w/ min
            return c_first.mean();
        }

        let mut cum = 0.;
        let mut idx = self.centroids.len() - 1;
        for (i, c) in self.centroids.iter().enumerate() {
            cum += c.count;
            if cum >= limit {
                idx = i;
                break;
            }
        }

        // right tail?
        if idx == self.centroids.len() - 1 {
            let c_last = &self.centroids[self.centroids.len() - 1];
            // TODO: interpolate w/ max
            return c_last.mean();
        }

        // default case
        let c_a = &self.centroids[idx];
        let c_b = &self.centroids[idx + 1];
        let t = (limit - cum) / c_b.count;
        t * c_b.mean() + (1. - t) * c_a.mean()
    }
}

/// TODO
///
/// https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf
#[derive(Debug)]
pub struct TDigest {
    inner: RefCell<TDigestInner>,
}

impl TDigest {
    /// TODO
    pub fn new(compression_factor: f64, max_backlog_size: usize) -> Self {
        // TODO assert compression_factor

        Self {
            inner: RefCell::new(TDigestInner::new(compression_factor, max_backlog_size)),
        }
    }

    /// TODO
    pub fn compression_factor(&self) -> f64 {
        self.inner.borrow().compression_factor
    }

    /// TODO
    pub fn max_backlog_size(&self) -> usize {
        self.inner.borrow().max_backlog_size
    }

    /// Check whether the digest has not received any positives weights yet.
    pub fn is_empty(&self) -> bool {
        self.inner.borrow().is_empty()
    }

    /// Get number of centroids tracked by the digest.
    pub fn n_centroids(&self) -> usize {
        // first apply compression
        self.inner.borrow_mut().compress();

        self.inner.borrow().centroids.len()
    }

    /// TODO
    pub fn insert(&mut self, x: f64) {
        self.insert_weighted(x, 1.);
    }

    /// TODO
    pub fn insert_weighted(&mut self, x: f64, w: f64) {
        // TODO: assert w and early exit for 0
        // TODO: check x for non-NaN
        // TODO: check w for non-NaN

        self.inner.borrow_mut().insert_weighted(x, w)
    }

    /// TODO
    pub fn quantile(&self, q: f64) -> f64 {
        // TODO: check q

        // apply compression if required
        self.inner.borrow_mut().compress();

        // get quantile on immutable state
        self.inner.borrow().quantile(q)
    }
}

#[cfg(test)]
mod tests {
    use super::{k1, k1_inv, TDigest};
    use rand::distributions::StandardNormal;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;

    #[test]
    fn k_test1() {
        let delta = 0.5;
        let q = 0.5;

        let k = k1(q, delta);
        assert_eq!(k, 0.);

        let q2 = k1_inv(k, delta);
        assert_eq!(q2, q);
    }

    #[test]
    fn k_test2() {
        let delta = 0.7;
        let q = 0.5;

        let k = k1(q, delta);
        assert_eq!(k, 0.);

        let q2 = k1_inv(k, delta);
        assert_eq!(q2, q);
    }

    #[test]
    fn k_test3() {
        let delta = 0.7;
        let q = 0.7;

        let k = k1(q, delta);
        assert_ne!(k, 0.);

        let q2 = k1_inv(k, delta);
        assert_eq!(q2, q);
    }

    #[test]
    fn new_empty() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let digest = TDigest::new(compression_factor, max_backlog_size);

        assert_eq!(digest.compression_factor(), 2.);
        assert_eq!(digest.max_backlog_size(), 13);
        assert_eq!(digest.n_centroids(), 0);
        assert!(digest.is_empty());
        assert!(digest.quantile(0.5).is_nan());
    }

    #[test]
    fn with_normal_distribution() {
        let compression_factor = 100.;
        let max_backlog_size = 10;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);

        let mut rng = ChaChaRng::from_seed([0; 32]);

        let n = 100_000;
        for _ in 0..n {
            let x = rng.sample(StandardNormal);
            digest.insert(x);
        }

        // compression works
        assert!(digest.n_centroids() < 100);

        // test some known quantiles
        assert!((-1.2816 - digest.quantile(0.10)).abs() < 0.05);
        assert!((-0.6745 - digest.quantile(0.25)).abs() < 0.05);
        assert!((0.0000 - digest.quantile(0.5)).abs() < 0.05);
        assert!((0.6745 - digest.quantile(0.75)).abs() < 0.05);
        assert!((1.2816 - digest.quantile(0.90)).abs() < 0.05);
    }
}
