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

/// Inner data structure for tdigest to enable interior mutability.
#[derive(Debug)]
struct TDigestInner {
    centroids: Vec<Centroid>,
    min: f64,
    max: f64,
    compression_factor: f64,
    backlog: Vec<Centroid>,
    max_backlog_size: usize,
}

impl TDigestInner {
    fn new(compression_factor: f64, max_backlog_size: usize) -> Self {
        Self {
            centroids: vec![],
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            compression_factor,
            backlog: vec![],
            max_backlog_size,
        }
    }

    fn is_empty(&self) -> bool {
        self.centroids.is_empty() && self.backlog.is_empty()
    }

    fn insert_weighted(&mut self, x: f64, w: f64) {
        self.backlog.push(Centroid {
            count: w,
            sum: x * w,
        });

        self.min = self.min.min(x);
        self.max = self.max.max(x);

        if self.backlog.len() > self.max_backlog_size {
            self.merge();
        }
    }

    fn merge(&mut self) {
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

    #[inline(always)]
    fn interpolate(a: f64, b: f64, t: f64) -> f64 {
        debug_assert!((t >= 0.) && (t <= 1.));
        debug_assert!(a <= b);
        t * b + (1. - t) * a
    }

    fn quantile(&self, q: f64) -> f64 {
        // empty case
        if self.centroids.is_empty() {
            return f64::NAN;
        }

        let s: f64 = self.count();
        let limit = s * q;

        // left tail?
        let c_first = &self.centroids[0];
        if limit <= c_first.count * 0.5 {
            let t = limit / (0.5 * c_first.count);
            return Self::interpolate(self.min, c_first.mean(), t);
        }

        let mut cum = 0.;
        for (i, c) in self.centroids.iter().enumerate() {
            if cum + c.count * 0.5 >= limit {
                // default case
                debug_assert!(i > 0);
                let c_last = &self.centroids[i - 1];
                cum -= 0.5 * c_last.count;
                let delta = 0.5 * (c_last.count + c.count);
                let t = (limit - cum) / delta;
                return Self::interpolate(c_last.mean(), c.mean(), t);
            }
            cum += c.count;
        }

        // right tail
        let c_last = &self.centroids[self.centroids.len() - 1];
        cum -= 0.5 * c_last.count;
        let delta = s - 0.5 * c_last.count;
        let t = (limit - cum) / delta;
        return Self::interpolate(c_last.mean(), self.max, t);
    }

    fn count(&self) -> f64 {
        self.centroids.iter().map(|c| c.count).sum()
    }

    fn sum(&self) -> f64 {
        self.centroids.iter().map(|c| c.sum).sum()
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
        assert!(
            (compression_factor > 1.) && compression_factor.is_finite(),
            "compression_factor ({}) must be greater than 1 and finite",
            compression_factor
        );

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
        self.inner.borrow_mut().merge();

        self.inner.borrow().centroids.len()
    }

    /// TODO
    pub fn insert(&mut self, x: f64) {
        self.insert_weighted(x, 1.);
    }

    /// TODO
    pub fn insert_weighted(&mut self, x: f64, w: f64) {
        assert!(x.is_finite(), "x ({}) must be finite", x);
        assert!(
            (w >= 0.) && w.is_finite(),
            "w ({}) must be greater or equal than zero and finite",
            w
        );

        // early return for zero-weight
        if w == 0. {
            return;
        }

        self.inner.borrow_mut().insert_weighted(x, w)
    }

    /// TODO
    pub fn quantile(&self, q: f64) -> f64 {
        assert!((q >= 0.) && (q <= 1.), "q ({}) must be in [0, 1]", q);

        // apply compression if required
        self.inner.borrow_mut().merge();

        // get quantile on immutable state
        self.inner.borrow().quantile(q)
    }

    /// TODO
    pub fn count(&self) -> f64 {
        // apply compression if required
        self.inner.borrow_mut().merge();

        // get quantile on immutable state
        self.inner.borrow().count()
    }

    /// TODO
    pub fn sum(&self) -> f64 {
        // apply compression if required
        self.inner.borrow_mut().merge();

        // get quantile on immutable state
        self.inner.borrow().sum()
    }

    /// TODO
    pub fn mean(&self) -> f64 {
        // apply compression if required
        self.inner.borrow_mut().merge();

        let inner = self.inner.borrow();
        inner.sum() / inner.count()
    }

    /// TODO
    pub fn min(&self) -> f64 {
        self.inner.borrow().min
    }

    /// TODO
    pub fn max(&self) -> f64 {
        self.inner.borrow().max
    }
}

#[cfg(test)]
mod tests {
    use super::{k1, k1_inv, TDigest};
    use rand::distributions::StandardNormal;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaChaRng;
    use std::f64;

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
        assert_eq!(digest.count(), 0.);
        assert_eq!(digest.sum(), 0.);
        assert!(digest.mean().is_nan());
        assert!(digest.min().is_infinite() && digest.min().is_sign_positive());
        assert!(digest.max().is_infinite() && digest.max().is_sign_negative());
    }

    #[test]
    #[should_panic(expected = "compression_factor (1) must be greater than 1 and finite")]
    fn new_panics_compression_factor_a() {
        let compression_factor = 1.;
        let max_backlog_size = 13;
        TDigest::new(compression_factor, max_backlog_size);
    }

    #[test]
    #[should_panic(expected = "compression_factor (inf) must be greater than 1 and finite")]
    fn new_panics_compression_factor_b() {
        let compression_factor = f64::INFINITY;
        let max_backlog_size = 13;
        TDigest::new(compression_factor, max_backlog_size);
    }

    #[test]
    fn with_normal_distribution() {
        let compression_factor = 100.;
        let max_backlog_size = 10;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);

        let mut rng = ChaChaRng::from_seed([0; 32]);

        let n = 100_000;
        let mut s = 0.;
        let mut a = f64::INFINITY;
        let mut b = f64::NEG_INFINITY;
        for _ in 0..n {
            let x = rng.sample(StandardNormal);
            digest.insert(x);
            s += x;
            a = a.min(x);
            b = b.max(x);
        }

        // generic tests
        assert_eq!(digest.count(), n as f64);
        assert!((s - digest.sum()).abs() < 0.0001);
        assert!((s / (n as f64) - digest.mean()).abs() < 0.0001);
        assert_eq!(digest.min(), a);
        assert_eq!(digest.max(), b);

        // compression works
        assert!(digest.n_centroids() < 100);

        // test some known quantiles
        assert!((-1.2816 - digest.quantile(0.10)).abs() < 0.01);
        assert!((-0.6745 - digest.quantile(0.25)).abs() < 0.01);
        assert!((0.0000 - digest.quantile(0.5)).abs() < 0.01);
        assert!((0.6745 - digest.quantile(0.75)).abs() < 0.01);
        assert!((1.2816 - digest.quantile(0.90)).abs() < 0.01);
    }

    #[test]
    fn with_single() {
        let compression_factor = 100.;
        let max_backlog_size = 10;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);

        digest.insert(13.37);

        // generic tests
        assert_eq!(digest.count(), 1.);
        assert_eq!(digest.sum(), 13.37);
        assert_eq!(digest.mean(), 13.37);
        assert_eq!(digest.min(), 13.37);
        assert_eq!(digest.max(), 13.37);

        // compression works
        assert_eq!(digest.n_centroids(), 1);

        // test some known quantiles
        assert_eq!(digest.quantile(0.), 13.37);
        assert_eq!(digest.quantile(0.5), 13.37);
        assert_eq!(digest.quantile(1.), 13.37);
    }

    #[test]
    fn with_two_symmetric() {
        let compression_factor = 100.;
        let max_backlog_size = 10;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);

        digest.insert(10.);
        digest.insert(20.);

        // generic tests
        assert_eq!(digest.count(), 2.);
        assert_eq!(digest.sum(), 30.);
        assert_eq!(digest.mean(), 15.);
        assert_eq!(digest.min(), 10.);
        assert_eq!(digest.max(), 20.);

        // compression works
        assert_eq!(digest.n_centroids(), 2);

        // test some known quantiles
        assert_eq!(digest.quantile(0.), 10.); // min
        assert_eq!(digest.quantile(0.25), 10.); // first centroid
        assert_eq!(digest.quantile(0.375), 12.5);
        assert_eq!(digest.quantile(0.5), 15.); // center
        assert_eq!(digest.quantile(0.625), 17.5);
        assert_eq!(digest.quantile(0.75), 20.); // second centroid
        assert_eq!(digest.quantile(1.), 20.); // max
    }

    #[test]
    fn with_two_assymmetric() {
        let compression_factor = 100.;
        let max_backlog_size = 10;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);

        digest.insert_weighted(10., 1.);
        digest.insert_weighted(20., 9.);

        // generic tests
        assert_eq!(digest.count(), 10.);
        assert_eq!(digest.sum(), 190.);
        assert_eq!(digest.mean(), 19.);
        assert_eq!(digest.min(), 10.);
        assert_eq!(digest.max(), 20.);

        // compression works
        assert_eq!(digest.n_centroids(), 2);

        // test some known quantiles
        assert_eq!(digest.quantile(0.), 10.); // min
        assert_eq!(digest.quantile(0.05), 10.); // first centroid
        assert_eq!(digest.quantile(0.175), 12.5);
        assert_eq!(digest.quantile(0.3), 15.); // center
        assert_eq!(digest.quantile(0.425), 17.5);
        assert_eq!(digest.quantile(0.55), 20.); // second centroid
        assert_eq!(digest.quantile(1.), 20.); // max
    }

    #[test]
    fn zero_weight() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);

        digest.insert_weighted(13.37, 0.);

        assert_eq!(digest.compression_factor(), 2.);
        assert_eq!(digest.max_backlog_size(), 13);
        assert_eq!(digest.n_centroids(), 0);
        assert!(digest.is_empty());
        assert!(digest.quantile(0.5).is_nan());
        assert_eq!(digest.count(), 0.);
        assert_eq!(digest.sum(), 0.);
        assert!(digest.mean().is_nan());
        assert!(digest.min().is_infinite() && digest.min().is_sign_positive());
        assert!(digest.max().is_infinite() && digest.max().is_sign_negative());
    }

    #[test]
    fn highly_compressed() {
        let compression_factor = 2.;
        let max_backlog_size = 10;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);

        digest.insert(10.);
        digest.insert(20.);
        for _ in 0..100 {
            digest.insert(15.);
        }

        // generic tests
        assert_eq!(digest.count(), 102.);
        assert_eq!(digest.sum(), 1530.);
        assert_eq!(digest.mean(), 15.);
        assert_eq!(digest.min(), 10.);
        assert_eq!(digest.max(), 20.);

        // compression works
        assert_eq!(digest.n_centroids(), 1);

        // test some known quantiles
        assert_eq!(digest.quantile(0.), 10.); // min
        assert_eq!(digest.quantile(0.125), 11.25); // tail
        assert_eq!(digest.quantile(0.25), 12.5); // tail
        assert_eq!(digest.quantile(0.5), 15.); // center (single centroid)
        assert_eq!(digest.quantile(0.75), 17.5); // tail
        assert_eq!(digest.quantile(0.875), 18.75); // tail
        assert_eq!(digest.quantile(1.), 20.); // max
    }

    #[test]
    #[should_panic(expected = "q (-1) must be in [0, 1]")]
    fn invalid_q_a() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let digest = TDigest::new(compression_factor, max_backlog_size);
        digest.quantile(-1.);
    }

    #[test]
    #[should_panic(expected = "q (2) must be in [0, 1]")]
    fn invalid_q_b() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let digest = TDigest::new(compression_factor, max_backlog_size);
        digest.quantile(2.);
    }

    #[test]
    #[should_panic(expected = "q (NaN) must be in [0, 1]")]
    fn invalid_q_c() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let digest = TDigest::new(compression_factor, max_backlog_size);
        digest.quantile(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "q (inf) must be in [0, 1]")]
    fn invalid_q_d() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let digest = TDigest::new(compression_factor, max_backlog_size);
        digest.quantile(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "w (-1) must be greater or equal than zero and finite")]
    fn invalid_w_a() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);
        digest.insert_weighted(13.37, -1.);
    }

    #[test]
    #[should_panic(expected = "w (inf) must be greater or equal than zero and finite")]
    fn invalid_w_b() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);
        digest.insert_weighted(13.37, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "x (-inf) must be finite")]
    fn invalid_x_a() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);
        digest.insert(f64::NEG_INFINITY);
    }

    #[test]
    #[should_panic(expected = "x (inf) must be finite")]
    fn invalid_x_b() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);
        digest.insert(f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "x (NaN) must be finite")]
    fn invalid_x_c() {
        let compression_factor = 2.;
        let max_backlog_size = 13;
        let mut digest = TDigest::new(compression_factor, max_backlog_size);
        digest.insert(f64::NAN);
    }
}
