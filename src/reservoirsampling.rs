use rand::Rng;
use std::fmt;

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
    pub fn new(k: usize, rng: R) -> ReservoirSampling<T, R> {
        ReservoirSampling {
            k: k,
            rng: rng,
            reservoir: vec![],
            i: 0,
            skip_until: 0,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn reservoir(&self) -> &Vec<T> {
        &self.reservoir
    }

    pub fn i(&self) -> usize {
        self.i
    }

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

    pub fn is_empty(&self) -> bool {
        self.i == 0
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

#[cfg(test)]
mod tests {
    use super::ReservoirSampling;
    use rand;

    #[test]
    fn getter() {
        let rs = ReservoirSampling::<u64, rand::ThreadRng>::new(10, rand::thread_rng());
        assert_eq!(rs.k(), 10);
        assert_eq!(rs.i(), 0);
        assert_eq!(rs.reservoir(), &vec![]);
    }

    #[test]
    fn empty() {
        let rs = ReservoirSampling::<u64, rand::ThreadRng>::new(10, rand::thread_rng());
        assert!(rs.is_empty());
        assert!(rs.reservoir().is_empty());
    }

    #[test]
    fn add_k() {
        let mut rs = ReservoirSampling::<u64, rand::ThreadRng>::new(10, rand::thread_rng());
        for i in 0..10 {
            rs.add(i);
        }
        assert_eq!(rs.i(), 10);
        assert_eq!(rs.reservoir(), &vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn add_parts() {
        let mut rs = ReservoirSampling::<u64, rand::ThreadRng>::new(100, rand::thread_rng());
        for _ in 0..150 {
            rs.add(0);
        }
        for _ in 0..750 {
            rs.add(1);
        }
        for _ in 0..100 {
            rs.add(0);
        }
        assert!((rs.reservoir().iter().sum::<u64>() as i64 - 75).abs() < 5);
    }

    #[test]
    fn debug() {
        let rs = ReservoirSampling::<u64, rand::ThreadRng>::new(10, rand::thread_rng());
        assert_eq!(format!("{:?}", rs), "ReservoirSampling { k: 10 }");
    }
}
