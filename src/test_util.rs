use std::hash::Hash;

use chacha20::ChaCha20Rng;
use rand::{SeedableRng, TryRng};

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub(crate) struct NotSend {
    _marker: *const u8,
}

impl Default for NotSend {
    fn default() -> Self {
        Self {
            _marker: std::ptr::null(),
        }
    }
}

pub(crate) fn assert_send<T>(_: &T)
where
    T: Send,
{
}

pub(crate) struct CloneableChacha20Rng {
    inner: ChaCha20Rng,
}

impl TryRng for CloneableChacha20Rng {
    type Error = <ChaCha20Rng as TryRng>::Error;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        self.inner.try_next_u32()
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        self.inner.try_next_u64()
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.inner.try_fill_bytes(dst)
    }
}

impl SeedableRng for CloneableChacha20Rng {
    type Seed = <ChaCha20Rng as SeedableRng>::Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            inner: ChaCha20Rng::from_seed(seed),
        }
    }
}

impl Clone for CloneableChacha20Rng {
    fn clone(&self) -> Self {
        Self {
            inner: ChaCha20Rng::deserialize_state(&self.inner.serialize_state()),
        }
    }
}
