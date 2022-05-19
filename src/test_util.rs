use std::hash::Hash;

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
