use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    convert::Infallible,
};

use pdatastructs::filters::Filter;
use sbbf_rs::{FilterFn, ALIGNMENT, BUCKET_SIZE};

pub struct Sbbf {
    filter_fn: FilterFn,
    buf: Buf,
    num_buckets: usize,
}

impl Sbbf {
    pub fn new(bits_per_key: usize, num_keys: usize) -> Self {
        let len = (bits_per_key / 8) * num_keys;
        let len = ((len + BUCKET_SIZE / 2) / BUCKET_SIZE) * BUCKET_SIZE;
        Self {
            filter_fn: FilterFn::new(),
            buf: Buf::new(len),
            num_buckets: len / BUCKET_SIZE,
        }
    }

    pub fn contains(&self, hash: u64) -> bool {
        unsafe {
            self.filter_fn
                .contains(self.buf.ptr, self.num_buckets, hash)
        }
    }

    pub fn insert(&mut self, hash: u64) {
        unsafe { self.filter_fn.insert(self.buf.ptr, self.num_buckets, hash) }
    }
}

struct Buf {
    ptr: *mut u8,
    layout: Layout,
}

impl Buf {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len, ALIGNMENT).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };

        Self { layout, ptr }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

impl Filter<u64> for Sbbf {
    type InsertErr = Infallible;

    fn clear(&mut self) {
        for i in 0..self.buf.layout.size() {
            unsafe { *self.buf.ptr.add(i) = 0 };
        }
    }

    fn insert(&mut self, obj: &u64) -> Result<bool, Self::InsertErr> {
        unsafe {
            let found = self
                .filter_fn
                .contains(self.buf.ptr, self.num_buckets, *obj);
            self.filter_fn.insert(self.buf.ptr, self.num_buckets, *obj);
            Ok(found)
        }
    }

    fn union(&mut self, other: &Self) -> Result<(), Self::InsertErr> {
        assert_eq!(self.buf.layout.size(), other.buf.layout.size());

        for i in 0..self.buf.layout.size() {
            unsafe { *self.buf.ptr.add(i) |= *other.buf.ptr.add(i) };
        }

        Ok(())
    }

    fn is_empty(&self) -> bool {
        for i in 0..self.buf.layout.size() {
            unsafe {
                if *self.buf.ptr.add(i) != 0 {
                    return false;
                }
            };
        }

        true
    }

    fn len(&self) -> usize {
        0
    }

    fn query(&self, obj: &u64) -> bool {
        unsafe {
            self.filter_fn
                .contains(self.buf.ptr, self.num_buckets, *obj)
        }
    }
}
