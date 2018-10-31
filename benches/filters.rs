#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use std::collections::HashSet;

use criterion::{Bencher, Criterion};
use pdatastructs::filters::bloomfilter::BloomFilter;
use pdatastructs::filters::cuckoofilter::CuckooFilter;
use pdatastructs::filters::quotientfilter::QuotientFilter;
use pdatastructs::filters::Filter;
use pdatastructs::rand::{ChaChaRng, SeedableRng};

fn run_insert_many<F>(mut filter: F, b: &mut Bencher)
where
    F: Filter<u64>,
{
    let mut obj: u64 = 0;

    b.iter(|| {
        filter.insert(&obj).unwrap();
        obj += 1;
    })
}

fn bloomfilter_insert_many(c: &mut Criterion) {
    c.bench_function("bloomfilter_insert_many", |b| {
        let false_positive_rate = 0.02; // = 2%
        let expected_elements = 1000 * 1000;
        let filter = BloomFilter::with_properties(expected_elements, false_positive_rate);

        run_insert_many(filter, b);
    });
}

fn cuckoofilter_insert_many(c: &mut Criterion) {
    c.bench_function("cuckoofilter_insert_many", |b| {
        let false_positive_rate = 0.02; // = 2%
        let expected_elements = 1000 * 1000;
        let rng = ChaChaRng::from_seed([0; 32]);
        let filter = CuckooFilter::with_properties_8(false_positive_rate, expected_elements, rng);

        run_insert_many(filter, b);
    });
}

fn hashset_insert_many(c: &mut Criterion) {
    c.bench_function("hashset_insert_many", |b| {
        let filter = HashSet::new();

        run_insert_many(filter, b);
    });
}

fn quotientfilter_insert_many(c: &mut Criterion) {
    c.bench_function("quotientfilter_insert_many", |b| {
        let bits_quotient = 24;
        let bits_remainder = 4;
        let filter = QuotientFilter::with_params(bits_quotient, bits_remainder);

        run_insert_many(filter, b);
    });
}

criterion_group!(
    benches,
    bloomfilter_insert_many,
    cuckoofilter_insert_many,
    hashset_insert_many,
    quotientfilter_insert_many
);
criterion_main!(benches);
