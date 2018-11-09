#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use std::collections::HashSet;

use criterion::{Bencher, Criterion, Fun};
use pdatastructs::filters::bloomfilter::BloomFilter;
use pdatastructs::filters::cuckoofilter::CuckooFilter;
use pdatastructs::filters::quotientfilter::QuotientFilter;
use pdatastructs::filters::Filter;
use pdatastructs::rand::{ChaChaRng, SeedableRng};

fn setup_bloomfilter() -> BloomFilter<u64> {
    let false_positive_rate = 0.02; // = 2%
    let expected_elements = 1_000_000;
    BloomFilter::with_properties(expected_elements, false_positive_rate)
}

fn setup_cuckoofilter() -> CuckooFilter<u64, ChaChaRng> {
    let false_positive_rate = 0.02; // = 2%
    let expected_elements = 1_000_000;
    let rng = ChaChaRng::from_seed([0; 32]);
    CuckooFilter::with_properties_8(false_positive_rate, expected_elements, rng)
}

fn setup_hashset() -> HashSet<u64> {
    HashSet::new()
}

fn setup_quotientfilter() -> QuotientFilter<u64> {
    let bits_quotient = 24;
    let bits_remainder = 4;
    QuotientFilter::with_params(bits_quotient, bits_remainder)
}

fn run_setup<F, S>(setup: S, b: &mut Bencher)
where
    S: Fn() -> F,
    F: Filter<u64>,
{
    b.iter(setup)
}

fn run_insert_many<F, S>(setup: S, b: &mut Bencher, n: u64)
where
    S: Fn() -> F,
    F: Filter<u64>,
{
    b.iter_with_setup(setup, |mut filter| {
        for i in 0..n {
            filter.insert(&i).unwrap();
        }
    })
}

fn benchmarks_setup(c: &mut Criterion) {
    let functions = vec![
        Fun::new("bloomfilter", |b, _| run_setup(setup_bloomfilter, b)),
        Fun::new("cuckoofilter", |b, _| run_setup(setup_cuckoofilter, b)),
        Fun::new("hashset", |b, _| run_setup(setup_hashset, b)),
        Fun::new("quotientfilter", |b, _| run_setup(setup_quotientfilter, b)),
    ];
    c.bench_functions("setup", functions, ());
}

fn benchmarks_insert_many(c: &mut Criterion) {
    let functions = vec![
        Fun::new("bloomfilter", |b, n| {
            run_insert_many(setup_bloomfilter, b, *n)
        }),
        Fun::new("cuckoofilter", |b, n| {
            run_insert_many(setup_cuckoofilter, b, *n)
        }),
        Fun::new("hashset", |b, n| run_insert_many(setup_hashset, b, *n)),
        Fun::new("quotientfilter", |b, n| {
            run_insert_many(setup_quotientfilter, b, *n)
        }),
    ];
    c.bench_functions("insert_many", functions, 100);
}

criterion_group!(benches, benchmarks_setup, benchmarks_insert_many,);
criterion_main!(benches);
