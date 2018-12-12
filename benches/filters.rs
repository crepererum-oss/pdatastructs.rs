#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use std::collections::HashSet;

use criterion::{Bencher, Criterion, Fun, ParameterizedBenchmark};
use pdatastructs::filters::bloomfilter::BloomFilter;
use pdatastructs::filters::cuckoofilter::CuckooFilter;
use pdatastructs::filters::quotientfilter::QuotientFilter;
use pdatastructs::filters::Filter;
use pdatastructs::rand::SeedableRng;
use rand_chacha::ChaChaRng;

fn setup_bloomfilter() -> BloomFilter<u64> {
    let false_positive_rate = 0.02; // = 2%
    let expected_elements = 10_000;
    BloomFilter::with_properties(expected_elements, false_positive_rate)
}

fn setup_cuckoofilter() -> CuckooFilter<u64, ChaChaRng> {
    let false_positive_rate = 0.02; // = 2%
    let expected_elements = 10_000;
    let rng = ChaChaRng::from_seed([0; 32]);
    CuckooFilter::with_properties_8(false_positive_rate, expected_elements, rng)
}

fn setup_hashset() -> HashSet<u64> {
    HashSet::new()
}

fn setup_quotientfilter() -> QuotientFilter<u64> {
    let bits_quotient = 15;
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

fn run_query_single<F, S>(setup: S, b: &mut Bencher, n: u64)
where
    S: Fn() -> F,
    F: Filter<u64>,
{
    let setup_and_fill = || {
        let mut filter = setup();
        for i in 0..n {
            filter.insert(&i).unwrap();
        }
        filter
    };
    b.iter_with_setup(setup_and_fill, |filter| filter.query(&0))
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
    let parameters = vec![4_000, 8_000, 12_000, 16_000, 20_000];
    let benchmark = ParameterizedBenchmark::new(
        "bloomfilter",
        |b, n| run_insert_many(setup_bloomfilter, b, *n),
        parameters,
    )
    .with_function("cuckoofilter", |b, n| {
        run_insert_many(setup_cuckoofilter, b, *n)
    })
    .with_function("hashset", |b, n| run_insert_many(setup_hashset, b, *n))
    .with_function("quotientfilter", |b, n| {
        run_insert_many(setup_quotientfilter, b, *n)
    })
    .sample_size(20);
    c.bench("insert_many", benchmark);
}

fn benchmarks_query_single(c: &mut Criterion) {
    let parameters = vec![4_000, 8_000];
    let benchmark = ParameterizedBenchmark::new(
        "bloomfilter",
        |b, n| run_query_single(setup_bloomfilter, b, *n),
        parameters,
    )
    .with_function("cuckoofilter", |b, n| {
        run_query_single(setup_cuckoofilter, b, *n)
    })
    .with_function("hashset", |b, n| run_query_single(setup_hashset, b, *n))
    .with_function("quotientfilter", |b, n| {
        run_query_single(setup_quotientfilter, b, *n)
    })
    .sample_size(20);
    c.bench("query_single", benchmark);
}

criterion_group!(
    benches,
    benchmarks_setup,
    benchmarks_insert_many,
    benchmarks_query_single,
);
criterion_main!(benches);
