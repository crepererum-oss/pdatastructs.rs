#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use criterion::Criterion;
use pdatastructs::filters::bloomfilter::BloomFilter;
use pdatastructs::filters::cuckoofilter::CuckooFilter;
use pdatastructs::filters::Filter;
use pdatastructs::rand::{ChaChaRng, SeedableRng};

fn bloomfilter_insert_single(c: &mut Criterion) {
    c.bench_function("bloomfilter_insert_single", |b| {
        let false_positive_rate = 0.02; // = 2%
        let expected_elements = 1000;
        let mut filter = BloomFilter::with_properties(expected_elements, false_positive_rate);
        let obj = "foo bar";

        b.iter(|| {
            filter.insert(&obj).unwrap();
        })
    });
}

fn cuckoofilter_insert_many(c: &mut Criterion) {
    c.bench_function("cuckoofilter_insert_many", |b| {
        let false_positive_rate = 0.02; // = 2%
        let expected_elements = 1000 * 1000;
        let rng = ChaChaRng::from_seed([0; 32]);
        let mut filter =
            CuckooFilter::with_properties_8(false_positive_rate, expected_elements, rng);
        let mut obj: u64 = 0;

        b.iter(|| {
            filter.insert(&obj).unwrap();
            obj += 1;
        })
    });
}

criterion_group!(benches, bloomfilter_insert_single, cuckoofilter_insert_many);
criterion_main!(benches);
