#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use criterion::Criterion;
use pdatastructs::filters::bloomfilter::BloomFilter;
use pdatastructs::filters::cuckoofilter::CuckooFilter;
use pdatastructs::rand::{ChaChaRng, SeedableRng};

fn bloomfilter_add_single(c: &mut Criterion) {
    c.bench_function("bloomfilter_add_single", |b| {
        let false_positive_rate = 0.02; // = 2%
        let expected_elements = 1000;
        let mut filter = BloomFilter::with_properties(expected_elements, false_positive_rate);
        let obj = "foo bar";

        b.iter(|| {
            filter.add(&obj);
        })
    });
}

fn cuckoofilter_add_many(c: &mut Criterion) {
    c.bench_function("cuckoofilter_add_many", |b| {
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

criterion_group!(benches, bloomfilter_add_single, cuckoofilter_add_many);
criterion_main!(benches);
