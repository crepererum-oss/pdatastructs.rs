#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use criterion::Criterion;
use pdatastructs::bloomfilter::BloomFilter;

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

criterion_group!(benches, bloomfilter_add_single);
criterion_main!(benches);
