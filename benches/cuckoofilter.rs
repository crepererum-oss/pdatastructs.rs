#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use criterion::Criterion;
use pdatastructs::cuckoofilter::CuckooFilter;
use pdatastructs::rand::{ChaChaRng, SeedableRng};

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

criterion_group!(benches, cuckoofilter_add_many);
criterion_main!(benches);
