#[macro_use]
extern crate criterion;
extern crate pdatastructs;

use criterion::Criterion;
use pdatastructs::hyperloglog::HyperLogLog;

fn hyperloglog_add_single(c: &mut Criterion) {
    c.bench_function("hyperloglog_add_single", |b| {
        let address_bits = 4; // so we store 2^4 = 16 registers in total
        let mut hll = HyperLogLog::new(address_bits);
        let obj = "foo bar";

        b.iter(|| {
            hll.add(&obj);
        })
    });
}

fn hyperloglog_count_empty(c: &mut Criterion) {
    c.bench_function("hyperloglog_count_empty", |b| {
        let address_bits = 4; // so we store 2^4 = 16 registers in total
        let hll = HyperLogLog::new(address_bits);

        b.iter(|| {
            hll.count();
        })
    });
}

criterion_group!(benches, hyperloglog_add_single, hyperloglog_count_empty);
criterion_main!(benches);
