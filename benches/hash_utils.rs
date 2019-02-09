use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasherDefault;

use criterion::{criterion_group, criterion_main, Criterion};
use pdatastructs::hash_utils::HashIterBuilder;

fn hash_iter(c: &mut Criterion) {
    c.bench_function("hash_iter", |b| {
        let bh = BuildHasherDefault::<DefaultHasher>::default();
        let max_result = 42;
        let number_functions = 10_000;
        let builder = HashIterBuilder::new(max_result, number_functions, bh);
        let obj: u64 = 1337;
        b.iter(|| builder.iter_for(&obj).sum::<usize>())
    });
}

criterion_group!(benches, hash_iter,);
criterion_main!(benches);
