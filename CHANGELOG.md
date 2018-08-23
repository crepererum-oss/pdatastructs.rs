# Changelog

## 0.4 - CuckooFilter, API Improvements, Docs

- CountMinSketch: use `num-traits` as counters, `add_n` takes element by reference
- CuckooFilter: new!
- HyperLogLog: faster byte-count
- TopK: `.values()` -> `.iter()`
- all: tests for asserts, way nicer docs
- license: change to MIT+Apache2


## 0.3 --- Reservoir Sampling, Top-K, useful traits

- ReservoirSampling: first implementation
- TopK: first implementation
- all: implement Extend trait


## 0.2 --- CountMinSketch, rusty getters

- Counter: a safe counter trait
- CountMinSketch: first implementation
- BloomFilter, HyperLogLog: remove `get_` prefix from getters


## 0.1 --- BloomFilter, HyperLogLog

This is the very first release with the following data structures:

- BloomFilter
- HyperLogLog
