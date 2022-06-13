# Changelog

## 0.7 --- Misc Improvements & Fixes

### All
- support `?Sized` where possible
- implement `Send` where possible

### Filters
- proper union support

### HyperLogLog
- fix potential out-of-bounds access (#74)
- improve performance

### T-Digest
- more scale functions

### Dependencies
- replace `Void` with `Infallible`
- update dependencies
- make all dependencies optional

## 0.6 --- T-Digest, Lossy Counter

### T-Digest
- new and shiny

### Filters
- improve documentation of BloomFilter

### Top-K
- rename former Top-K to CMS Heap
- add Lossy Counter implementation

### CountMinSketch
- `add` and `add_n` now return count after addition operation

### Dependencies
- update bytecount to 0.5

## 0.5 --- Performance, Filter Trait, Rust 2018

### Global
- benchmarking system
- improved hash performance
- various tiny documentation improvements
- misc performance improvements
- use stdlib `BuildHasherDefault` instead of own version
- make all containers typed and implement `AnyHash` for dynamically typed containers
- enforce formatting in CI
- Rust 2018

### Dependencies
- update bytecount to 0.4
- update rand to 0.6

### HyperLogLog
- implement `relative_error`
- enhanced bias correction
- extend value range for `b`

### Filters
- unified trait-based filter interface
- add QuotientFilter


## 0.4 --- CuckooFilter, API Improvements, Docs

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
