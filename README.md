# pdatastructs

A collection of data structures that are based probability theory and therefore only provide the correct answer if a
certain probability. In exchange they have a better runtime and memory complexity compared to traditional data structures.

[![Build Status](https://github.com/crepererum/pdatastructs.rs/workflows/CI/badge.svg)](https://github.com/crepererum/pdatastructs.rs/actions?query=workflow%3ACI)
[![Crates.io](https://img.shields.io/crates/v/pdatastructs.svg)](https://crates.io/crates/pdatastructs)
[![Documentation](https://docs.rs/pdatastructs/badge.svg)](https://docs.rs/pdatastructs/)
[![License](https://img.shields.io/crates/l/pdatastructs.svg)](#license)

The following data structures are implemented:

- CountMinSketch
- Filters:
  - BloomFilter
  - CuckooFilter
  - QuotientFilter
- HyperLogLog
- ReservoirSampling
- T-Digest
- Top-K:
  - CMSHeap
  - LossyCounter

## License

Licensed under either of these:

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
 * MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

### Contributing

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in the work, as defined
in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
