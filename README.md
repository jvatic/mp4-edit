mp4-edit
========

[![Build Status]][GitHub Actions]
[![Crate]][crates.io]
[![Documentation]][docs.rs]
[![Dependency Status]][deps.rs]
[![Downloads]][crates.io]
[![License]](LICENSE-MIT)

This crate provides tools for lossless editing of MP4 files, with a focus on audiobooks.

## Status

> [!WARNING]
> Unstable. The API is likely to change, and more testing is needed.

## Why yet another mp4 parser?

The short answer is I needed one that could handle non-standard mp4s plus some other features I didn't see in existing crates, and this evolved out of my learning the file format. And why not? It's been fun! (See below for a list of alternatives.)

## Usage

See the examples dir for in-depth usage.

## Highlights

- Lossless MP4 editing[^lossless].
- Duration trimming/slicing[^trimming] (WIP, requires `experimental-trim` feature flag).
- Track interleaving[^interleaving].
- Chapter track builder.
- Easy fast start (it's possible in most cases to insert metadata before `mdat` in a single pass).
- Async API built on `futures` traits[^async] (`tokio` support via `tokio_util::compat`).

## Examples

You may run the examples in this repo as usual with `cargo run --example <EXAMPLE>`, but for most examples you'll want to build them first with `cargo build --example <EXAMPLE> --release` before running them to get good performance.

- **mp4copy**: Makes a copy of the mp4 file, converting it to fast-start if it isn't already.
- **mp4dump**: Useful for seeing what's inside an mp4 file.
- **extract_leaf_atoms**: Generate atom parser test data from an mp4 file.

## Alternatives

Here are some other mp4 crates to consider (in alphabetical order):

- [mp4-atom](https://github.com/kixelated/mp4-atom)
- [mp4-rust](https://github.com/alfg/mp4-rust)
- [mp4ameta](https://github.com/saecki/mp4ameta)
- [mp4parse-rust](https://github.com/mozilla/mp4parse-rust)
- [mtag](https://github.com/insomnimus/mtag)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) license at your option.

## Contributing

Please open an issue if you have a feature request or a bug report. I'm happy to accept changes in line with the goals
of this crate (editing mp4 bytes).

When opening a pull request, please make a best effort to use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[//]: # (footnotes)

[^lossless]: The output will use the most efficient size headers possible, doesn't maintain non-standard reserved field values. Other than that, all data should be the same unless explicitly changed.

[^trimming]: Limitations apply; Trimming is a work in progress and is currently only supported for a single audio track. The edit list is not taken into consideration. It _should_ be possible to expand this support, with the help of edit lists, to include all track types on multi-track files (see https://github.com/jvatic/mp4-edit/issues/1).

[^interleaving]: While re-ordering chunks is not yet supported, new tracks may be added that interleave with existing ones, and any original interleaving is maintained. You are responsible for ensuring data is added in the correct locations via the `ChunkParser`. Higher level abstractions may be added later.

[^async]: The main API works with async IO, but each atom parser is itself sync.

[//]: # (badges)

[Build Status]: https://github.com/jvatic/mp4-edit/actions/workflows/rust.yml/badge.svg?branch=main
[Crate]: https://img.shields.io/crates/v/mp4-edit
[Documentation]: https://docs.rs/mp4-edit/badge.svg
[Dependency Status]: https://deps.rs/repo/github/jvatic/mp4-edit/status.svg
[Downloads]: https://img.shields.io/crates/d/mp4-edit
[License]: https://img.shields.io/crates/l/mp4-edit

[GitHub Actions]: https://github.com/jvatic/mp4-edit/actions?query=branch:main
[crates.io]: https://crates.io/crates/mp4-edit
[docs.rs]: https://docs.rs/mp4-edit
[deps.rs]: https://deps.rs/repo/github/jvatic/mp4-edit
