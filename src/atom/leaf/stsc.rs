use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};

use std::fmt;

use crate::{
    atom::{util::DebugList, FourCC},
    parser::ParseAtomData,
    writer::SerializeAtom,
    ParseError,
};

#[cfg(feature = "experimental-trim")]
use {
    crate::atom::stco_co64::ChunkOffsetOperationUnresolved,
    rangemap::RangeSet,
    std::{iter::Peekable, ops::Range, slice},
};

pub const STSC: FourCC = FourCC::new(b"stsc");

#[derive(Default, Clone, Deref, DerefMut)]
pub struct SampleToChunkEntries(Vec<SampleToChunkEntry>);

impl SampleToChunkEntries {
    pub fn inner(&self) -> &[SampleToChunkEntry] {
        &self.0
    }

    #[cfg(feature = "experimental-trim")]
    fn expanded_iter(&self, total_chunks: usize) -> ExpandedSampleToChunkEntryIter<'_> {
        ExpandedSampleToChunkEntryIter::new(total_chunks, &self.0)
    }
}

impl From<Vec<SampleToChunkEntry>> for SampleToChunkEntries {
    fn from(inner: Vec<SampleToChunkEntry>) -> Self {
        Self(inner)
    }
}

impl fmt::Debug for SampleToChunkEntries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&DebugList::new(self.0.iter(), 10), f)
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "experimental-trim")]
struct ExpandedSampleToChunkEntry {
    pub chunk_index: usize,
    pub sample_indices: Range<usize>,
    pub samples_per_chunk: u32,
    pub sample_description_index: u32,
}

/// Iterates over [`SampleToChunkEntry`]s expanded to a single entry per chunk.
#[cfg(feature = "experimental-trim")]
struct ExpandedSampleToChunkEntryIter<'a> {
    total_chunks: usize,
    next_sample_index: usize,
    current_entry: Option<(&'a SampleToChunkEntry, usize, usize)>,
    iter: Peekable<slice::Iter<'a, SampleToChunkEntry>>,
}

#[cfg(feature = "experimental-trim")]
impl<'a> ExpandedSampleToChunkEntryIter<'a> {
    fn new(total_chunks: usize, entries: &'a [SampleToChunkEntry]) -> Self {
        let iter = entries.iter().peekable();
        Self {
            total_chunks,
            next_sample_index: 0,
            current_entry: None,
            iter,
        }
    }
}

#[cfg(feature = "experimental-trim")]
impl<'a> Iterator for ExpandedSampleToChunkEntryIter<'a> {
    type Item = ExpandedSampleToChunkEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let (entry, chunk_index, chunk_count) = self.current_entry.take().or_else(|| {
            let entry = self.iter.next()?;
            let chunk_index = entry.first_chunk as usize - 1;
            let chunk_count = match self.iter.peek() {
                Some(next_entry) => (next_entry.first_chunk - entry.first_chunk) as usize,
                None => self.total_chunks - chunk_index,
            };
            Some((entry, chunk_index, chunk_count))
        })?;

        let first_sample_index = self.next_sample_index;
        self.next_sample_index += entry.samples_per_chunk as usize;

        if chunk_count > 1 {
            self.current_entry = Some((entry, chunk_index + 1, chunk_count - 1));
        }

        let sample_count = entry.samples_per_chunk as usize;
        let last_sample_index = first_sample_index + sample_count.saturating_sub(1);
        let sample_indices = first_sample_index..last_sample_index + 1;

        // probably best not to use the builder in a large loop
        Some(ExpandedSampleToChunkEntry {
            chunk_index,
            sample_indices,
            samples_per_chunk: entry.samples_per_chunk,
            sample_description_index: entry.sample_description_index,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.total_chunks, Some(self.total_chunks))
    }
}

/// Sample-to-Chunk entry - maps samples to chunks
#[derive(Debug, Clone, PartialEq, Eq, Builder)]
pub struct SampleToChunkEntry {
    /// First chunk number (1-based) that uses this entry
    pub first_chunk: u32,
    /// Number of samples in each chunk
    pub samples_per_chunk: u32,
    /// Sample description index (1-based, references stsd atom)
    pub sample_description_index: u32,
}

/// Sample-to-Chunk Atom - contains sample-to-chunk mapping table
#[derive(Default, Debug, Clone)]
pub struct SampleToChunkAtom {
    /// Version of the stsc atom format (0)
    pub version: u8,
    /// Flags for the stsc atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample-to-chunk entries
    pub entries: SampleToChunkEntries,
}

#[bon]
impl SampleToChunkAtom {
    #[builder]
    pub fn new(
        #[builder(default = 0)] version: u8,
        #[builder(default = [0u8; 3])] flags: [u8; 3],
        #[builder(with = FromIterator::from_iter)] entries: Vec<SampleToChunkEntry>,
    ) -> Self {
        Self {
            version,
            flags,
            entries: entries.into(),
        }
    }

    /// Removes sample indices from the sample-to-chunk mapping table,
    /// and returns the indices (starting from zero) of any chunks which are now empty (and should be removed)
    ///
    /// TODO: return a list of operations to apply to chunk offsets
    /// - [x] Remove(RangeSet of chunk indices to remove)
    /// - [ ] Insert(chunk index, sample index range) (will need to be cross referenced with sample_sizes)
    /// - [ ] ShiftLeft(chunk index, sample index range) (ditto '')
    ///
    /// `sample_indices_to_remove` must contain contiguous sample indices as a single range,
    /// multiple ranges must not overlap.
    #[cfg(feature = "experimental-trim")]
    pub(crate) fn remove_sample_indices(
        &mut self,
        sample_indices_to_remove: &RangeSet<usize>,
        total_chunks: usize,
    ) -> Vec<ChunkOffsetOperationUnresolved> {
        let mut chunk_ops = Vec::new();

        let mut num_removed_chunks = 0usize;
        let mut num_inserted_chunks = 0usize;

        struct Context<'a> {
            sample_indices_to_remove: &'a RangeSet<usize>,

            chunk_ops: &'a mut Vec<ChunkOffsetOperationUnresolved>,

            num_removed_chunks: &'a mut usize,
            num_inserted_chunks: &'a mut usize,

            next_entries: &'a mut Vec<SampleToChunkEntry>,
        }

        impl<'a> Context<'a> {
            pub fn process_entry(&mut self, entry: ExpandedSampleToChunkEntry) {
                // get only the sample indices that overlap with this entry
                if let Some(sample_indices_to_remove) =
                    entry_samples_to_remove(&entry.sample_indices, self.sample_indices_to_remove)
                        .first()
                {
                    if sample_indices_to_remove.len() >= entry.sample_indices.len() {
                        // sample indices to remove fully includes this entry
                        self.remove_chunk_offset(entry.chunk_index);
                        *self.num_removed_chunks += 1;
                    } else {
                        self.process_entry_partial_match(sample_indices_to_remove, entry);
                    }
                } else {
                    // no samples/chunks to remove for this entry
                    match self.next_entries.last() {
                        Some(prev_entry)
                            if prev_entry.samples_per_chunk == entry.samples_per_chunk
                                && prev_entry.sample_description_index
                                    == entry.sample_description_index =>
                        {
                            // redundand with prev entry
                        }
                        _ => {
                            self.insert_or_update_chunk_entry(SampleToChunkEntry {
                                first_chunk: (entry.chunk_index + 1) as u32,
                                samples_per_chunk: entry.samples_per_chunk,
                                sample_description_index: entry.sample_description_index,
                            });
                        }
                    }
                }
            }

            fn process_entry_partial_match(
                &mut self,
                sample_indices_to_remove: &Range<usize>,
                entry: ExpandedSampleToChunkEntry,
            ) {
                if sample_indices_to_remove.start == entry.sample_indices.start {
                    /*
                     * process trim start
                     *
                     * e.g.
                     * ------------------------------------------------------------------
                     * | 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 |
                     * | ^-----------------^  ^---------------------------------------^ |
                     * |     trim range                   chunk 0 (remainder)           |
                     * | |- offset --------->|  (+= size(trim...))                      |
                     * ------------------------------------------------------------------
                     */

                    // chunk offset increases by the size of the removed samples
                    self.shift_chunk_offset_right(
                        entry.chunk_index,
                        sample_indices_to_remove.clone(),
                    );

                    // chunk sample count decreases by n removed samples
                    self.insert_or_update_chunk_entry(SampleToChunkEntry {
                        first_chunk: entry.chunk_index as u32 + 1,
                        samples_per_chunk: entry.samples_per_chunk
                            - sample_indices_to_remove.len() as u32,
                        sample_description_index: entry.sample_description_index,
                    });

                    // process any additional trim ranges (e.g. middle and/or end)
                    self.process_entry(ExpandedSampleToChunkEntry {
                        chunk_index: entry.chunk_index,
                        sample_indices: sample_indices_to_remove.end..entry.sample_indices.end,
                        samples_per_chunk: entry.samples_per_chunk
                            - sample_indices_to_remove.len() as u32,
                        sample_description_index: entry.sample_description_index,
                    });
                } else if sample_indices_to_remove.end == entry.sample_indices.end {
                    /*
                     * process trim end
                     *
                     * e.g.
                     * ------------------------------------------------------------------
                     * | 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 |
                     * | ^------------------------------^ ^---------------------------^ |
                     * |       chunk 0 (remainder)                 trim range           |
                     * ------------------------------------------------------------------
                     */

                    // chunk sample count decreases by n removed samples
                    self.insert_or_update_chunk_entry(SampleToChunkEntry {
                        first_chunk: entry.chunk_index as u32 + 1,
                        samples_per_chunk: entry.samples_per_chunk
                            - sample_indices_to_remove.len() as u32,
                        sample_description_index: entry.sample_description_index,
                    });

                    // since we reached the end of the chunk/entry, and trim ranges are processed in order,
                    // there are no additional matches to be had
                } else {
                    /*
                     * process trim middle
                     *
                     * e.g.
                     * ------------------------------------------------------------------
                     * | 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 |
                     * | ^------------^ ^------------------^ ^------------------------^ |
                     * |    chunk 0    |    trim range      |      new chunk 1          |
                     * | |- offsetA    +   size(trim...)    = offsetB (chunk 1)         |
                     * ------------------------------------------------------------------
                     */

                    // insert a new chunk after the current one,
                    // whose offset is the existing offset + size of removed samples
                    self.insert_chunk_offset(
                        entry.chunk_index + 1,
                        sample_indices_to_remove.clone(),
                    );

                    // insert or update entry for the current chunk
                    self.insert_or_update_chunk_entry(SampleToChunkEntry {
                        first_chunk: entry.chunk_index as u32 + 1,
                        samples_per_chunk: (entry.sample_indices.len()
                            - (sample_indices_to_remove.start..entry.sample_indices.end).len())
                            as u32,
                        sample_description_index: entry.sample_description_index,
                    });

                    // insert or update entry for the new chunk
                    self.insert_or_update_chunk_entry(SampleToChunkEntry {
                        first_chunk: entry.chunk_index as u32 + 2,
                        samples_per_chunk: (entry.sample_indices.len()
                            - (entry.sample_indices.start..sample_indices_to_remove.end).len())
                            as u32,
                        sample_description_index: entry.sample_description_index,
                    });

                    // increment the counter after we've inserted the entry
                    *self.num_inserted_chunks += 1;

                    // process any additional trim ranges on the new chunk (e.g. middle and/or end)
                    self.process_entry(ExpandedSampleToChunkEntry {
                        chunk_index: entry.chunk_index + 1,
                        sample_indices: sample_indices_to_remove.end..entry.sample_indices.end,
                        samples_per_chunk: entry.samples_per_chunk,
                        sample_description_index: entry.sample_description_index,
                    });
                }
            }

            fn adjusted_chunk_index(&self, chunk_index: usize) -> usize {
                chunk_index + *self.num_inserted_chunks - *self.num_removed_chunks
            }

            fn insert_or_update_chunk_entry(&mut self, mut entry: SampleToChunkEntry) {
                entry.first_chunk = self.adjusted_chunk_index(entry.first_chunk as usize) as u32;
                match self.next_entries.last_mut() {
                    Some(prev_entry) if prev_entry.first_chunk == entry.first_chunk => {
                        *prev_entry = entry
                    }
                    _ => {
                        self.next_entries.push(entry);
                    }
                }
            }

            /// increase chunk offset by the size of a removed sample range
            fn shift_chunk_offset_right(
                &mut self,
                chunk_index: usize,
                removed_sample_indices: Range<usize>,
            ) {
                self.chunk_ops
                    .push(ChunkOffsetOperationUnresolved::ShiftRight {
                        chunk_index_unadjusted: chunk_index,
                        chunk_index: self.adjusted_chunk_index(chunk_index),
                        sample_indices: removed_sample_indices,
                    });
            }

            fn insert_chunk_offset(
                &mut self,
                chunk_index: usize,
                removed_sample_indices: Range<usize>,
            ) {
                self.chunk_ops.push(ChunkOffsetOperationUnresolved::Insert {
                    chunk_index_unadjusted: chunk_index,
                    chunk_index: self.adjusted_chunk_index(chunk_index),
                    sample_indices: removed_sample_indices,
                });
            }

            fn remove_chunk_offset(&mut self, chunk_index: usize) {
                let chunk_index = self.adjusted_chunk_index(chunk_index);
                match self.chunk_ops.last_mut() {
                    Some(ChunkOffsetOperationUnresolved::Remove(prev_op))
                        if prev_op.start == chunk_index =>
                    {
                        // either merge with the previous remove range
                        prev_op.end += 1;
                    }
                    _ => {
                        // or insert a new remove range
                        self.chunk_ops.push(ChunkOffsetOperationUnresolved::Remove(
                            chunk_index..chunk_index + 1,
                        ));
                    }
                }
            }
        }

        let num_sample_ranges_to_remove = sample_indices_to_remove.iter().count();
        let prev_len = self.entries.len();
        self.entries = SampleToChunkEntries(self.entries.expanded_iter(total_chunks).fold(
            // TODO: evaluate the actual worst-case additional entries
            Vec::with_capacity(prev_len + (num_sample_ranges_to_remove * 4)),
            |mut next_entries, entry| {
                let mut ctx = Context {
                    sample_indices_to_remove,

                    chunk_ops: &mut chunk_ops,

                    num_removed_chunks: &mut num_removed_chunks,
                    num_inserted_chunks: &mut num_inserted_chunks,

                    next_entries: &mut next_entries,
                };

                ctx.process_entry(entry);

                next_entries
            },
        ));
        self.entries.shrink_to_fit();

        chunk_ops
    }
}

#[cfg(feature = "experimental-trim")]
fn entry_samples_to_remove(
    entry_sample_indices: &Range<usize>,
    sample_indices_to_remove: &RangeSet<usize>,
) -> RangeSet<usize> {
    let mut entry_samples_to_remove = RangeSet::new();

    for range in sample_indices_to_remove.overlapping(entry_sample_indices) {
        let range =
            range.start.max(entry_sample_indices.start)..range.end.min(entry_sample_indices.end);
        entry_samples_to_remove.insert(range);
    }

    entry_samples_to_remove
}

impl<S: sample_to_chunk_atom_builder::State> SampleToChunkAtomBuilder<S> {
    pub fn entry(
        self,
        entry: impl Into<SampleToChunkEntry>,
    ) -> SampleToChunkAtomBuilder<sample_to_chunk_atom_builder::SetEntries<S>>
    where
        S::Entries: sample_to_chunk_atom_builder::IsUnset,
    {
        self.entries(vec![entry.into()])
    }
}

impl From<Vec<SampleToChunkEntry>> for SampleToChunkAtom {
    fn from(entries: Vec<SampleToChunkEntry>) -> Self {
        SampleToChunkAtom {
            version: 0,
            flags: [0u8; 3],
            entries: entries.into(),
        }
    }
}

impl ParseAtomData for SampleToChunkAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, STSC);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_stsc_data.parse(stream(input))?)
    }
}

impl SerializeAtom for SampleToChunkAtom {
    fn atom_type(&self) -> FourCC {
        STSC
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_stsc_atom(self)
    }
}

mod serializer {
    use crate::atom::util::serializer::be_u32;

    use super::SampleToChunkAtom;

    pub fn serialize_stsc_atom(stsc: SampleToChunkAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(stsc.version);
        data.extend(stsc.flags);

        data.extend(be_u32(
            stsc.entries
                .len()
                .try_into()
                .expect("entries len should fit in u32"),
        ));

        for entry in stsc.entries.iter() {
            data.extend(entry.first_chunk.to_be_bytes());
            data.extend(entry.samples_per_chunk.to_be_bytes());
            data.extend(entry.sample_description_index.to_be_bytes());
        }

        data
    }
}

mod parser {
    use winnow::{
        binary::{be_u32, length_repeat},
        combinator::{seq, trace},
        error::{StrContext, StrContextValue},
        ModalResult, Parser,
    };

    use super::{SampleToChunkAtom, SampleToChunkEntries, SampleToChunkEntry};
    use crate::atom::util::parser::{flags3, version, Stream};

    pub fn parse_stsc_data(input: &mut Stream<'_>) -> ModalResult<SampleToChunkAtom> {
        trace(
            "stsc",
            seq!(SampleToChunkAtom {
                version: version,
                flags: flags3,
                entries: length_repeat(be_u32, entry)
                    .map(SampleToChunkEntries)
                    .context(StrContext::Label("entries")),
            })
            .context(StrContext::Label("stsc")),
        )
        .parse_next(input)
    }

    fn entry(input: &mut Stream<'_>) -> ModalResult<SampleToChunkEntry> {
        trace(
            "entry",
            seq!(SampleToChunkEntry {
                first_chunk: be_u32
                    .verify(|first_chunk| *first_chunk > 0)
                    .context(StrContext::Label("first_chunk"))
                    .context(StrContext::Expected(StrContextValue::Description(
                        "1-based index"
                    ))),
                samples_per_chunk: be_u32
                    .verify(|samples_per_chunk| *samples_per_chunk > 0)
                    .context(StrContext::Label("samples_per_chunk"))
                    .context(StrContext::Expected(StrContextValue::Description(
                        "sample count > 0"
                    ))),
                sample_description_index: be_u32
                    .context(StrContext::Label("sample_description_index"))
                    .context(StrContext::Expected(StrContextValue::Description(
                        "1-based index"
                    ))),
            })
            .context(StrContext::Label("entry")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available stco/co64 test data files
    #[test]
    fn test_stsc_roundtrip() {
        test_atom_roundtrip::<SampleToChunkAtom>(STSC);
    }
}

#[cfg(feature = "experimental-trim")]
#[cfg(test)]
mod trim_tests {
    use super::*;

    struct EntrySamplesToRemoveTestCase {
        entry_start_sample_index: usize,
        entry_end_sample_index: usize,
        sample_indices_to_remove: Vec<Range<usize>>,
        expected_entry_samples_to_remove: Vec<Range<usize>>,
    }

    fn test_entry_samples_to_remove(tc: EntrySamplesToRemoveTestCase) {
        let sample_indices_to_remove = RangeSet::from_iter(tc.sample_indices_to_remove.into_iter());
        let entry_sample_indices = tc.entry_start_sample_index..tc.entry_end_sample_index + 1;
        let actual_entry_samples_to_remove =
            entry_samples_to_remove(&entry_sample_indices, &sample_indices_to_remove);
        let expected_entry_samples_to_remove =
            RangeSet::from_iter(tc.expected_entry_samples_to_remove.into_iter());
        assert_eq!(
            actual_entry_samples_to_remove,
            expected_entry_samples_to_remove
        );
    }

    mod test_entry_samples_to_remove {
        use super::*;

        macro_rules! test_entry_samples_to_remove {
            ($(
                $name:ident => {
                    $( $field:ident: $value:expr ),+ $(,)?
                },
            )*) => {
                $(
                    #[test]
                    fn $name() {
                        test_entry_samples_to_remove!(@inner $( $field: $value ),+);
                    }
                )*
            };

            (@inner $( $field:ident: $value:expr ),+) => {
                let tc = EntrySamplesToRemoveTestCase {
                    $( $field: $value ),+,
                };

                test_entry_samples_to_remove(tc);
            };
        }

        test_entry_samples_to_remove!(
            entry_contained_in_single_range => {
                entry_start_sample_index: 800,
                entry_end_sample_index: 1200,
                sample_indices_to_remove: vec![300..2000],
                expected_entry_samples_to_remove: vec![800..1201],
            },
            entry_contained_in_multiple_ranges => {
                entry_start_sample_index: 800,
                entry_end_sample_index: 1200,
                sample_indices_to_remove: vec![300..900, 1000..2000],
                expected_entry_samples_to_remove: vec![800..900, 1000..1201],
            },
            entry_starts_in_single_range => {
                entry_start_sample_index: 800,
                entry_end_sample_index: 1200,
                sample_indices_to_remove: vec![1000..2000],
                expected_entry_samples_to_remove: vec![1000..1201],
            },
            entry_ends_in_single_range => {
                entry_start_sample_index: 800,
                entry_end_sample_index: 1200,
                sample_indices_to_remove: vec![100..1000],
                expected_entry_samples_to_remove: vec![800..1000],
            },
            single_range_contained_in_entry => {
                entry_start_sample_index: 800,
                entry_end_sample_index: 1200,
                sample_indices_to_remove: vec![900..1000],
                expected_entry_samples_to_remove: vec![900..1000],
            },
        );
    }

    #[derive(Builder)]
    struct RemoveSampleIndicesTestCase {
        #[builder(default = 20)]
        total_chunks: usize,
        sample_indices_to_remove: Vec<Range<usize>>,
        expected_removed_chunk_indices: Vec<Range<usize>>,
        expected_entries: Vec<SampleToChunkEntry>,
    }

    fn test_remove_sample_indices_default_stsc() -> SampleToChunkAtom {
        SampleToChunkAtom::builder()
            .entries(vec![
                // chunks   0..1  (1)
                // samples 0..10 (10)
                SampleToChunkEntry {
                    first_chunk: 1,
                    samples_per_chunk: 10,
                    sample_description_index: 1,
                },
                // chunks    1..3  (2)
                // samples 10..50 (40)
                SampleToChunkEntry {
                    first_chunk: 2,
                    samples_per_chunk: 20,
                    sample_description_index: 1,
                },
                // chunks     3..9  (6)
                // samples 50..110 (60)
                SampleToChunkEntry {
                    first_chunk: 4,
                    samples_per_chunk: 10,
                    sample_description_index: 1,
                },
                // total chunks = 20
                // chunks     9..20  (11)
                // samples 110..220 (100)
                SampleToChunkEntry {
                    first_chunk: 10,
                    samples_per_chunk: 10,
                    // NOTE: this is different to test this entry won't be merged
                    sample_description_index: 2,
                },
            ])
            .build()
    }

    fn test_remove_sample_indices<F>(mut stsc: SampleToChunkAtom, test_case: F)
    where
        F: FnOnce(&SampleToChunkAtom) -> RemoveSampleIndicesTestCase,
    {
        let test_case = test_case(&stsc);
        let total_chunks = test_case.total_chunks;
        let sample_indices_to_remove =
            RangeSet::from_iter(test_case.sample_indices_to_remove.into_iter());
        let actual_chunk_offset_ops =
            stsc.remove_sample_indices(&sample_indices_to_remove, total_chunks);

        // TODO: add assertions for all the ops
        let actual_removed_chunk_indices = actual_chunk_offset_ops
            .iter()
            .filter_map(|op| match op {
                ChunkOffsetOperationUnresolved::Remove(chunk_offsets) => Some(chunk_offsets),
                _ => None,
            })
            .cloned()
            // chunk ops take into account previous operations having been applied
            // adjust ranges to be in terms of input chunk indices (easier to reason about in the test case)
            .scan(0usize, |n_removed, range| {
                let range = (range.start + *n_removed)..(range.end + *n_removed);
                *n_removed += range.len();
                Some(range)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            actual_removed_chunk_indices, test_case.expected_removed_chunk_indices,
            "removed chunk indices don't match what's expected",
        );

        stsc.entries
            .iter()
            .zip(test_case.expected_entries.iter())
            .enumerate()
            .for_each(|(index, (actual, expected))| {
                assert_eq!(
                    actual, expected,
                    "sample to chunk entries[{index}] doesn't match what's expected\n{:#?}",
                    stsc.entries,
                );
            });

        assert_eq!(
            stsc.entries.0.len(),
            test_case.expected_entries.len(),
            "expected {} sample to chunk entries, got {}: {:#?}",
            test_case.expected_entries.len(),
            stsc.entries.len(),
            stsc.entries,
        );
    }

    macro_rules! test_remove_sample_indices {
        ($($name:ident $($stsc:expr)? => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    test_remove_sample_indices!(@inner $($stsc)? => $test_case);
                }
            )*
        };
        (@inner => $test_case:expr) => {
            test_remove_sample_indices!(@inner test_remove_sample_indices_default_stsc() => $test_case);
        };
        (@inner $stsc:expr => $test_case:expr) => {
            test_remove_sample_indices($stsc, $test_case);
        };
    }

    // TODO: test inserted and adjusted chunk offsets in addition to removed offsets
    mod test_remove_sample_indices {
        use super::*;

        test_remove_sample_indices!(
            remove_first_entry => |stsc| RemoveSampleIndicesTestCase::builder().
                sample_indices_to_remove(vec![0..10]).
                expected_removed_chunk_indices(vec![0..1]).
                expected_entries(stsc.entries[1..].iter().cloned().map(|mut entry| {
                    entry.first_chunk -= 1;
                    entry
                }).collect::<Vec<_>>()).build(),
            remove_first_sample_from_first_entry => |stsc| {
                let mut expected_entries = stsc.entries.0.clone();
                expected_entries[0].samples_per_chunk -= 1;
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![0..1]).
                    expected_removed_chunk_indices(vec![]).
                    expected_entries(expected_entries).build()
            },
            remove_last_sample_from_first_entry => |stsc| {
                let mut expected_entries = stsc.entries.0.clone();
                // there's just a single chunk in the first entry
                expected_entries[0].samples_per_chunk -= 1;
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![9..10]).
                    expected_removed_chunk_indices(vec![]).
                    expected_entries(expected_entries).build()
            },
            remove_sample_from_second_entry => |stsc| {
                let mut expected_entries = stsc.entries.0.clone();
                let mut inserted_entry = expected_entries[1].clone();
                expected_entries[1].first_chunk += 1;
                inserted_entry.samples_per_chunk -= 1;
                expected_entries.insert(1, inserted_entry);
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![10..11]).
                    expected_removed_chunk_indices(vec![]).
                    expected_entries(expected_entries).build()
            },
            remove_first_chunk_from_second_entry => |stsc| {
                let mut expected_entries = stsc.entries.0.clone();
                expected_entries.iter_mut().skip(2).for_each(|entry| entry.first_chunk -= 1);
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![10..30]).
                    expected_removed_chunk_indices(vec![1..2]).
                    expected_entries(expected_entries).build()
            },
            remove_five_samples_from_last_entry_middle => |stsc| {
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![151..156]).
                    expected_removed_chunk_indices(vec![]).
                    // we're removing 5 samples from chunk index 13
                    // so the last entry (starting at chunk index 10) should be split into 4
                    expected_entries(vec![
                        stsc.entries[0].clone(),
                        stsc.entries[1].clone(),
                        stsc.entries[2].clone(),
                        // 1. unaffected chunks:
                        // chunks     9..13  (4)
                        // samples 110..150 (40)
                        SampleToChunkEntry {
                            first_chunk: 10,
                            samples_per_chunk: 10,
                            sample_description_index: 2,
                        },
                        // 2. chunk with samples removed from middle:
                        // this is the left side of the trim range (1 sample remaining)
                        // chunks    13..14 (1)
                        // samples 150..151 (1)
                        SampleToChunkEntry {
                            first_chunk: 14,
                            samples_per_chunk: 1,
                            sample_description_index: 2,
                        },
                        // 3. chunk with samples removed from middle:
                        // this is the right side of the trim range where we've inserted a new chunk
                        // chunks    13..14 (1) -> 14..15
                        // samples 156..160 (4)
                        SampleToChunkEntry {
                            first_chunk: 15,
                            samples_per_chunk: 4,
                            sample_description_index: 2,
                        },
                        // 4. remaining unaffected chunks:
                        // total chunks = 20 (0 chunks removed)
                        // chunks     14..19 (5) -> 15..20
                        // samples 155..205 (50) (5 samples removed)
                        SampleToChunkEntry {
                            first_chunk: 16,
                            samples_per_chunk: 10,
                            sample_description_index: 2,
                        },
                    ]).build()
            },
            remove_fifteen_samples_from_last_entry_middle => |stsc| {
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![150..165]).
                    expected_removed_chunk_indices(vec![13..14]).
                    // we're removing 15 samples starting from chunk index 13,
                    // which will remove chunk index 13, and 5 samples from chunk index 14
                    // so the last entry (starting at chunk index 10) should be split into 3
                    expected_entries(vec![
                        stsc.entries[0].clone(),
                        stsc.entries[1].clone(),
                        stsc.entries[2].clone(),
                        // 1. unaffected chunks:
                        // chunks     9..13  (4)
                        // samples 110..150 (40)
                        SampleToChunkEntry {
                            first_chunk: 10,
                            samples_per_chunk: 10,
                            sample_description_index: 2,
                        },
                        // 2. chunk 13..4 removed
                        // 3. chunk with samples removed:
                        // chunks    14..15 (1)
                        // samples 150..155 (5)
                        SampleToChunkEntry {
                            first_chunk: 14,
                            samples_per_chunk: 5,
                            sample_description_index: 2,
                        },
                        // 4. remaining unaffected chunks:
                        // total chunks = 19 (1 chunk removed)
                        // chunks    14..18  (4)
                        // samples 160..210 (40) (15 samples removed)
                        SampleToChunkEntry {
                            first_chunk: 15,
                            samples_per_chunk: 10,
                            sample_description_index: 2,
                        },
                    ]).build()
            },
            remove_second_entry_merge_first_and_third => |stsc| RemoveSampleIndicesTestCase::builder().
                sample_indices_to_remove(vec![10..50]).
                expected_removed_chunk_indices(vec![1..3]).
                expected_entries(vec![
                    stsc.entries.first().cloned().unwrap(),
                    stsc.entries.last().cloned().map(|mut entry| {
                        entry.first_chunk -= 2;
                        entry
                    }).unwrap(),
                ]).build(),
            remove_second_and_third_entry_no_merge => |stsc| RemoveSampleIndicesTestCase::builder().
                sample_indices_to_remove(vec![10..110]).
                expected_removed_chunk_indices(vec![1..9]).
                expected_entries(vec![
                    stsc.entries.first().cloned().unwrap(),
                    stsc.entries.last().cloned().map(|mut entry| {
                        entry.first_chunk -= 8;
                        entry
                    }).unwrap(),
                ]).build(),
            remove_first_and_last_entry => |stsc| RemoveSampleIndicesTestCase::builder().
                sample_indices_to_remove(vec![0..10, 110..220]).
                expected_removed_chunk_indices(vec![0..1, 9..20]).
                expected_entries(vec![
                    stsc.entries.get(1).cloned().map(|mut entry| {
                        entry.first_chunk = 1;
                        entry
                    }).unwrap(),
                    stsc.entries.get(2).cloned().map(|mut entry| {
                        entry.first_chunk = 3;
                        entry
                    }).unwrap(),
                ]).build(),

            remove_last_chunk_single_entry {
                SampleToChunkAtom::builder().
                    entry(SampleToChunkEntry::builder().
                        first_chunk(1).
                        samples_per_chunk(2).
                        sample_description_index(1).
                        build(),
                    ).build()
            } => |stsc| RemoveSampleIndicesTestCase::builder().
                sample_indices_to_remove(vec![38..40]).
                expected_removed_chunk_indices(vec![19..20]).
                expected_entries(vec![
                    stsc.entries.first().cloned().unwrap(),
                ]).build(),

            remove_multiple_ranges_from_single_entry {
                SampleToChunkAtom::builder().
                    entry(SampleToChunkEntry::builder().
                        first_chunk(1).
                        samples_per_chunk(1).
                        sample_description_index(1).
                        build(),
                    ).build()
            } => |stsc| RemoveSampleIndicesTestCase::builder().
                total_chunks(100).
                sample_indices_to_remove(vec![20..41, 60..81]).
                expected_removed_chunk_indices(vec![20..41, 60..81]).
                expected_entries(vec![
                    stsc.entries.first().cloned().unwrap(),
                ]).build(),

            remove_mid_second_chunk_to_mid_last_chunk => |stsc| {
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![15..215]).
                    expected_removed_chunk_indices(vec![2..19]).
                    expected_entries(vec![
                        // first entry is unchanged
                        stsc.entries[0].clone(),
                        // samples 10..15 remain intact
                        SampleToChunkEntry {
                            first_chunk: 2,
                            samples_per_chunk: 5,
                            sample_description_index: 1,
                        },
                        // samples 215..220 remain intact
                        SampleToChunkEntry {
                            first_chunk: 3,
                            samples_per_chunk: 5,
                            sample_description_index: 2,
                        },
                    ]).build()
            },

            remove_mid_fifth_chunk_to_mid_last_chunk => |stsc| {
                RemoveSampleIndicesTestCase::builder().
                    sample_indices_to_remove(vec![65..215]).
                    expected_removed_chunk_indices(vec![5..19]).
                    expected_entries(vec![
                        // first two entries are unchanged
                        stsc.entries[0].clone(),
                        stsc.entries[1].clone(),
                        // 4th chunk remains unchanged
                        SampleToChunkEntry {
                            first_chunk: 4,
                            samples_per_chunk: 10,
                            sample_description_index: 1,
                        },
                        // samples 60..65 remain intact
                        SampleToChunkEntry {
                            first_chunk: 5,
                            samples_per_chunk: 5,
                            sample_description_index: 1,
                        },
                        // samples 215..220 remain intact
                        SampleToChunkEntry {
                            first_chunk: 6,
                            samples_per_chunk: 5,
                            sample_description_index: 2,
                        },
                    ]).build()
            },
        );
    }
}
