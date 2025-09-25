use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};

use futures_io::AsyncRead;
use std::{fmt, ops::Range};

use crate::{
    atom::{
        util::{read_to_end, DebugEllipsis, RangeCollection},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const STSC: &[u8; 4] = b"stsc";

#[derive(Default, Clone, Deref, DerefMut)]
pub struct SampleToChunkEntries(Vec<SampleToChunkEntry>);

impl SampleToChunkEntries {
    pub fn inner(&self) -> &[SampleToChunkEntry] {
        &self.0
    }
}

impl From<Vec<SampleToChunkEntry>> for SampleToChunkEntries {
    fn from(inner: Vec<SampleToChunkEntry>) -> Self {
        Self(inner)
    }
}

impl fmt::Debug for SampleToChunkEntries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() <= 10 {
            return f.debug_list().entries(self.0.iter()).finish();
        }
        f.debug_list()
            .entries(self.0.iter().take(10))
            .entry(&DebugEllipsis(Some(self.0.len() - 10)))
            .finish()
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
    /// `sample_indices_to_remove` must contain contiguous sample indices as a single range,
    /// multiple ranges must not overlap.
    pub(crate) fn remove_sample_indices(
        &mut self,
        sample_indices_to_remove: &[Range<usize>],
        total_chunks: usize,
    ) -> Vec<Range<usize>> {
        let mut next_sample_index = 0usize;

        let mut removed_chunk_indices = RangeCollection::new();
        let mut num_removed_chunks = 0usize;

        let mut remove_entry_range = RangeCollection::new();

        let mut insert_entries: Vec<(usize, SampleToChunkEntry)> = Vec::new();
        let mut num_inserted_entries = 0usize;

        let mut entries = self.entries.iter_mut().enumerate().peekable();
        while let Some((entry_index, entry)) = entries.next() {
            let entry_index = entry_index + num_inserted_entries;

            let chunk_index = entry.first_chunk as usize - 1;

            let entry_chunk_count = match entries.peek() {
                Some((_, next_entry)) => (next_entry.first_chunk - entry.first_chunk) as usize,
                None => total_chunks - entry.first_chunk as usize,
            };

            // ensure we're updating chunk indices after removing/inserting chunks
            entry.first_chunk -= num_removed_chunks as u32;

            let entry_sample_count = entry.samples_per_chunk as usize * entry_chunk_count;

            let entry_start_sample_index = next_sample_index;
            next_sample_index += entry.samples_per_chunk as usize * entry_chunk_count;

            let entry_end_sample_index =
                entry_start_sample_index + entry_sample_count.saturating_sub(1);

            let entry_sample_range_to_remove = entry_samples_to_remove(
                entry_start_sample_index,
                entry_end_sample_index,
                sample_indices_to_remove,
            );

            let entry_samples_to_remove = entry_sample_range_to_remove.len();

            // sample indices to remove fully excludes this entry
            if entry_samples_to_remove == 0 {
                continue;
            }

            // sample indices to remove fully includes this entry
            if entry_samples_to_remove == entry_sample_count {
                remove_entry_range.insert(entry_index..entry_index + 1);
                removed_chunk_indices.insert(chunk_index..(chunk_index + entry_chunk_count));
                num_removed_chunks += entry_chunk_count;
                continue;
            }

            // sample indices to remove partially includes this entry

            let relative_sample_range_to_remove = (entry_sample_range_to_remove.start
                - entry_start_sample_index)
                ..(entry_end_sample_index + 1 - entry_sample_range_to_remove.end);
            let first_affected_chunk_index = chunk_index
                + (relative_sample_range_to_remove.start / entry.samples_per_chunk as usize);

            let chunks_to_remove = entry_samples_to_remove / entry.samples_per_chunk as usize;

            let mut chunk_index = chunk_index;
            let mut entry_chunk_count = entry_chunk_count;
            let insert_entry_index = if first_affected_chunk_index > chunk_index {
                let num_chunks = first_affected_chunk_index - chunk_index;

                // we need to insert an entry for chunks upto the affected chunk
                insert_entries.push((
                    entry_index,
                    SampleToChunkEntry::builder()
                        .first_chunk(entry.first_chunk)
                        .samples_per_chunk(entry.samples_per_chunk)
                        .sample_description_index(entry.sample_description_index)
                        .build(),
                ));
                num_inserted_entries += 1;

                entry.first_chunk += num_chunks as u32;
                entry_chunk_count -= num_chunks;
                chunk_index += num_chunks;

                entry_index + 1
            } else {
                entry_index
            };

            if chunks_to_remove > 0 {
                removed_chunk_indices.insert(chunk_index..(chunk_index + chunks_to_remove));
                num_removed_chunks += chunks_to_remove;
            }

            let entry_samples_to_remove =
                entry_samples_to_remove - (chunks_to_remove * entry.samples_per_chunk as usize);

            if entry_samples_to_remove == 0 {
                continue;
            }

            if entry_chunk_count == 1 {
                entry.samples_per_chunk -= entry_samples_to_remove as u32;
            } else {
                // we need to insert a new entry
                insert_entries.push((
                    insert_entry_index,
                    SampleToChunkEntry::builder()
                        .first_chunk(entry.first_chunk)
                        .samples_per_chunk(entry.samples_per_chunk - entry_samples_to_remove as u32)
                        .sample_description_index(entry.sample_description_index)
                        .build(),
                ));
                num_inserted_entries += 1;

                // and then update the current one
                entry.first_chunk += 1;
            }
        }

        for (insert_index, entry) in insert_entries {
            self.entries.insert(insert_index, entry);
        }

        for mut range in remove_entry_range.into_iter() {
            // maybe merge entries before and after the removed ones
            if range.start > 0 {
                if let Ok([entry_prev, entry_next]) = self
                    .entries
                    .as_mut_slice()
                    .get_disjoint_mut([range.start - 1, range.end])
                {
                    if entry_prev.samples_per_chunk == entry_next.samples_per_chunk
                        && entry_prev.sample_description_index
                            == entry_next.sample_description_index
                    {
                        range.end += 1;
                    }
                }
            }

            self.entries.drain(range);
        }

        removed_chunk_indices.into_iter().collect()
    }
}

fn entry_samples_to_remove(
    entry_start_sample_index: usize,
    entry_end_sample_index: usize,
    sample_indices_to_remove: &[Range<usize>],
) -> Range<usize> {
    for range in sample_indices_to_remove.iter() {
        // entry is contained in range
        if range.contains(&entry_start_sample_index) && range.contains(&entry_end_sample_index) {
            return entry_start_sample_index..entry_end_sample_index + 1;
        }

        // range is contained in entry
        if range.start >= entry_start_sample_index && range.end <= entry_end_sample_index {
            return range.clone();
        }

        // range starts inside of entry
        if range.start >= entry_start_sample_index {
            return range.start..entry_end_sample_index + 1;
        }

        // range ends inside of entry
        if range.contains(&entry_start_sample_index)
            && range.start < entry_start_sample_index
            && range.end <= entry_end_sample_index
        {
            return entry_start_sample_index..range.end;
        }
    }

    0..0
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

impl ParseAtom for SampleToChunkAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STSC {
            return Err(ParseError::new_unexpected_atom(atom_type, STSC));
        }
        let data = read_to_end(reader).await?;
        parser::parse_stsc_data(&data)
    }
}

impl SerializeAtom for SampleToChunkAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STSC)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serialier::serialize_stsc_atom(self)
    }
}

mod serialier {
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
    use crate::atom::util::parser::{flags3, stream, version, Stream};

    pub fn parse_stsc_data(input: &[u8]) -> Result<SampleToChunkAtom, crate::ParseError> {
        parse_stsc_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_stsc_data_inner(input: &mut Stream<'_>) -> ModalResult<SampleToChunkAtom> {
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
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available stco/co64 test data files
    #[test]
    fn test_stsc_roundtrip() {
        test_atom_roundtrip_sync::<SampleToChunkAtom>(STSC);
    }

    struct RemoveSampleIndicesTestCase {
        sample_indices_to_remove: Vec<Range<usize>>,
        expected_removed_chunk_indices: Vec<Range<usize>>,
        expected_entries: Vec<SampleToChunkEntry>,
    }

    fn test_remove_sample_indices_stsc() -> SampleToChunkAtom {
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
                // chunks     9..19  (10)
                // samples 110..210 (100)
                SampleToChunkEntry {
                    first_chunk: 10,
                    samples_per_chunk: 10,
                    // NOTE: this is different to test this entry won't be merged
                    sample_description_index: 2,
                },
            ])
            .build()
    }

    fn test_remove_sample_indices<F>(test_case: F)
    where
        F: FnOnce(&SampleToChunkAtom) -> RemoveSampleIndicesTestCase,
    {
        let total_chunks = 20;
        let mut stsc = test_remove_sample_indices_stsc();
        let test_case = test_case(&stsc);
        let actual_removed_chunk_indices =
            stsc.remove_sample_indices(&test_case.sample_indices_to_remove, total_chunks);

        assert_eq!(
            actual_removed_chunk_indices, test_case.expected_removed_chunk_indices,
            "removed chunk indices don't match what's expected",
        );

        eprintln!("{:#?}", stsc.entries.0);

        stsc.entries
            .iter()
            .zip(test_case.expected_entries.iter())
            .enumerate()
            .for_each(|(index, (actual, expected))| {
                assert_eq!(
                    actual, expected,
                    "sample to chunk entries[{index}] doesn't match what's expected"
                );
            });

        assert_eq!(
            stsc.entries.0.len(),
            test_case.expected_entries.len(),
            "sample to chunk entries don't match what's expected",
        );
    }

    macro_rules! test_remove_sample_indices {
        ($($name:ident => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    test_remove_sample_indices($test_case);
                }
            )*
        };
    }

    test_remove_sample_indices!(
        remove_first_entry => |stsc| RemoveSampleIndicesTestCase {
            sample_indices_to_remove: vec![0..10],
            expected_removed_chunk_indices: vec![0..1],
            expected_entries: stsc.entries[1..].iter().cloned().map(|mut entry| {
                entry.first_chunk -= 1;
                entry
            }).collect::<Vec<_>>(),
        },
        remove_first_sample_from_first_entry => |stsc| {
            let mut expected_entries = stsc.entries.0.clone();
            expected_entries[0].samples_per_chunk -= 1;
            RemoveSampleIndicesTestCase {
                sample_indices_to_remove: vec![0..1],
                expected_removed_chunk_indices: vec![],
                expected_entries,
            }
        },
        remove_last_sample_from_first_entry => |stsc| {
            let mut expected_entries = stsc.entries.0.clone();
            expected_entries[0].samples_per_chunk -= 1;
            RemoveSampleIndicesTestCase {
                sample_indices_to_remove: vec![9..10],
                expected_removed_chunk_indices: vec![],
                expected_entries,
            }
        },
        remove_sample_from_second_entry => |stsc| {
            let mut expected_entries = stsc.entries.0.clone();
            let mut inserted_entry = expected_entries[1].clone();
            expected_entries[1].first_chunk += 1;
            inserted_entry.samples_per_chunk -= 1;
            expected_entries.insert(1, inserted_entry);
            RemoveSampleIndicesTestCase {
                sample_indices_to_remove: vec![10..11],
                expected_removed_chunk_indices: vec![],
                expected_entries,
            }
        },
        remove_chunk_from_second_entry => |stsc| {
            let mut expected_entries = stsc.entries.0.clone();
            expected_entries.iter_mut().skip(2).for_each(|entry| entry.first_chunk -= 1);
            RemoveSampleIndicesTestCase {
                sample_indices_to_remove: vec![10..30],
                expected_removed_chunk_indices: vec![1..2],
                expected_entries,
            }
        },
        remove_five_samples_from_last_entry_middle => |stsc| {
            RemoveSampleIndicesTestCase {
                sample_indices_to_remove: vec![151..156],
                expected_removed_chunk_indices: vec![],
                // we're removing 5 samples from chunk index 13
                // so the last entry (starting at chunk index 10) should be split into 3
                expected_entries: vec![
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
                    // 2. chunk with samples removed:
                    // chunks    13..14 (1)
                    // samples 150..155 (5)
                    SampleToChunkEntry {
                        first_chunk: 14,
                        samples_per_chunk: 5,
                        sample_description_index: 2,
                    },
                    // 3. remaining unaffected chunks:
                    // total chunks = 20 (0 chunks removed)
                    // chunks     14..19 (5)
                    // samples 155..205 (50) (5 samples removed)
                    SampleToChunkEntry {
                        first_chunk: 15,
                        samples_per_chunk: 10,
                        sample_description_index: 2,
                    },
                ],
            }
        },
        remove_fifteen_samples_from_last_entry_middle => |stsc| {
            RemoveSampleIndicesTestCase {
                sample_indices_to_remove: vec![151..166],
                expected_removed_chunk_indices: vec![13..14],
                // we're removing 15 samples starting from chunk index 13,
                // which will remove chunk index 13, and 5 samples from chunk index 14
                // so the last entry (starting at chunk index 10) should be split into 3
                expected_entries: vec![
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
                    // 2. chunk with samples removed:
                    // chunks    13..14 (1)
                    // samples 150..155 (5)
                    SampleToChunkEntry {
                        first_chunk: 14,
                        samples_per_chunk: 5,
                        sample_description_index: 2,
                    },
                    // 3. remaining unaffected chunks:
                    // total chunks = 19 (1 chunk removed)
                    // chunks    14..18  (4)
                    // samples 155..195 (40) (15 samples removed)
                    SampleToChunkEntry {
                        first_chunk: 15,
                        samples_per_chunk: 10,
                        sample_description_index: 2,
                    },
                ],
            }
        },
        remove_second_entry_merge_first_and_third => |stsc| RemoveSampleIndicesTestCase {
            sample_indices_to_remove: vec![10..50],
            expected_removed_chunk_indices: vec![1..3],
            expected_entries: vec![
                stsc.entries.first().cloned().unwrap(),
                stsc.entries.last().cloned().map(|mut entry| {
                    entry.first_chunk -= 2;
                    entry
                }).unwrap(),
            ],
        },
        remove_second_and_third_entry_no_merge => |stsc| RemoveSampleIndicesTestCase {
            sample_indices_to_remove: vec![10..110],
            expected_removed_chunk_indices: vec![1..9],
            expected_entries: vec![
                stsc.entries.first().cloned().unwrap(),
                stsc.entries.last().cloned().map(|mut entry| {
                    entry.first_chunk -= 8;
                    entry
                }).unwrap(),
            ],
        },
    );
}
