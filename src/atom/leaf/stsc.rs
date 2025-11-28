use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};

use futures_io::AsyncRead;
use rangemap::RangeSet;
use std::{fmt, ops::Range};

use crate::{
    atom::{
        util::{read_to_end, DebugList},
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
        fmt::Debug::fmt(&DebugList::new(self.0.iter(), 10), f)
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
    ///
    /// TODO: simplify this function, it's too complex
    pub(crate) fn remove_sample_indices(
        &mut self,
        sample_indices_to_remove: &RangeSet<usize>,
        total_chunks: usize,
    ) -> Vec<Range<usize>> {
        let mut next_sample_index = 0usize;

        let mut removed_chunk_indices = RangeSet::new();
        let mut num_removed_chunks = 0usize;

        let mut remove_entry_range = RangeSet::new();

        let mut insert_entries: Vec<(usize, SampleToChunkEntry)> = Vec::new();
        let mut num_inserted_entries = 0usize;

        struct Context<'a> {
            sample_indices_to_remove: &'a RangeSet<usize>,

            removed_chunk_indices: &'a mut RangeSet<usize>,
            num_removed_chunks: &'a mut usize,

            remove_entry_range: &'a mut RangeSet<usize>,

            insert_entries: &'a mut Vec<(usize, SampleToChunkEntry)>,
            num_inserted_entries: &'a mut usize,
        }

        struct Entry<'a> {
            index: usize,
            inner: &'a mut SampleToChunkEntry,

            chunk_index: usize,
            chunk_count: usize,
            first_sample_index: usize,
        }

        impl<'a> Entry<'a> {
            fn process(self, ctx: &mut Context<'a>) {
                if self.chunk_count == 0 {
                    self.remove_entry(ctx);
                    return;
                }

                let sample_count = self.inner.samples_per_chunk as usize * self.chunk_count;
                let last_sample_index = self.first_sample_index + sample_count.saturating_sub(1);
                let samples = self.first_sample_index..last_sample_index + 1;

                // get only the sample indices that overlap with this entry
                if let Some(sample_indices_to_remove) =
                    entry_samples_to_remove(&samples, ctx.sample_indices_to_remove).first()
                {
                    // sample indices to remove fully includes this entry
                    if sample_indices_to_remove.len() >= samples.len() {
                        self.remove_entry(ctx);
                    } else {
                        // sample indices to remove partially includes this entry
                        self.remove_partial(ctx, sample_indices_to_remove);
                    }
                }
            }

            fn remove_entry(mut self, ctx: &mut Context<'a>) {
                ctx.remove_entry_range.insert(self.index..self.index + 1);
                self.remove_chunks(ctx, self.chunk_count);
            }

            fn remove_partial(
                mut self,
                ctx: &mut Context<'a>,
                sample_indices_to_remove: &Range<usize>,
            ) {
                let first_affected_chunk_index =
                    if sample_indices_to_remove.start > self.first_sample_index {
                        self.chunk_index
                            + ((sample_indices_to_remove.start - self.first_sample_index)
                                / self.inner.samples_per_chunk as usize)
                    } else {
                        self.chunk_index
                    };

                // number of whole chunks to remove
                let n_chunks_to_remove =
                    sample_indices_to_remove.len() / self.inner.samples_per_chunk as usize;

                // number of samples to remove after removing `n_chunks_to_remove`
                let n_samples_to_remove = sample_indices_to_remove.len()
                    - (n_chunks_to_remove * self.inner.samples_per_chunk as usize);

                // there are leading samples
                if n_chunks_to_remove > 0
                    && first_affected_chunk_index == self.chunk_index
                    && sample_indices_to_remove.start > self.first_sample_index
                {
                    let num_samples =
                        (sample_indices_to_remove.start - self.first_sample_index) as u32;
                    // insert an entry before this one up to the affected sample index
                    self.insert_before(
                        ctx,
                        SampleToChunkEntry::builder()
                            .first_chunk(self.inner.first_chunk)
                            .samples_per_chunk(num_samples)
                            .sample_description_index(self.inner.sample_description_index)
                            .build(),
                    );
                    self.inner.first_chunk += 1;

                    Entry {
                        index: self.index,
                        inner: self.inner,
                        chunk_index: self.chunk_index + 1,
                        chunk_count: self.chunk_count - 1,
                        first_sample_index: sample_indices_to_remove.start,
                    }
                    .process(ctx);
                    return;
                }

                // there are leading chunks
                if first_affected_chunk_index > self.chunk_index {
                    let num_chunks = first_affected_chunk_index - self.chunk_index;

                    self.insert_before(
                        ctx,
                        SampleToChunkEntry::builder()
                            .first_chunk(self.inner.first_chunk)
                            .samples_per_chunk(self.inner.samples_per_chunk)
                            .sample_description_index(self.inner.sample_description_index)
                            .build(),
                    );
                    self.inner.first_chunk += num_chunks as u32;

                    let samples_per_chunk = self.inner.samples_per_chunk as usize;
                    Entry {
                        index: self.index,
                        inner: self.inner,
                        chunk_index: self.chunk_index + num_chunks,
                        chunk_count: self.chunk_count - num_chunks,
                        first_sample_index: self.first_sample_index
                            + (num_chunks * samples_per_chunk),
                    }
                    .process(ctx);
                    return;
                };

                // remove any full chunks that matched the remove sample range
                if n_chunks_to_remove > 0 {
                    self.remove_chunks(ctx, n_chunks_to_remove);

                    let samples_per_chunk = self.inner.samples_per_chunk as usize;
                    Entry {
                        index: self.index,
                        inner: self.inner,
                        chunk_index: self.chunk_index + n_chunks_to_remove,
                        chunk_count: self.chunk_count - n_chunks_to_remove,
                        first_sample_index: self.first_sample_index
                            + (n_chunks_to_remove * samples_per_chunk),
                    }
                    .process(ctx);
                    return;
                }

                // there are still samples left to remove for this entry that are less than a full chunk
                if n_samples_to_remove > 0 {
                    if self.chunk_count > 1 {
                        // there are full chunks in this entry that remain intact after removing these samples
                        // so we need to insert a new entry to handle the resulting partial chunk
                        self.insert_before(
                            ctx,
                            SampleToChunkEntry::builder()
                                .first_chunk(self.inner.first_chunk)
                                .samples_per_chunk(
                                    self.inner.samples_per_chunk - n_samples_to_remove as u32,
                                )
                                .sample_description_index(self.inner.sample_description_index)
                                .build(),
                        );

                        self.inner.first_chunk += 1;

                        Entry {
                            index: self.index,
                            inner: self.inner,
                            chunk_index: self.chunk_index + 1,
                            chunk_count: self.chunk_count - 1,
                            first_sample_index: sample_indices_to_remove.end,
                        }
                        .process(ctx);
                        return;
                    } else {
                        // the remaining samples belong to the last remaining chunk in this entry
                        // so we can just remove them directly
                        self.inner.samples_per_chunk -= n_samples_to_remove as u32;

                        // there's nothing left to process for this entry
                        return;
                    }
                }

                unreachable!(
                    "there should always be samples to remove when remove_partial is called"
                );
            }

            fn remove_chunks(&mut self, ctx: &mut Context<'a>, n_chunks: usize) {
                if n_chunks == 0 {
                    return;
                }

                let chunk_index = self.chunk_index;

                ctx.removed_chunk_indices
                    .insert(chunk_index..chunk_index + n_chunks);
                *ctx.num_removed_chunks += n_chunks;
            }

            fn insert_before(&mut self, ctx: &mut Context<'a>, entry: SampleToChunkEntry) {
                ctx.insert_entries.push((self.index, entry));
                *ctx.num_inserted_entries += 1;
                self.index += 1;
            }
        }

        let mut entries = self.entries.iter_mut().enumerate().peekable();
        while let Some((entry_index, entry)) = entries.next() {
            let entry_index = entry_index + num_inserted_entries;
            let chunk_index = entry.first_chunk as usize - 1;

            let chunk_count = match entries.peek() {
                Some((_, next_entry)) => (next_entry.first_chunk - entry.first_chunk) as usize,
                None => total_chunks - chunk_index,
            };

            // ensure we're updating chunk indices after removing/inserting chunks
            entry.first_chunk -= num_removed_chunks as u32;

            let first_sample_index = next_sample_index;
            next_sample_index += entry.samples_per_chunk as usize * chunk_count;

            let mut ctx = Context {
                sample_indices_to_remove,

                removed_chunk_indices: &mut removed_chunk_indices,
                num_removed_chunks: &mut num_removed_chunks,

                remove_entry_range: &mut remove_entry_range,

                insert_entries: &mut insert_entries,
                num_inserted_entries: &mut num_inserted_entries,
            };

            Entry {
                index: entry_index,
                inner: entry,

                chunk_index,
                chunk_count,
                first_sample_index,
            }
            .process(&mut ctx);
        }

        let mut inserted_indices = Vec::with_capacity(insert_entries.len());

        for (insert_index, entry) in insert_entries {
            self.entries.insert(insert_index, entry);
            inserted_indices.push(insert_index);
        }

        // inserting redundant entries make the above logic easier to reason about
        // so we'll need to clean them up, looping backwards so we get the correct first_chunk
        for index in inserted_indices.into_iter().rev() {
            let entry = self.entries.get(index).expect("entry exists");
            match self.entries.get(index + 1) {
                Some(next_entry)
                    if next_entry.samples_per_chunk == entry.samples_per_chunk
                        && next_entry.sample_description_index
                            == entry.sample_description_index =>
                {
                    // remove redundant entry
                    remove_entry_range.insert(index + 1..index + 2);
                }
                _ => {}
            }
        }

        let mut n_removed = 0;
        for range in remove_entry_range.into_iter() {
            let mut range = (range.start - n_removed)..(range.end - n_removed);
            n_removed += range.len();

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
        let actual_removed_chunk_indices =
            stsc.remove_sample_indices(&sample_indices_to_remove, total_chunks);

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
