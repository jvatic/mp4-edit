use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};

use futures_io::AsyncRead;
use std::{
    fmt::{self, Debug},
    ops::{Bound, Range, RangeBounds, Sub},
};

use crate::{
    atom::{
        util::{read_to_end, DebugEllipsis, RangeCollection},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const STTS: &[u8; 4] = b"stts";

#[derive(Default, Clone, Deref, DerefMut)]
pub struct TimeToSampleEntries(Vec<TimeToSampleEntry>);

impl From<Vec<TimeToSampleEntry>> for TimeToSampleEntries {
    fn from(entries: Vec<TimeToSampleEntry>) -> Self {
        Self::new(entries)
    }
}

impl TimeToSampleEntries {
    pub fn new(inner: Vec<TimeToSampleEntry>) -> Self {
        Self(inner)
    }

    pub fn inner(&self) -> &[TimeToSampleEntry] {
        &self.0
    }
}

impl fmt::Debug for TimeToSampleEntries {
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

/// Defines duration for a consecutive group of samples
#[derive(Debug, Clone, PartialEq, Eq, Builder)]
pub struct TimeToSampleEntry {
    /// Number of consecutive samples with the same duration
    pub sample_count: u32,
    /// Duration of each sample in timescale units (see MDHD atom)
    pub sample_duration: u32,
}

/// Time-to-Sample (stts) atom
#[derive(Default, Debug, Clone)]
pub struct TimeToSampleAtom {
    pub version: u8,
    pub flags: [u8; 3],
    pub entries: TimeToSampleEntries,
}

#[bon]
impl TimeToSampleAtom {
    #[builder]
    pub fn new(
        #[builder(default = 0)] version: u8,
        #[builder(default = [0u8; 3])] flags: [u8; 3],
        #[builder(with = FromIterator::from_iter)] entries: Vec<TimeToSampleEntry>,
    ) -> Self {
        Self {
            version,
            flags,
            entries: entries.into(),
        }
    }

    /// Removes samples contained in the given `trim_duration`, excluding partially matched samples.
    /// Returns actual duration trimmed and indices of removed samples.
    ///
    /// # Panics
    ///
    /// This method panics if the trim ranges overlap.
    ///
    /// WARNING: failing to update other atoms appropriately will cause file corruption.
    pub(crate) fn trim_duration<R>(&mut self, trim_ranges: &[R]) -> (u64, Vec<Range<usize>>)
    where
        R: RangeBounds<u64> + Debug,
    {
        let mut trim_range_index = 0;
        let mut removed_sample_indices = RangeCollection::with_capacity(trim_ranges.len());
        let mut remove_entry_range = RangeCollection::new();
        let mut next_duration_offset = 0u64;
        let mut next_sample_index = 0usize;
        let mut total_duration_trimmed = 0u64;

        for (entry_index, entry) in self.entries.iter_mut().enumerate() {
            let current_duration_offset = next_duration_offset;
            next_duration_offset =
                current_duration_offset + entry.sample_count as u64 * entry.sample_duration as u64;

            let current_sample_index = next_sample_index;
            next_sample_index += entry.sample_count as usize;

            let entry_duration = {
                let entry_duration_start = current_duration_offset;
                let entry_duration_end = current_duration_offset
                    + (entry.sample_count as u64 * entry.sample_duration as u64).saturating_sub(1);
                entry_duration_start..entry_duration_end + 1
            };

            let (i, trim_duration, entry_trim_duration) =
                match entry_trim_duration(&entry_duration, trim_ranges) {
                    Some(m) => m,
                    None => {
                        // Entire entry is outside trim range
                        continue;
                    }
                };

            debug_assert!(
                i >= trim_range_index,
                "invariant: trim ranges must not overlap"
            );
            trim_range_index = i;

            // Entire entry is inside trim range
            if trim_duration.contains(&entry_duration.start)
                && trim_duration.contains(&(entry_duration.end - 1))
            {
                remove_entry_range.insert(entry_index..entry_index + 1);
                removed_sample_indices.insert(
                    current_sample_index..current_sample_index + entry.sample_count as usize,
                );
                total_duration_trimmed += entry_duration.end - entry_duration.start;
                continue;
            }

            // Partial overlap

            let sample_duration = entry.sample_duration as u64;

            let trim_sample_start_index = (current_sample_index as u64
                + (entry_trim_duration.start - entry_duration.start).div_ceil(sample_duration))
                as usize;
            let trim_sample_end_index =
                match (entry_trim_duration.end - entry_duration.start) / sample_duration {
                    0 => trim_sample_start_index,
                    end => current_sample_index + end as usize - 1,
                };

            removed_sample_indices.insert(trim_sample_start_index..(trim_sample_end_index + 1));

            let num_samples_to_remove = trim_sample_end_index + 1 - trim_sample_start_index;
            entry.sample_count = entry.sample_count.sub(num_samples_to_remove as u32);

            total_duration_trimmed += ((trim_sample_end_index as u64 + 1) * sample_duration)
                - (trim_sample_start_index as u64 * sample_duration);
        }

        for mut range in remove_entry_range.into_iter() {
            // maybe merge entries before and after the removed ones
            if range.start > 0 {
                if let Ok([prev_entry, next_entry]) = self
                    .entries
                    .as_mut_slice()
                    .get_disjoint_mut([range.start - 1, range.end])
                {
                    if prev_entry.sample_duration == next_entry.sample_duration {
                        prev_entry.sample_count += next_entry.sample_count;
                        range.end += 1;
                    }
                }
            }

            self.entries.drain(range);
        }

        (
            total_duration_trimmed,
            removed_sample_indices.into_iter().collect(),
        )
    }
}

fn entry_trim_duration<'a, R>(
    entry_range: &Range<u64>,
    trim_range: &'a [R],
) -> Option<(usize, &'a R, Range<u64>)>
where
    R: RangeBounds<u64>,
{
    for (i, trim_range) in trim_range.iter().enumerate() {
        // entry is contained in range
        if trim_range.contains(&entry_range.start) && trim_range.contains(&(entry_range.end - 1)) {
            return Some((i, trim_range, entry_range.clone()));
        }

        let finite_trim_range = convert_range(entry_range, trim_range);

        // trim range is contained in entry
        if entry_range.contains(&finite_trim_range.start)
            && finite_trim_range.end > 0
            && entry_range.contains(&(finite_trim_range.end - 1))
        {
            return Some((i, trim_range, finite_trim_range));
        }

        // trim range starts inside of entry
        if finite_trim_range.start >= entry_range.start && finite_trim_range.start < entry_range.end
        {
            return Some((i, trim_range, finite_trim_range.start..entry_range.end));
        }

        // trim range ends inside of entry
        if trim_range.contains(&entry_range.start)
            && finite_trim_range.start < entry_range.start
            && finite_trim_range.end <= entry_range.end
        {
            return Some((i, trim_range, entry_range.start..finite_trim_range.end));
        }
    }

    None
}

fn convert_range(reference_range: &Range<u64>, range: &impl RangeBounds<u64>) -> Range<u64> {
    let start = match range.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => *start + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(end) => *end + 1,
        Bound::Excluded(end) => *end,
        Bound::Unbounded => reference_range.end,
    };
    start..end
}

impl<S: time_to_sample_atom_builder::State> TimeToSampleAtomBuilder<S> {
    pub fn entry(
        self,
        entry: impl Into<TimeToSampleEntry>,
    ) -> TimeToSampleAtomBuilder<time_to_sample_atom_builder::SetEntries<S>>
    where
        S::Entries: time_to_sample_atom_builder::IsUnset,
    {
        self.entries(vec![entry.into()])
    }
}

impl From<Vec<TimeToSampleEntry>> for TimeToSampleAtom {
    fn from(entries: Vec<TimeToSampleEntry>) -> Self {
        TimeToSampleAtom {
            version: 0,
            flags: [0u8; 3],
            entries: entries.into(),
        }
    }
}

impl ParseAtom for TimeToSampleAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STTS {
            return Err(ParseError::new_unexpected_atom(atom_type, STTS));
        }
        let data = read_to_end(reader).await?;
        parser::parse_stts_data(&data)
    }
}

impl SerializeAtom for TimeToSampleAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STTS)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_mdhd_atom(self)
    }
}

mod serializer {
    use crate::atom::util::serializer::be_u32;

    use super::TimeToSampleAtom;

    pub fn serialize_mdhd_atom(stts: TimeToSampleAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(stts.version);
        data.extend(stts.flags);
        data.extend(be_u32(
            u32::try_from(stts.entries.len()).expect("stts entries len must fit in u32"),
        ));

        for entry in stts.entries.0.into_iter() {
            data.extend(entry.sample_count.to_be_bytes());
            data.extend(entry.sample_duration.to_be_bytes());
        }

        data
    }
}

mod parser {
    use winnow::{
        binary::{be_u32, length_repeat},
        combinator::{seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::{TimeToSampleAtom, TimeToSampleEntries, TimeToSampleEntry};
    use crate::atom::util::parser::{flags3, stream, version, Stream};

    pub fn parse_stts_data(input: &[u8]) -> Result<TimeToSampleAtom, crate::ParseError> {
        parse_stts_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_stts_data_inner(input: &mut Stream<'_>) -> ModalResult<TimeToSampleAtom> {
        trace(
            "stts",
            seq!(TimeToSampleAtom {
                version: version,
                flags: flags3,
                entries: length_repeat(be_u32, entry)
                    .map(TimeToSampleEntries)
                    .context(StrContext::Label("entries")),
            })
            .context(StrContext::Label("stts")),
        )
        .parse_next(input)
    }

    fn entry(input: &mut Stream<'_>) -> ModalResult<TimeToSampleEntry> {
        trace(
            "entry",
            seq!(TimeToSampleEntry {
                sample_count: be_u32.context(StrContext::Label("sample_count")),
                sample_duration: be_u32.context(StrContext::Label("sample_duration")),
            })
            .context(StrContext::Label("entry")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;

    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available stts test data files
    #[test]
    fn test_stts_roundtrip() {
        test_atom_roundtrip_sync::<TimeToSampleAtom>(STTS);
    }

    struct TrimDurationTestCase {
        trim_duration: Vec<(Bound<u64>, Bound<u64>)>,
        expect_removed_samples: Vec<Range<usize>>,
        expect_removed_duration: u64,
        expect_entries: Vec<TimeToSampleEntry>,
    }

    fn test_trim_duration_stts() -> TimeToSampleAtom {
        TimeToSampleAtom::builder()
            .entries(vec![
                // samples  0..1
                // duration 0..100
                TimeToSampleEntry {
                    sample_count: 1,
                    sample_duration: 100,
                },
                // samples 1..5
                // duration 100..900
                TimeToSampleEntry {
                    sample_count: 4,
                    sample_duration: 200,
                },
                // samples 5..9
                // duration 900..1300
                TimeToSampleEntry {
                    sample_count: 4,
                    sample_duration: 100,
                },
            ])
            .build()
    }

    fn test_trim_duration<F>(mut stts: TimeToSampleAtom, test_case: F)
    where
        F: FnOnce(&TimeToSampleAtom) -> TrimDurationTestCase,
    {
        let test_case = test_case(&stts);
        let (actual_removed_duration, actual_removed_samples) =
            stts.trim_duration(&test_case.trim_duration);
        assert_eq!(
            actual_removed_samples, test_case.expect_removed_samples,
            "removed sample indices don't match what's expected"
        );
        assert_eq!(
            actual_removed_duration, test_case.expect_removed_duration,
            "removed duration doesn't match what's expected"
        );
        assert_eq!(
            stts.entries.0, test_case.expect_entries,
            "time to sample entries don't match what's expected"
        )
    }

    macro_rules! test_trim_duration {
        ($($name:ident $(($stts:expr))? => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let stts = test_trim_duration!(@get_stts $($stts)?);
                    test_trim_duration(stts, $test_case);
                }
            )*
        };

        (@get_stts $stts:expr) => { $stts };
        (@get_stts) => { test_trim_duration_stts() };
    }

    test_trim_duration!(
        trim_first_entry_unbounded_start => |stts| TrimDurationTestCase {
            trim_duration: vec![(Bound::Unbounded, Bound::Excluded(100))],
            expect_removed_samples: vec![0..1],
            expect_removed_duration: 100,
            expect_entries: stts.entries[1..].to_vec(),
        },
        trim_first_entry_included_start => |stts| TrimDurationTestCase {
            trim_duration: vec![(Bound::Included(0), Bound::Excluded(100))],
            expect_removed_samples: vec![0..1],
            expect_removed_duration: 100,
            expect_entries: stts.entries[1..].to_vec(),
        },
        trim_last_sample_unbounded_end => |stts| {
            let mut expect_entries = stts.entries.clone().0;
            expect_entries.last_mut().unwrap().sample_count = 3;
            TrimDurationTestCase {
                trim_duration: vec![(Bound::Included(1_200), Bound::Unbounded)],
                expect_removed_duration: 100,
                expect_removed_samples: vec![8..9],
                expect_entries,
            }
        },
        trim_last_three_samples_unbounded_end => |stts| {
            let mut expect_entries = stts.entries.clone().0;
            expect_entries.last_mut().unwrap().sample_count = 1;
            TrimDurationTestCase {
                trim_duration: vec![(Bound::Included(1_000), Bound::Unbounded)],
                expect_removed_duration: 300,
                expect_removed_samples: vec![6..9],
                expect_entries,
            }
        },
        trim_last_sample_included_end => |stts| {
            let mut expect_entries = stts.entries.clone().0;
            expect_entries.last_mut().unwrap().sample_count = 3;
            TrimDurationTestCase {
                trim_duration: vec![(Bound::Included(1_200), Bound::Included(1_300 - 1))],
                expect_removed_duration: 100,
                expect_removed_samples: vec![8..9],
                expect_entries,
            }
        },
        trim_middle_entry_excluded_end => |_| TrimDurationTestCase {
            trim_duration: vec![(Bound::Included(100), Bound::Excluded(900))],
            expect_removed_duration: 800,
            expect_removed_samples: vec![1..5],
            expect_entries: vec![
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 100,
                },
            ],
        },
        trim_middle_entry_excluded_start => |_| TrimDurationTestCase {
            trim_duration: vec![(Bound::Excluded(99), Bound::Excluded(900))],
            expect_removed_duration: 800,
            expect_removed_samples: vec![1..5],
            expect_entries: vec![
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 100,
                },
            ],
        },
        trim_middle_entry_excluded_start_included_end => |_| TrimDurationTestCase {
            trim_duration: vec![(Bound::Excluded(99), Bound::Included(899))],
            expect_removed_duration: 800,
            expect_removed_samples: vec![1..5],
            expect_entries: vec![
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 100,
                },
            ],
        },
        trim_middle_samples => |stts| TrimDurationTestCase {
            // entry 1 samples:
            //  sample index 1 starts at 100 (not trimmed)
            //  sample index 2 starts at 300 (trimmed)
            //  sample index 3 starts at 500 (trimmed)
            //  sample index 4 starts at 700 (not trimmed)
            trim_duration: vec![(Bound::Included(300), Bound::Excluded(700))],
            expect_removed_duration: 400,
            expect_removed_samples: vec![2..4],
            expect_entries: vec![
                stts.entries[0].clone(),
                TimeToSampleEntry {
                    sample_count: 2,
                    sample_duration: 200,
                },
                stts.entries[2].clone(),
            ],
        },
        trim_middle_samples_partial => |stts| TrimDurationTestCase {
            // partially matching samples should be left intact
            // entry 1 samples:
            //  sample index 1 starts at 100 (partially matched, not trimmed)
            //  sample index 2 starts at 300 (trimmed)
            //  sample index 3 starts at 500 (trimmed)
            //  sample index 4 starts at 700 (partially matched, not trimmed)
            trim_duration: vec![(Bound::Included(240), Bound::Excluded(850))],
            expect_removed_duration: 400,
            expect_removed_samples: vec![2..4],
            expect_entries: vec![
                stts.entries[0].clone(),
                TimeToSampleEntry {
                    sample_count: 2,
                    sample_duration: 200,
                },
                stts.entries[2].clone(),
            ],
        },
        trim_everything => |_| TrimDurationTestCase {
            trim_duration: vec![(Bound::Unbounded, Bound::Unbounded)],
            expect_removed_duration: 1_300,
            expect_removed_samples: vec![0..9],
            expect_entries: Vec::new(),
        },
        trim_middle_from_large_entry ({
            TimeToSampleAtom::builder().entry(
                // samples  0..10
                // duration 0..10_000
                //
                // sample 0 => 0
                // sample 1 => 10_000
                // sample 2 => 20_000 (trimmed)
                // sample 3 => 30_000 (trimmed)
                // sample 4 => 40_000 (trimmed)
                // sample 5 => 50_000
                // ...
                TimeToSampleEntry {
                    sample_count: 10,
                    sample_duration: 10_000,
                },
            ).build()
        }) => |stts| TrimDurationTestCase {
            trim_duration: vec![(Bound::Excluded(19_999), Bound::Included(50_000))],
            expect_removed_duration: 30_000,
            expect_removed_samples: vec![2..5],
            expect_entries: stts.entries.iter().cloned().map(|mut entry| {
                entry.sample_count = 7;
                entry
            }).collect::<Vec<_>>(),
        },
        trim_start_and_end => |stts| {
            let mut expect_entries = stts.entries[1..].to_vec();
            expect_entries.last_mut().unwrap().sample_count = 1;
            TrimDurationTestCase {
                trim_duration: vec![
                    (Bound::Included(0), Bound::Excluded(100)),
                    (Bound::Included(1_000), Bound::Excluded(1_300)),
                ],
                expect_removed_duration: 100 + 300,
                expect_removed_samples: vec![0..1, 6..9],
                expect_entries,
            }
        },
    );
}
