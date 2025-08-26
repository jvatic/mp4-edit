use anyhow::{anyhow, Context};
use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};

use futures_io::AsyncRead;
use std::{
    fmt::{self, Debug},
    io::Read,
    ops::{Bound, Div, Range, RangeBounds, RangeInclusive, Sub},
};

use crate::{
    atom::{
        util::{async_to_sync_read, DebugEllipsis},
        FourCC,
    },
    parser::Parse,
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

/// Time-to-Sample entry - defines duration for a consecutive group of samples
#[derive(Debug, Clone, PartialEq, Eq, Builder)]
pub struct TimeToSampleEntry {
    /// Number of consecutive samples with the same duration
    pub sample_count: u32,
    /// Duration of each sample in time units (timescale units)
    pub sample_duration: u32,
}

/// Time-to-Sample Atom - contains time-to-sample mapping table
#[derive(Default, Debug, Clone)]
pub struct TimeToSampleAtom {
    /// Version of the stts atom format (0)
    pub version: u8,
    /// Flags for the stts atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of time-to-sample entries
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

    /// Removes samples contained in the given `trim_duration` and returns indices of removed samples.
    ///
    /// WARNING: failing to update other atoms appropriately will cause file corruption.
    pub fn trim_duration<R>(&mut self, trim_duration: R) -> Vec<Range<usize>>
    where
        R: RangeBounds<u64> + Debug,
    {
        // removed sample indices will be contiguous
        let mut removed_sample_indices: Option<Range<usize>> = None;
        // removed entries will be contiguous
        let mut remove_entry_range: Option<RangeInclusive<usize>> = None;
        let mut current_duration_offset = 0u64;
        let mut current_sample_index = 0usize;

        for (entry_index, entry) in self.entries.iter_mut().enumerate() {
            let entry_duration_start = current_duration_offset;
            let entry_duration_end = current_duration_offset
                + (entry.sample_count as u64 * entry.sample_duration as u64).saturating_sub(1);

            let next_duration_offset =
                current_duration_offset + entry.sample_count as u64 * entry.sample_duration as u64;

            // Entire entry is outside trim range
            let entry_duration_range = entry_duration_start..entry_duration_end + 1;

            let entry_trim_duration = entry_trim_duration(&entry_duration_range, &trim_duration);

            // Entry is not in trim range
            if entry_trim_duration.is_empty() {
                current_duration_offset = next_duration_offset;
                current_sample_index += entry.sample_count as usize;
                continue;
            }

            // Entire entry is inside trim range
            if trim_duration.contains(&entry_duration_start)
                && trim_duration.contains(&entry_duration_end)
            {
                remove_entry_range = Some(match remove_entry_range {
                    Some(range) => {
                        debug_assert_eq!(
                            *range.end(),
                            entry_index.saturating_sub(1),
                            "invariant: non-contiguous entry index range"
                        );
                        (*range.start())..=entry_index
                    }
                    None => entry_index..=entry_index,
                });
                removed_sample_indices = Some(match removed_sample_indices {
                    Some(range) => {
                        debug_assert_eq!(
                            range.end, current_sample_index,
                            "invariant: non-contiguous sample index range"
                        );
                        range.start..(current_sample_index + entry.sample_count as usize)
                    }
                    None => {
                        current_sample_index..(current_sample_index + entry.sample_count as usize)
                    }
                });
                current_duration_offset = next_duration_offset;
                current_sample_index += entry.sample_count as usize;
                continue;
            }

            // Partial overlap

            let sample_duration = entry.sample_duration as u64;

            let trim_sample_start_index = (current_sample_index as u64
                + (entry_trim_duration.start - entry_duration_range.start)
                    .div_ceil(sample_duration)) as usize;
            let trim_sample_end_index = match ((entry_trim_duration.end / sample_duration)
                * sample_duration)
                - entry_duration_range.start
            {
                0 => trim_sample_start_index,
                end => current_sample_index + end.div(sample_duration) as usize - 1,
            };

            removed_sample_indices = Some(match removed_sample_indices {
                Some(range) => {
                    debug_assert_eq!(
                        range.end,
                        (trim_sample_start_index as usize),
                        "invariant: non-contiguous sample index range"
                    );
                    range.start..(trim_sample_end_index as usize + 1)
                }
                None => (trim_sample_start_index as usize)..(trim_sample_end_index as usize + 1),
            });

            let samples_to_remove = trim_sample_end_index + 1 - trim_sample_start_index;
            entry.sample_count = entry.sample_count.sub(samples_to_remove as u32);

            current_duration_offset = next_duration_offset;
            current_sample_index += entry.sample_count as usize;
        }

        if let Some(mut range) = remove_entry_range {
            // maybe merge entries before and after the removed ones
            if *range.start() > 0 {
                let mut prev_entry_sample_count = None;
                let prev_entry_index = *range.start() - 1;
                let next_entry_index = *range.end() + 1;
                if let Some(entry_prev) = self.entries.get(prev_entry_index) {
                    if let Some(entry_next) = self.entries.get(next_entry_index) {
                        if entry_prev.sample_duration == entry_next.sample_duration {
                            range = (*range.start())..=next_entry_index;
                            prev_entry_sample_count =
                                Some(entry_prev.sample_count + entry_next.sample_count);
                        }
                    }
                }

                if let Some(prev_entry_sample_count) = prev_entry_sample_count {
                    let prev_entry = self.entries.get_mut(prev_entry_index).unwrap();
                    prev_entry.sample_count = prev_entry_sample_count;
                }
            }

            self.entries.drain(range);
        }

        removed_sample_indices.map_or_else(|| Vec::new(), |r| vec![r])
    }
}

fn entry_trim_duration(entry_range: &Range<u64>, trim_range: &impl RangeBounds<u64>) -> Range<u64> {
    // entry is contained in range
    if trim_range.contains(&entry_range.start) && trim_range.contains(&(entry_range.end - 1)) {
        return entry_range.clone();
    }

    let finite_trim_range = convert_range(entry_range, trim_range);

    // trim range is contained in entry
    if entry_range.contains(&finite_trim_range.start)
        && finite_trim_range.end > 0
        && entry_range.contains(&(finite_trim_range.end - 1))
    {
        return finite_trim_range;
    }

    // trim range starts inside of entry
    if finite_trim_range.start >= entry_range.start && finite_trim_range.start < entry_range.end {
        return finite_trim_range.start..entry_range.end;
    }

    // trim range ends inside of entry
    if trim_range.contains(&entry_range.start)
        && finite_trim_range.start < entry_range.start
        && finite_trim_range.end <= entry_range.end
    {
        return entry_range.start..finite_trim_range.end;
    }

    0..0
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

impl Parse for TimeToSampleAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STTS {
            return Err(ParseError::new_unexpected_atom(atom_type, STTS));
        }
        let mut cursor = async_to_sync_read(reader).await?;
        parse_stts_data(&mut cursor).map_err(ParseError::new_atom_parse)
    }
}

fn parse_stts_data<R: Read>(mut reader: R) -> Result<TimeToSampleAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version
    if version != 0 {
        return Err(anyhow!("unsupported version {}", version));
    }

    // Read entry count
    let mut count_buf = [0u8; 4];
    reader
        .read_exact(&mut count_buf)
        .context("read entry count")?;
    let entry_count = u32::from_be_bytes(count_buf);

    // Validate entry count (reasonable limit to prevent memory exhaustion)
    if entry_count > 1_000_000 {
        return Err(anyhow!("Too many time-to-sample entries: {}", entry_count));
    }

    let mut entries = Vec::with_capacity(entry_count as usize);

    for i in 0..entry_count {
        // Read sample count (4 bytes)
        let mut sample_count_buf = [0u8; 4];
        reader
            .read_exact(&mut sample_count_buf)
            .context(format!("read sample count for entry {i}"))?;
        let sample_count = u32::from_be_bytes(sample_count_buf);

        // Read sample duration (4 bytes)
        let mut sample_duration_buf = [0u8; 4];
        reader
            .read_exact(&mut sample_duration_buf)
            .context(format!("read sample duration for entry {i}"))?;
        let sample_duration = u32::from_be_bytes(sample_duration_buf);

        entries.push(TimeToSampleEntry {
            sample_count,
            sample_duration,
        });
    }

    Ok(TimeToSampleAtom {
        version,
        flags,
        entries: TimeToSampleEntries(entries),
    })
}

impl SerializeAtom for TimeToSampleAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STTS)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version and flags (4 bytes)
        let version_flags = u32::from(self.version) << 24
            | u32::from(self.flags[0]) << 16
            | u32::from(self.flags[1]) << 8
            | u32::from(self.flags[2]);
        data.extend_from_slice(&version_flags.to_be_bytes());

        // Entry count (4 bytes)
        data.extend_from_slice(&(self.entries.len() as u32).to_be_bytes());

        // Entries (8 bytes each: sample_count + sample_duration)
        for entry in self.entries.iter() {
            data.extend_from_slice(&entry.sample_count.to_be_bytes());
            data.extend_from_slice(&entry.sample_duration.to_be_bytes());
        }

        data
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

    fn test_trim_duration<F>(test_case: F)
    where
        F: FnOnce(&TimeToSampleAtom) -> TrimDurationTestCase,
    {
        let mut stts = test_trim_duration_stts();
        let test_case = test_case(&stts);
        let actual_removed_samples = stts.trim_duration(test_case.trim_duration[0]);
        assert_eq!(
            actual_removed_samples, test_case.expect_removed_samples,
            "removed sample indices don't match what's expected"
        );
        assert_eq!(
            stts.entries.0, test_case.expect_entries,
            "time to sample entries don't match what's expected"
        )
    }

    macro_rules! test_trim_duration {
        ($($name:ident => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    test_trim_duration($test_case);
                }
            )*
        };
    }

    test_trim_duration!(
        trim_first_entry_unbounded_start => |stts| TrimDurationTestCase {
            trim_duration: vec![(Bound::Unbounded, Bound::Excluded(100))],
            expect_removed_samples: vec![0..1],
            expect_entries: stts.entries.as_slice()[1..].to_vec(),
        },
        trim_first_entry_inclusive_start => |stts| TrimDurationTestCase {
            trim_duration: vec![(Bound::Included(0), Bound::Excluded(100))],
            expect_removed_samples: vec![0..1],
            expect_entries: stts.entries.as_slice()[1..].to_vec(),
        },
        trim_last_sample_unbounded_end => |stts| {
            let mut expect_entries = stts.entries.clone().0;
            expect_entries.last_mut().unwrap().sample_count = 3;
            TrimDurationTestCase {
                trim_duration: vec![(Bound::Included(1_200), Bound::Unbounded)],
                expect_removed_samples: vec![8..9],
                expect_entries,
            }
        },
        trim_last_three_samples_unbounded_end => |stts| {
            let mut expect_entries = stts.entries.clone().0;
            expect_entries.last_mut().unwrap().sample_count = 1;
            TrimDurationTestCase {
                trim_duration: vec![(Bound::Included(1_000), Bound::Unbounded)],
                expect_removed_samples: vec![6..9],
                expect_entries,
            }
        },
        // TODO: trim_last_entry_unbounded_end
        trim_last_sample_inclusive_end => |stts| {
            let mut expect_entries = stts.entries.clone().0;
            expect_entries.last_mut().unwrap().sample_count = 3;
            TrimDurationTestCase {
                trim_duration: vec![(Bound::Included(1_200), Bound::Included(1_300 - 1))],
                expect_removed_samples: vec![8..9],
                expect_entries,
            }
        },
        trim_middle_entry => |_| TrimDurationTestCase {
            trim_duration: vec![(Bound::Included(100), Bound::Excluded(900))],
            expect_removed_samples: vec![1..5],
            expect_entries: vec![
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 100,
                },
            ],
        },
        trim_middle_samples => |stts| TrimDurationTestCase {
            trim_duration: vec![(Bound::Included(300), Bound::Excluded(700))],
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
            trim_duration: vec![(Bound::Included(250), Bound::Excluded(750))],
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
            expect_removed_samples: vec![0..9],
            expect_entries: Vec::new(),
        },
    );
}
