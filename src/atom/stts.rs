use anyhow::{anyhow, Context};
use bon::{bon, Builder};
use derive_more::{Deref, DerefMut};

use futures_io::AsyncRead;
use std::{fmt, io::Read, ops::RangeBounds};

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
        #[builder(field = Vec::new())] entries: Vec<TimeToSampleEntry>,
        #[builder(default = 0)] version: u8,
        #[builder(default = [0u8; 3])] flags: [u8; 3],
        #[builder(setters(vis = ""), overwritable)]
        #[allow(unused)]
        entries_marker: bool,
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
    pub fn trim_samples(&mut self, trim_duration: impl RangeBounds<u64>) -> Vec<usize> {
        // Convert range bounds to concrete values for easier math
        let trim_start = match trim_duration.start_bound() {
            std::ops::Bound::Included(&start) => start,
            std::ops::Bound::Excluded(&start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let trim_end = match trim_duration.end_bound() {
            std::ops::Bound::Included(&end) => end,
            std::ops::Bound::Excluded(&end) => end.saturating_sub(1),
            std::ops::Bound::Unbounded => u64::MAX,
        };

        let mut new_entries = Vec::new();
        let mut removed_sample_indices = Vec::new();
        let mut current_duration_offset = 0u64;
        let mut current_sample_index = 0usize;

        for entry in self.entries.iter() {
            let entry_duration_start = current_duration_offset;
            let entry_duration_end = current_duration_offset
                + (entry.sample_count as u64 * entry.sample_duration as u64).saturating_sub(1);

            // Fast path: entire entry is outside trim range
            if entry_duration_end < trim_start || entry_duration_start > trim_end {
                new_entries.push(entry.clone());
                current_duration_offset += entry.sample_count as u64 * entry.sample_duration as u64;
                current_sample_index += entry.sample_count as usize;
                continue;
            }

            // Fast path: entire entry is inside trim range
            if entry_duration_start >= trim_start && entry_duration_end <= trim_end {
                // Remove all samples from this entry
                for i in 0..entry.sample_count {
                    removed_sample_indices.push(current_sample_index + i as usize);
                }
                current_duration_offset += entry.sample_count as u64 * entry.sample_duration as u64;
                current_sample_index += entry.sample_count as usize;
                continue;
            }

            // Partial overlap: calculate which samples to remove using range arithmetic
            let mut kept_samples = 0u32;
            let sample_duration = entry.sample_duration as u64;

            for sample_idx in 0..entry.sample_count {
                let sample_start = current_duration_offset + (sample_idx as u64 * sample_duration);
                let sample_end = sample_start + sample_duration - 1;

                // Sample overlaps with trim range if there's any intersection
                let overlaps = sample_start <= trim_end && sample_end >= trim_start;

                if overlaps {
                    removed_sample_indices.push(current_sample_index + sample_idx as usize);
                } else {
                    kept_samples += 1;
                }
            }

            // Add entry for kept samples if any
            if kept_samples > 0 {
                new_entries.push(TimeToSampleEntry {
                    sample_count: kept_samples,
                    sample_duration: entry.sample_duration,
                });
            }

            current_duration_offset += entry.sample_count as u64 * entry.sample_duration as u64;
            current_sample_index += entry.sample_count as usize;
        }

        self.entries = new_entries.into();
        removed_sample_indices
    }

    /// Removes samples from the end and returns the number of samples removed.
    ///
    /// `duration_to_trim` is in media timescale units (in [`crate::atom::MovieHeaderAtom`]).
    ///
    /// WARNING: failing to update other atoms appropriately will cause file corruption.
    pub fn trim_samples_from_end(&mut self, duration_to_trim: u64) -> u32 {
        let (entries_to_remove, samples_removed) =
            trim_samples(self.entries.iter_mut().rev(), duration_to_trim);

        // Remove completely consumed entries from the end
        let new_len = self.entries.len().saturating_sub(entries_to_remove);
        self.entries.truncate(new_len);

        samples_removed
    }
}

impl<S: time_to_sample_atom_builder::State> TimeToSampleAtomBuilder<S> {
    pub fn entry(
        mut self,
        entry: impl Into<TimeToSampleEntry>,
    ) -> TimeToSampleAtomBuilder<time_to_sample_atom_builder::SetEntriesMarker<S>> {
        self.entries.push(entry.into());
        self.entries_marker(true)
    }

    pub fn entries(
        mut self,
        entries: impl Into<Vec<TimeToSampleEntry>>,
    ) -> TimeToSampleAtomBuilder<time_to_sample_atom_builder::SetEntriesMarker<S>>
    where
        S::EntriesMarker: time_to_sample_atom_builder::IsUnset,
    {
        self.entries = entries.into();
        self.entries_marker(true)
    }
}

fn trim_samples<'a>(
    entries: impl Iterator<Item = &'a mut TimeToSampleEntry>,
    duration_to_trim: u64,
) -> (usize, u32) {
    let mut time_trimmed = 0u64;
    let mut samples_removed = 0u32;
    let mut entries_to_remove = 0usize;

    for entry in entries {
        let entry_total_duration =
            u64::from(entry.sample_count).saturating_mul(u64::from(entry.sample_duration));

        if time_trimmed.saturating_add(entry_total_duration) <= duration_to_trim {
            // Remove this entire entry
            time_trimmed = time_trimmed.saturating_add(entry_total_duration);
            samples_removed = samples_removed.saturating_add(entry.sample_count);
            entries_to_remove = entries_to_remove.saturating_add(1);
        } else {
            // Partial removal from this entry
            let remaining_duration = duration_to_trim.saturating_sub(time_trimmed);
            let samples_to_remove = if entry.sample_duration == 0 {
                0u32
            } else {
                (remaining_duration / u64::from(entry.sample_duration)) as u32
            };

            samples_removed = samples_removed.saturating_add(samples_to_remove);

            if samples_to_remove > 0 {
                entry.sample_count = entry.sample_count.saturating_sub(samples_to_remove);
            }
            break;
        }
    }

    (entries_to_remove, samples_removed)
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
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available stts test data files
    #[test]
    fn test_stts_roundtrip() {
        test_atom_roundtrip_sync::<TimeToSampleAtom>(STTS);
    }
}
