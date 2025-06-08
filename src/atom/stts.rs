use anyhow::{anyhow, Context};
use derive_more::Deref;
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis};

pub const STTS: &[u8; 4] = b"stts";

#[derive(Clone, Deref)]
pub struct TimeToSampleEntries(Vec<TimeToSampleEntry>);

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeToSampleEntry {
    /// Number of consecutive samples with the same duration
    pub sample_count: u32,
    /// Duration of each sample in time units (timescale units)
    pub sample_duration: u32,
}

/// Time-to-Sample Atom - contains time-to-sample mapping table
#[derive(Debug, Clone)]
pub struct TimeToSampleAtom {
    /// Version of the stts atom format (0)
    pub version: u8,
    /// Flags for the stts atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of time-to-sample entries
    pub entries: TimeToSampleEntries,
}

impl TimeToSampleAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_time_to_sample_atom(reader)
    }

    /// Get the number of time-to-sample entries
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the total number of samples across all entries
    pub fn total_sample_count(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.sample_count as u64)
            .sum()
    }

    /// Get the total duration in timescale units
    pub fn total_duration(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.sample_count as u64 * entry.sample_duration as u64)
            .sum()
    }

    /// Convert time to sample number (0-based)
    /// Returns None if time is beyond the total duration
    pub fn time_to_sample(&self, time: u64) -> Option<u64> {
        if self.entries.is_empty() {
            return None;
        }

        let mut current_time = 0u64;
        let mut current_sample = 0u64;

        for entry in self.entries.iter() {
            let entry_duration = entry.sample_count as u64 * entry.sample_duration as u64;

            if time < current_time + entry_duration {
                // Time falls within this entry
                let time_in_entry = time - current_time;
                let sample_in_entry = time_in_entry / entry.sample_duration as u64;
                return Some(current_sample + sample_in_entry);
            }

            current_time += entry_duration;
            current_sample += entry.sample_count as u64;
        }

        None // Time is beyond total duration
    }

    /// Convert sample number to time (0-based sample index)
    /// Returns None if sample is beyond the total sample count
    pub fn sample_to_time(&self, sample: u64) -> Option<u64> {
        if self.entries.is_empty() {
            return None;
        }

        let mut current_time = 0u64;
        let mut current_sample = 0u64;

        for entry in self.entries.iter() {
            let next_sample = current_sample + entry.sample_count as u64;

            if sample < next_sample {
                // Sample falls within this entry
                let sample_in_entry = sample - current_sample;
                return Some(current_time + sample_in_entry * entry.sample_duration as u64);
            }

            current_time += entry.sample_count as u64 * entry.sample_duration as u64;
            current_sample = next_sample;
        }

        None // Sample is beyond total count
    }

    /// Get the duration of a specific sample
    pub fn get_sample_duration(&self, sample: u64) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }

        let mut current_sample = 0u64;

        for entry in self.entries.iter() {
            let next_sample = current_sample + entry.sample_count as u64;

            if sample < next_sample {
                return Some(entry.sample_duration);
            }

            current_sample = next_sample;
        }

        None
    }

    /// Get time range for a specific sample (start_time, end_time)
    pub fn get_sample_time_range(&self, sample: u64) -> Option<(u64, u64)> {
        let start_time = self.sample_to_time(sample)?;
        let duration = self.get_sample_duration(sample)?;
        Some((start_time, start_time + duration as u64))
    }

    /// Get samples within a time range
    pub fn get_samples_in_time_range(&self, start_time: u64, end_time: u64) -> Vec<u64> {
        let start_sample = self.time_to_sample(start_time).unwrap_or(0);
        let mut samples = Vec::new();

        let mut sample = start_sample;
        while let Some(sample_time) = self.sample_to_time(sample) {
            if sample_time >= end_time {
                break;
            }
            if sample_time >= start_time {
                samples.push(sample);
            }
            sample += 1;
        }

        samples
    }

    /// Check if all samples have the same duration (constant frame rate)
    pub fn is_constant_duration(&self) -> bool {
        if self.entries.is_empty() {
            return true;
        }

        let first_duration = self.entries[0].sample_duration;
        self.entries
            .iter()
            .all(|entry| entry.sample_duration == first_duration)
    }

    /// Get the most common sample duration
    pub fn most_common_duration(&self) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }

        // Count samples for each duration
        let mut duration_counts: std::collections::HashMap<u32, u64> =
            std::collections::HashMap::new();

        for entry in self.entries.iter() {
            *duration_counts.entry(entry.sample_duration).or_insert(0) += entry.sample_count as u64;
        }

        duration_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(duration, _)| duration)
    }

    /// Get statistics about the time-to-sample table
    pub fn get_statistics(&self) -> TimeToSampleStatistics {
        if self.entries.is_empty() {
            return TimeToSampleStatistics::default();
        }

        let total_samples = self.total_sample_count();
        let total_duration = self.total_duration();
        let entry_count = self.entries.len();

        let min_duration = self
            .entries
            .iter()
            .map(|e| e.sample_duration)
            .min()
            .unwrap_or(0);
        let max_duration = self
            .entries
            .iter()
            .map(|e| e.sample_duration)
            .max()
            .unwrap_or(0);

        let average_duration = if total_samples > 0 {
            (total_duration as f64 / total_samples as f64) as u32
        } else {
            0
        };

        let is_constant = self.is_constant_duration();
        let most_common = self.most_common_duration().unwrap_or(0);

        TimeToSampleStatistics {
            entry_count,
            total_samples,
            total_duration,
            min_duration,
            max_duration,
            average_duration,
            most_common_duration: most_common,
            is_constant_duration: is_constant,
        }
    }

    /// Calculate frame rate if this represents video samples
    /// Returns frames per second based on timescale
    pub fn calculate_frame_rate(&self, timescale: u32) -> Option<f64> {
        if self.entries.is_empty() || timescale == 0 {
            return None;
        }

        let most_common_duration = self.most_common_duration()?;
        Some(timescale as f64 / most_common_duration as f64)
    }

    /// Validate the time-to-sample table for consistency
    pub fn validate(&self) -> Result<(), String> {
        if self.entries.is_empty() {
            return Ok(());
        }

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.sample_count == 0 {
                return Err(format!("Entry {} has zero sample count", i));
            }
            if entry.sample_duration == 0 {
                return Err(format!("Entry {} has zero sample duration", i));
            }
        }

        Ok(())
    }
}

/// Statistics about the time-to-sample table
#[derive(Debug, Clone, Default)]
pub struct TimeToSampleStatistics {
    /// Number of entries in the table
    pub entry_count: usize,
    /// Total number of samples
    pub total_samples: u64,
    /// Total duration in timescale units
    pub total_duration: u64,
    /// Minimum sample duration
    pub min_duration: u32,
    /// Maximum sample duration
    pub max_duration: u32,
    /// Average sample duration
    pub average_duration: u32,
    /// Most common sample duration
    pub most_common_duration: u32,
    /// Whether all samples have the same duration
    pub is_constant_duration: bool,
}

impl TryFrom<&[u8]> for TimeToSampleAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_time_to_sample_atom(reader)
    }
}

fn parse_time_to_sample_atom<R: Read>(reader: R) -> Result<TimeToSampleAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != STTS {
        return Err(anyhow!("Invalid atom type: {} (expected stts)", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_stts_data(&mut cursor)
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
            .context(format!("read sample count for entry {}", i))?;
        let sample_count = u32::from_be_bytes(sample_count_buf);

        // Read sample duration (4 bytes)
        let mut sample_duration_buf = [0u8; 4];
        reader
            .read_exact(&mut sample_duration_buf)
            .context(format!("read sample duration for entry {}", i))?;
        let sample_duration = u32::from_be_bytes(sample_duration_buf);

        // Validate entry
        if sample_count == 0 {
            return Err(anyhow!("Entry {} has zero sample count", i));
        }
        if sample_duration == 0 {
            return Err(anyhow!("Entry {} has zero sample duration", i));
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_stts() -> TimeToSampleAtom {
        TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![
                TimeToSampleEntry {
                    sample_count: 10,
                    sample_duration: 1000,
                }, // 10 samples, 1000 units each
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 2000,
                }, // 5 samples, 2000 units each
                TimeToSampleEntry {
                    sample_count: 3,
                    sample_duration: 1500,
                }, // 3 samples, 1500 units each
            ]),
        }
    }

    #[test]
    fn test_sample_to_time() {
        let stts = create_test_stts();

        // First entry
        assert_eq!(stts.sample_to_time(0), Some(0));
        assert_eq!(stts.sample_to_time(1), Some(1000));
        assert_eq!(stts.sample_to_time(9), Some(9000));

        // Second entry
        assert_eq!(stts.sample_to_time(10), Some(10000));
        assert_eq!(stts.sample_to_time(11), Some(12000));
        assert_eq!(stts.sample_to_time(14), Some(18000));

        // Third entry
        assert_eq!(stts.sample_to_time(15), Some(20000));
        assert_eq!(stts.sample_to_time(16), Some(21500));
        assert_eq!(stts.sample_to_time(17), Some(23000));

        // Beyond range
        assert_eq!(stts.sample_to_time(18), None);
    }

    #[test]
    fn test_sample_duration() {
        let stts = create_test_stts();

        assert_eq!(stts.get_sample_duration(0), Some(1000));
        assert_eq!(stts.get_sample_duration(9), Some(1000));
        assert_eq!(stts.get_sample_duration(10), Some(2000));
        assert_eq!(stts.get_sample_duration(14), Some(2000));
        assert_eq!(stts.get_sample_duration(15), Some(1500));
        assert_eq!(stts.get_sample_duration(17), Some(1500));
        assert_eq!(stts.get_sample_duration(18), None);
    }

    #[test]
    fn test_sample_time_range() {
        let stts = create_test_stts();

        assert_eq!(stts.get_sample_time_range(0), Some((0, 1000)));
        assert_eq!(stts.get_sample_time_range(10), Some((10000, 12000)));
        assert_eq!(stts.get_sample_time_range(15), Some((20000, 21500)));
        assert_eq!(stts.get_sample_time_range(18), None);
    }

    #[test]
    fn test_constant_duration_detection() {
        let constant_stts = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![
                TimeToSampleEntry {
                    sample_count: 10,
                    sample_duration: 1000,
                },
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 1000,
                },
            ]),
        };
        assert!(constant_stts.is_constant_duration());

        let variable_stts = create_test_stts();
        assert!(!variable_stts.is_constant_duration());

        let empty_stts = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![]),
        };
        assert!(empty_stts.is_constant_duration());
    }

    #[test]
    fn test_most_common_duration() {
        let stts = create_test_stts();
        // 10 samples with 1000 duration, 5 with 2000, 3 with 1500
        // 1000 is most common
        assert_eq!(stts.most_common_duration(), Some(1000));

        let equal_stts = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 1000,
                },
                TimeToSampleEntry {
                    sample_count: 5,
                    sample_duration: 2000,
                },
            ]),
        };
        // Should return one of them (behavior may vary based on HashMap iteration)
        let result = equal_stts.most_common_duration();
        assert!(result == Some(1000) || result == Some(2000));
    }

    #[test]
    fn test_frame_rate_calculation() {
        let stts = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![TimeToSampleEntry {
                sample_count: 100,
                sample_duration: 1000,
            }]),
        };

        // With timescale 30000, duration 1000 -> 30 fps
        assert_eq!(stts.calculate_frame_rate(30000), Some(30.0));

        // With timescale 24000, duration 1000 -> 24 fps
        assert_eq!(stts.calculate_frame_rate(24000), Some(24.0));

        // Zero timescale
        assert_eq!(stts.calculate_frame_rate(0), None);

        // Empty table
        let empty_stts = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![]),
        };
        assert_eq!(empty_stts.calculate_frame_rate(30000), None);
    }

    #[test]
    fn test_validation() {
        let valid_stts = create_test_stts();
        assert!(valid_stts.validate().is_ok());

        let invalid_stts = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![
                TimeToSampleEntry {
                    sample_count: 0,
                    sample_duration: 1000,
                }, // Invalid: zero count
            ]),
        };
        assert!(invalid_stts.validate().is_err());

        let invalid_stts2 = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![
                TimeToSampleEntry {
                    sample_count: 10,
                    sample_duration: 0,
                }, // Invalid: zero duration
            ]),
        };
        assert!(invalid_stts2.validate().is_err());
    }

    #[test]
    fn test_empty_table() {
        let empty_stts = TimeToSampleAtom {
            version: 0,
            flags: [0; 3],
            entries: TimeToSampleEntries(vec![]),
        };

        assert_eq!(empty_stts.entry_count(), 0);
        assert!(empty_stts.is_empty());
        assert_eq!(empty_stts.total_sample_count(), 0);
        assert_eq!(empty_stts.total_duration(), 0);
        assert_eq!(empty_stts.time_to_sample(0), None);
        assert_eq!(empty_stts.sample_to_time(0), None);
        assert_eq!(empty_stts.get_sample_duration(0), None);
        assert_eq!(empty_stts.most_common_duration(), None);
        assert!(empty_stts.validate().is_ok());
    }
}
