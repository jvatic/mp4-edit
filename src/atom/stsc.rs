use anyhow::{anyhow, Context};
use derive_more::Deref;
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis};

pub const STSC: &[u8; 4] = b"stsc";

#[derive(Clone, Deref)]
pub struct SampleToChunkEntries(Vec<SampleToChunkEntry>);

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleToChunkEntry {
    /// First chunk number (1-based) that uses this entry
    pub first_chunk: u32,
    /// Number of samples in each chunk
    pub samples_per_chunk: u32,
    /// Sample description index (1-based, references stsd atom)
    pub sample_description_index: u32,
}

/// Sample-to-Chunk Atom - contains sample-to-chunk mapping table
#[derive(Debug, Clone)]
pub struct SampleToChunkAtom {
    /// Version of the stsc atom format (0)
    pub version: u8,
    /// Flags for the stsc atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample-to-chunk entries
    pub entries: SampleToChunkEntries,
}

impl SampleToChunkAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_sample_to_chunk_atom(reader)
    }

    /// Get the number of sample-to-chunk entries
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Calculate the total number of samples given the total chunk count
    pub fn calculate_total_samples(&self, total_chunks: u32) -> u64 {
        if self.entries.is_empty() || total_chunks == 0 {
            return 0;
        }

        let mut total_samples = 0u64;

        for (i, entry) in self.entries.iter().enumerate() {
            let chunk_start = entry.first_chunk;
            let chunk_end = if i + 1 < self.entries.len() {
                self.entries[i + 1].first_chunk
            } else {
                total_chunks + 1
            };

            if chunk_start <= total_chunks {
                let actual_chunk_end = chunk_end.min(total_chunks + 1);
                let chunk_count = actual_chunk_end - chunk_start;
                total_samples += chunk_count as u64 * entry.samples_per_chunk as u64;
            }
        }

        total_samples
    }

    /// Get the chunk number (1-based) that contains the given sample (0-based)
    pub fn sample_to_chunk(&self, sample: u64) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }

        let mut current_sample = 0u64;
        let mut current_chunk = 1u32;

        for (i, entry) in self.entries.iter().enumerate() {
            let next_chunk = if i + 1 < self.entries.len() {
                self.entries[i + 1].first_chunk
            } else {
                u32::MAX // Last entry extends indefinitely
            };

            // Skip to the first chunk of this entry if we're not there yet
            if current_chunk < entry.first_chunk {
                current_chunk = entry.first_chunk;
            }

            // Process chunks in this entry
            while current_chunk < next_chunk {
                let chunk_sample_count = entry.samples_per_chunk as u64;

                if sample < current_sample + chunk_sample_count {
                    return Some(current_chunk);
                }

                current_sample += chunk_sample_count;
                current_chunk += 1;
            }
        }

        None
    }

    /// Get the first sample number (0-based) in the given chunk (1-based)
    pub fn chunk_to_first_sample(&self, chunk: u32) -> Option<u64> {
        if self.entries.is_empty() || chunk == 0 {
            return None;
        }

        let mut current_sample = 0u64;
        let mut current_chunk = 1u32;

        for (i, entry) in self.entries.iter().enumerate() {
            let next_chunk = if i + 1 < self.entries.len() {
                self.entries[i + 1].first_chunk
            } else {
                u32::MAX
            };

            // Skip to the first chunk of this entry if we're not there yet
            if current_chunk < entry.first_chunk {
                current_chunk = entry.first_chunk;
            }

            // Check if our target chunk is in this entry
            if chunk >= entry.first_chunk && chunk < next_chunk {
                let chunks_before = chunk - current_chunk;
                return Some(
                    current_sample + chunks_before as u64 * entry.samples_per_chunk as u64,
                );
            }

            // Process all chunks in this entry
            let chunks_in_entry = next_chunk.saturating_sub(entry.first_chunk);
            current_sample += chunks_in_entry as u64 * entry.samples_per_chunk as u64;
            current_chunk = next_chunk;
        }

        None
    }

    /// Get the number of samples in the given chunk (1-based)
    pub fn get_chunk_sample_count(&self, chunk: u32) -> Option<u32> {
        if self.entries.is_empty() || chunk == 0 {
            return None;
        }

        for (i, entry) in self.entries.iter().enumerate() {
            let next_chunk = if i + 1 < self.entries.len() {
                self.entries[i + 1].first_chunk
            } else {
                u32::MAX
            };

            if chunk >= entry.first_chunk && chunk < next_chunk {
                return Some(entry.samples_per_chunk);
            }
        }

        None
    }

    /// Get the sample description index for the given chunk (1-based)
    pub fn get_chunk_sample_description_index(&self, chunk: u32) -> Option<u32> {
        if self.entries.is_empty() || chunk == 0 {
            return None;
        }

        for (i, entry) in self.entries.iter().enumerate() {
            let next_chunk = if i + 1 < self.entries.len() {
                self.entries[i + 1].first_chunk
            } else {
                u32::MAX
            };

            if chunk >= entry.first_chunk && chunk < next_chunk {
                return Some(entry.sample_description_index);
            }
        }

        None
    }

    /// Get all samples in the given chunk (1-based) as a range (start, count)
    pub fn get_chunk_sample_range(&self, chunk: u32) -> Option<(u64, u32)> {
        let first_sample = self.chunk_to_first_sample(chunk)?;
        let sample_count = self.get_chunk_sample_count(chunk)?;
        Some((first_sample, sample_count))
    }

    /// Get chunks within a sample range (0-based samples)
    pub fn get_chunks_for_sample_range(&self, start_sample: u64, end_sample: u64) -> Vec<u32> {
        if start_sample >= end_sample {
            return Vec::new();
        }

        let start_chunk = self.sample_to_chunk(start_sample).unwrap_or(1);
        let end_chunk = self
            .sample_to_chunk(end_sample.saturating_sub(1))
            .unwrap_or(start_chunk);

        (start_chunk..=end_chunk).collect()
    }

    /// Check if all chunks have the same number of samples
    pub fn is_constant_samples_per_chunk(&self) -> bool {
        if self.entries.is_empty() {
            return true;
        }

        let first_samples_per_chunk = self.entries[0].samples_per_chunk;
        self.entries
            .iter()
            .all(|entry| entry.samples_per_chunk == first_samples_per_chunk)
    }

    /// Get the most common samples-per-chunk value
    pub fn most_common_samples_per_chunk(&self, total_chunks: u32) -> Option<u32> {
        if self.entries.is_empty() {
            return None;
        }

        let mut samples_per_chunk_counts: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();

        for (i, entry) in self.entries.iter().enumerate() {
            let chunk_end = if i + 1 < self.entries.len() {
                self.entries[i + 1].first_chunk
            } else {
                total_chunks + 1
            };

            let chunk_count = chunk_end.saturating_sub(entry.first_chunk);
            *samples_per_chunk_counts
                .entry(entry.samples_per_chunk)
                .or_insert(0) += chunk_count;
        }

        samples_per_chunk_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(samples_per_chunk, _)| samples_per_chunk)
    }

    /// Get statistics about the sample-to-chunk table
    pub fn get_statistics(&self, total_chunks: u32) -> SampleToChunkStatistics {
        if self.entries.is_empty() {
            return SampleToChunkStatistics::default();
        }

        let entry_count = self.entries.len();
        let total_samples = self.calculate_total_samples(total_chunks);

        let min_samples_per_chunk = self
            .entries
            .iter()
            .map(|e| e.samples_per_chunk)
            .min()
            .unwrap_or(0);
        let max_samples_per_chunk = self
            .entries
            .iter()
            .map(|e| e.samples_per_chunk)
            .max()
            .unwrap_or(0);

        let is_constant = self.is_constant_samples_per_chunk();
        let most_common = self
            .most_common_samples_per_chunk(total_chunks)
            .unwrap_or(0);

        // Count unique sample description indices
        let unique_sample_descriptions: std::collections::HashSet<u32> = self
            .entries
            .iter()
            .map(|e| e.sample_description_index)
            .collect();

        SampleToChunkStatistics {
            entry_count,
            total_chunks,
            total_samples,
            min_samples_per_chunk,
            max_samples_per_chunk,
            most_common_samples_per_chunk: most_common,
            is_constant_samples_per_chunk: is_constant,
            unique_sample_description_count: unique_sample_descriptions.len(),
        }
    }

    /// Validate the sample-to-chunk table for consistency
    pub fn validate(&self) -> Result<(), String> {
        if self.entries.is_empty() {
            return Ok(());
        }

        // Check that entries are sorted by first_chunk
        for i in 1..self.entries.len() {
            if self.entries[i].first_chunk <= self.entries[i - 1].first_chunk {
                return Err(format!(
                    "Entry {} has first_chunk {} <= previous entry's first_chunk {}",
                    i,
                    self.entries[i].first_chunk,
                    self.entries[i - 1].first_chunk
                ));
            }
        }

        // Validate individual entries
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.first_chunk == 0 {
                return Err(format!(
                    "Entry {} has zero first_chunk (should be 1-based)",
                    i
                ));
            }
            if entry.samples_per_chunk == 0 {
                return Err(format!("Entry {} has zero samples_per_chunk", i));
            }
            if entry.sample_description_index == 0 {
                return Err(format!(
                    "Entry {} has zero sample_description_index (should be 1-based)",
                    i
                ));
            }
        }

        Ok(())
    }

    /// Find the entry that applies to the given chunk
    pub fn find_entry_for_chunk(&self, chunk: u32) -> Option<&SampleToChunkEntry> {
        if self.entries.is_empty() || chunk == 0 {
            return None;
        }

        for (i, entry) in self.entries.iter().enumerate() {
            let next_chunk = if i + 1 < self.entries.len() {
                self.entries[i + 1].first_chunk
            } else {
                u32::MAX
            };

            if chunk >= entry.first_chunk && chunk < next_chunk {
                return Some(entry);
            }
        }

        None
    }
}

/// Statistics about the sample-to-chunk table
#[derive(Debug, Clone, Default)]
pub struct SampleToChunkStatistics {
    /// Number of entries in the table
    pub entry_count: usize,
    /// Total number of chunks
    pub total_chunks: u32,
    /// Total number of samples
    pub total_samples: u64,
    /// Minimum samples per chunk
    pub min_samples_per_chunk: u32,
    /// Maximum samples per chunk
    pub max_samples_per_chunk: u32,
    /// Most common samples per chunk value
    pub most_common_samples_per_chunk: u32,
    /// Whether all chunks have the same number of samples
    pub is_constant_samples_per_chunk: bool,
    /// Number of unique sample description indices
    pub unique_sample_description_count: usize,
}

impl TryFrom<&[u8]> for SampleToChunkAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_sample_to_chunk_atom(reader)
    }
}

fn parse_sample_to_chunk_atom<R: Read>(reader: R) -> Result<SampleToChunkAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != STSC {
        return Err(anyhow!("Invalid atom type: {} (expected stsc)", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_stsc_data(&mut cursor)
}

fn parse_stsc_data<R: Read>(mut reader: R) -> Result<SampleToChunkAtom, anyhow::Error> {
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
        return Err(anyhow!("Too many sample-to-chunk entries: {}", entry_count));
    }

    let mut entries = Vec::with_capacity(entry_count as usize);

    for i in 0..entry_count {
        // Read first chunk (4 bytes)
        let mut first_chunk_buf = [0u8; 4];
        reader
            .read_exact(&mut first_chunk_buf)
            .context(format!("read first_chunk for entry {}", i))?;
        let first_chunk = u32::from_be_bytes(first_chunk_buf);

        // Read samples per chunk (4 bytes)
        let mut samples_per_chunk_buf = [0u8; 4];
        reader
            .read_exact(&mut samples_per_chunk_buf)
            .context(format!("read samples_per_chunk for entry {}", i))?;
        let samples_per_chunk = u32::from_be_bytes(samples_per_chunk_buf);

        // Read sample description index (4 bytes)
        let mut sample_description_index_buf = [0u8; 4];
        reader
            .read_exact(&mut sample_description_index_buf)
            .context(format!("read sample_description_index for entry {}", i))?;
        let sample_description_index = u32::from_be_bytes(sample_description_index_buf);

        // Validate entry
        if first_chunk == 0 {
            return Err(anyhow!(
                "Entry {} has zero first_chunk (should be 1-based)",
                i
            ));
        }
        if samples_per_chunk == 0 {
            return Err(anyhow!("Entry {} has zero samples_per_chunk", i));
        }
        if sample_description_index == 0 {
            return Err(anyhow!(
                "Entry {} has zero sample_description_index (should be 1-based)",
                i
            ));
        }

        entries.push(SampleToChunkEntry {
            first_chunk,
            samples_per_chunk,
            sample_description_index,
        });
    }

    Ok(SampleToChunkAtom {
        version,
        flags,
        entries: SampleToChunkEntries(entries),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_stsc() -> SampleToChunkAtom {
        SampleToChunkAtom {
            version: 0,
            flags: [0; 3],
            entries: SampleToChunkEntries(vec![
                SampleToChunkEntry {
                    first_chunk: 1,
                    samples_per_chunk: 2,
                    sample_description_index: 1,
                }, // Chunks 1-4: 2 samples each
                SampleToChunkEntry {
                    first_chunk: 5,
                    samples_per_chunk: 3,
                    sample_description_index: 1,
                }, // Chunks 5-7: 3 samples each
                SampleToChunkEntry {
                    first_chunk: 8,
                    samples_per_chunk: 1,
                    sample_description_index: 2,
                }, // Chunks 8+: 1 sample each
            ]),
        }
    }

    #[test]
    fn test_basic_properties() {
        let stsc = create_test_stsc();

        assert_eq!(stsc.entry_count(), 3);
        assert!(!stsc.is_empty());

        // With 10 total chunks: 4*2 + 3*3 + 3*1 = 8 + 9 + 3 = 20 samples
        assert_eq!(stsc.calculate_total_samples(10), 20);
    }

    #[test]
    fn test_sample_to_chunk() {
        let stsc = create_test_stsc();

        // First entry: chunks 1-4, 2 samples each
        assert_eq!(stsc.sample_to_chunk(0), Some(1)); // Sample 0 -> Chunk 1
        assert_eq!(stsc.sample_to_chunk(1), Some(1)); // Sample 1 -> Chunk 1
        assert_eq!(stsc.sample_to_chunk(2), Some(2)); // Sample 2 -> Chunk 2
        assert_eq!(stsc.sample_to_chunk(7), Some(4)); // Sample 7 -> Chunk 4

        // Second entry: chunks 5-7, 3 samples each
        assert_eq!(stsc.sample_to_chunk(8), Some(5)); // Sample 8 -> Chunk 5
        assert_eq!(stsc.sample_to_chunk(10), Some(5)); // Sample 10 -> Chunk 5
        assert_eq!(stsc.sample_to_chunk(11), Some(6)); // Sample 11 -> Chunk 6
        assert_eq!(stsc.sample_to_chunk(16), Some(7)); // Sample 16 -> Chunk 7

        // Third entry: chunks 8+, 1 sample each
        assert_eq!(stsc.sample_to_chunk(17), Some(8)); // Sample 17 -> Chunk 8
        assert_eq!(stsc.sample_to_chunk(18), Some(9)); // Sample 18 -> Chunk 9
    }

    #[test]
    fn test_chunk_to_first_sample() {
        let stsc = create_test_stsc();

        // First entry: chunks 1-4, 2 samples each
        assert_eq!(stsc.chunk_to_first_sample(1), Some(0)); // Chunk 1 starts at sample 0
        assert_eq!(stsc.chunk_to_first_sample(2), Some(2)); // Chunk 2 starts at sample 2
        assert_eq!(stsc.chunk_to_first_sample(4), Some(6)); // Chunk 4 starts at sample 6

        // Second entry: chunks 5-7, 3 samples each
        assert_eq!(stsc.chunk_to_first_sample(5), Some(8)); // Chunk 5 starts at sample 8
        assert_eq!(stsc.chunk_to_first_sample(6), Some(11)); // Chunk 6 starts at sample 11
        assert_eq!(stsc.chunk_to_first_sample(7), Some(14)); // Chunk 7 starts at sample 14

        // Third entry: chunks 8+, 1 sample each
        assert_eq!(stsc.chunk_to_first_sample(8), Some(17)); // Chunk 8 starts at sample 17
        assert_eq!(stsc.chunk_to_first_sample(9), Some(18)); // Chunk 9 starts at sample 18

        // Invalid chunk
        assert_eq!(stsc.chunk_to_first_sample(0), None);
    }

    #[test]
    fn test_chunk_sample_count() {
        let stsc = create_test_stsc();

        assert_eq!(stsc.get_chunk_sample_count(1), Some(2));
        assert_eq!(stsc.get_chunk_sample_count(4), Some(2));
        assert_eq!(stsc.get_chunk_sample_count(5), Some(3));
        assert_eq!(stsc.get_chunk_sample_count(7), Some(3));
        assert_eq!(stsc.get_chunk_sample_count(8), Some(1));
        assert_eq!(stsc.get_chunk_sample_count(100), Some(1)); // Last entry extends
        assert_eq!(stsc.get_chunk_sample_count(0), None);
    }

    #[test]
    fn test_chunk_sample_description_index() {
        let stsc = create_test_stsc();

        assert_eq!(stsc.get_chunk_sample_description_index(1), Some(1));
        assert_eq!(stsc.get_chunk_sample_description_index(7), Some(1));
        assert_eq!(stsc.get_chunk_sample_description_index(8), Some(2));
        assert_eq!(stsc.get_chunk_sample_description_index(100), Some(2));
        assert_eq!(stsc.get_chunk_sample_description_index(0), None);
    }

    #[test]
    fn test_chunk_sample_range() {
        let stsc = create_test_stsc();

        assert_eq!(stsc.get_chunk_sample_range(1), Some((0, 2))); // Chunk 1: samples 0-1
        assert_eq!(stsc.get_chunk_sample_range(5), Some((8, 3))); // Chunk 5: samples 8-10
        assert_eq!(stsc.get_chunk_sample_range(8), Some((17, 1))); // Chunk 8: sample 17
        assert_eq!(stsc.get_chunk_sample_range(0), None);
    }

    #[test]
    fn test_chunks_for_sample_range() {
        let stsc = create_test_stsc();

        // Samples 0-3 span chunks 1-2
        assert_eq!(stsc.get_chunks_for_sample_range(0, 4), vec![1, 2]);

        // Samples 8-16 span chunks 5-7
        assert_eq!(stsc.get_chunks_for_sample_range(8, 17), vec![5, 6, 7]);

        // Single sample
        assert_eq!(stsc.get_chunks_for_sample_range(17, 18), vec![8]);

        // Invalid range
        assert_eq!(stsc.get_chunks_for_sample_range(10, 5), vec![]);
    }

    #[test]
    fn test_constant_samples_per_chunk() {
        let constant_stsc = SampleToChunkAtom {
            version: 0,
            flags: [0; 3],
            entries: SampleToChunkEntries(vec![
                SampleToChunkEntry {
                    first_chunk: 1,
                    samples_per_chunk: 2,
                    sample_description_index: 1,
                },
                SampleToChunkEntry {
                    first_chunk: 5,
                    samples_per_chunk: 2,
                    sample_description_index: 2,
                },
            ]),
        };
        assert!(constant_stsc.is_constant_samples_per_chunk());

        let variable_stsc = create_test_stsc();
        assert!(!variable_stsc.is_constant_samples_per_chunk());
    }

    #[test]
    fn test_most_common_samples_per_chunk() {
        let stsc = create_test_stsc();
        // With 10 chunks: 4 chunks with 2 samples, 3 chunks with 3 samples, 3 chunks with 1 sample
        // 4 > 3 > 3, so 2 should be most common
        assert_eq!(stsc.most_common_samples_per_chunk(10), Some(2));
    }

    #[test]
    fn test_statistics() {
        let stsc = create_test_stsc();
        let stats = stsc.get_statistics(10);

        assert_eq!(stats.entry_count, 3);
        assert_eq!(stats.total_chunks, 10);
        assert_eq!(stats.total_samples, 20);
        assert_eq!(stats.min_samples_per_chunk, 1);
        assert_eq!(stats.max_samples_per_chunk, 3);
        assert_eq!(stats.most_common_samples_per_chunk, 2);
        assert!(!stats.is_constant_samples_per_chunk);
        assert_eq!(stats.unique_sample_description_count, 2);
    }

    #[test]
    fn test_validation() {
        let valid_stsc = create_test_stsc();
        assert!(valid_stsc.validate().is_ok());

        // Test unsorted entries
        let invalid_stsc = SampleToChunkAtom {
            version: 0,
            flags: [0; 3],
            entries: SampleToChunkEntries(vec![
                SampleToChunkEntry {
                    first_chunk: 5,
                    samples_per_chunk: 2,
                    sample_description_index: 1,
                },
                SampleToChunkEntry {
                    first_chunk: 3, // Should be > 5
                    samples_per_chunk: 1,
                    sample_description_index: 1,
                },
            ]),
        };
        assert!(invalid_stsc.validate().is_err());

        // Test zero values
        let invalid_stsc2 = SampleToChunkAtom {
            version: 0,
            flags: [0; 3],
            entries: SampleToChunkEntries(vec![SampleToChunkEntry {
                first_chunk: 0, // Should be >= 1
                samples_per_chunk: 2,
                sample_description_index: 1,
            }]),
        };
        assert!(invalid_stsc2.validate().is_err());
    }

    #[test]
    fn test_find_entry_for_chunk() {
        let stsc = create_test_stsc();

        let entry1 = stsc.find_entry_for_chunk(3).unwrap();
        assert_eq!(entry1.first_chunk, 1);
        assert_eq!(entry1.samples_per_chunk, 2);

        let entry2 = stsc.find_entry_for_chunk(6).unwrap();
        assert_eq!(entry2.first_chunk, 5);
        assert_eq!(entry2.samples_per_chunk, 3);

        let entry3 = stsc.find_entry_for_chunk(10).unwrap();
        assert_eq!(entry3.first_chunk, 8);
        assert_eq!(entry3.samples_per_chunk, 1);

        assert!(stsc.find_entry_for_chunk(0).is_none());
    }

    #[test]
    fn test_empty_table() {
        let empty_stsc = SampleToChunkAtom {
            version: 0,
            flags: [0; 3],
            entries: SampleToChunkEntries(vec![]),
        };

        assert_eq!(empty_stsc.entry_count(), 0);
        assert!(empty_stsc.is_empty());
        assert_eq!(empty_stsc.calculate_total_samples(10), 0);
        assert_eq!(empty_stsc.sample_to_chunk(0), None);
        assert_eq!(empty_stsc.chunk_to_first_sample(1), None);
        assert_eq!(empty_stsc.get_chunk_sample_count(1), None);
        assert_eq!(empty_stsc.most_common_samples_per_chunk(10), None);
        assert!(empty_stsc.validate().is_ok());
    }
}
