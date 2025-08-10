use anyhow::{anyhow, Context};
use derive_more::{Deref, DerefMut};
use futures_io::AsyncRead;
use std::{fmt, io::Read};

use crate::{
    atom::{
        util::{async_to_sync_read, DebugEllipsis},
        FourCC,
    },
    parser::Parse,
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
#[derive(Default, Debug, Clone)]
pub struct SampleToChunkAtom {
    /// Version of the stsc atom format (0)
    pub version: u8,
    /// Flags for the stsc atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample-to-chunk entries
    pub entries: SampleToChunkEntries,
}

impl SampleToChunkAtom {
    /// Calculates how many chunks need to be removed for the given number of samples
    /// and updates the chunk mapping accordingly
    pub fn trim_chunks_for_samples(&mut self, samples_to_remove: u32, total_chunks: u32) -> u32 {
        let mut chunks_to_remove = 0u32;
        let mut samples_accounted_for = 0u32;
        let mut entries_to_remove = 0usize;

        for (entry_idx, entry) in self.entries.iter().enumerate() {
            let next_first_chunk = if entry_idx.saturating_add(1) < self.entries.len() {
                self.entries[entry_idx.saturating_add(1)].first_chunk
            } else {
                total_chunks.saturating_add(1) // Beyond the last chunk
            };

            let chunks_in_this_entry = next_first_chunk.saturating_sub(entry.first_chunk);
            let samples_in_this_entry =
                chunks_in_this_entry.saturating_mul(entry.samples_per_chunk);

            if samples_accounted_for.saturating_add(samples_in_this_entry) <= samples_to_remove {
                // Remove all chunks from this entry
                chunks_to_remove = chunks_to_remove.saturating_add(chunks_in_this_entry);
                samples_accounted_for = samples_accounted_for.saturating_add(samples_in_this_entry);
                entries_to_remove = entry_idx.saturating_add(1);
            } else {
                // Partial removal from this entry
                let remaining_samples = samples_to_remove.saturating_sub(samples_accounted_for);
                let chunks_to_remove_from_entry = if entry.samples_per_chunk == 0 {
                    0u32
                } else {
                    // Calculate ceiling division using saturating arithmetic
                    let numerator =
                        remaining_samples.saturating_add(entry.samples_per_chunk.saturating_sub(1));
                    numerator / entry.samples_per_chunk
                };

                chunks_to_remove = chunks_to_remove.saturating_add(chunks_to_remove_from_entry);
                break;
            }
        }

        // Remove completely consumed entries
        self.entries.drain(0..entries_to_remove);

        // Update first_chunk indices for remaining entries
        self.update_first_chunk_indices_after_removal(chunks_to_remove);

        chunks_to_remove
    }

    /// Updates all first_chunk indices after chunks have been removed
    fn update_first_chunk_indices_after_removal(&mut self, chunks_removed: u32) {
        for entry in self.entries.iter_mut() {
            entry.first_chunk = entry.first_chunk.saturating_sub(chunks_removed);
            // Ensure first_chunk is at least 1 (MP4 uses 1-based indexing)
            if entry.first_chunk == 0 {
                entry.first_chunk = 1;
            }
        }
    }

    // Calculates how many chunks need to be removed from the end for the given number of samples
    /// and updates the chunk mapping accordingly
    pub fn trim_chunks_from_end_for_samples(
        &mut self,
        samples_to_remove: u32,
        total_chunks: u32,
    ) -> u32 {
        let mut chunks_to_remove = 0u32;
        let mut samples_accounted_for = 0u32;
        let mut entries_to_remove = 0usize;

        // Work backwards through entries
        for (entry_idx, entry) in self.entries.iter().enumerate().rev() {
            let next_first_chunk = if entry_idx.saturating_add(1) < self.entries.len() {
                self.entries[entry_idx.saturating_add(1)].first_chunk
            } else {
                total_chunks.saturating_add(1) // Beyond the last chunk
            };

            let chunks_in_this_entry = next_first_chunk.saturating_sub(entry.first_chunk);
            let samples_in_this_entry =
                chunks_in_this_entry.saturating_mul(entry.samples_per_chunk);

            if samples_accounted_for.saturating_add(samples_in_this_entry) <= samples_to_remove {
                // Remove all chunks from this entry
                chunks_to_remove = chunks_to_remove.saturating_add(chunks_in_this_entry);
                samples_accounted_for = samples_accounted_for.saturating_add(samples_in_this_entry);
                entries_to_remove = entries_to_remove.saturating_add(1);
            } else {
                // Partial removal from this entry
                let remaining_samples = samples_to_remove.saturating_sub(samples_accounted_for);
                let chunks_to_remove_from_entry = if entry.samples_per_chunk == 0 {
                    0u32
                } else {
                    // Calculate ceiling division using saturating arithmetic
                    let numerator =
                        remaining_samples.saturating_add(entry.samples_per_chunk.saturating_sub(1));
                    numerator / entry.samples_per_chunk
                };

                chunks_to_remove = chunks_to_remove.saturating_add(chunks_to_remove_from_entry);
                break;
            }
        }

        // Remove completely consumed entries from the end
        let new_len = self.entries.len().saturating_sub(entries_to_remove);
        self.entries.truncate(new_len);

        chunks_to_remove
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

impl Parse for SampleToChunkAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STSC {
            return Err(ParseError::new_unexpected_atom(atom_type, STSC));
        }
        let mut cursor = async_to_sync_read(reader).await?;
        parse_stsc_data(&mut cursor).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for SampleToChunkAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STSC)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(&(self.entries.len() as u32).to_be_bytes());

        // Entries
        for entry in self.entries.iter() {
            // First chunk (4 bytes, big-endian)
            data.extend_from_slice(&entry.first_chunk.to_be_bytes());
            // Samples per chunk (4 bytes, big-endian)
            data.extend_from_slice(&entry.samples_per_chunk.to_be_bytes());
            // Sample description index (4 bytes, big-endian)
            data.extend_from_slice(&entry.sample_description_index.to_be_bytes());
        }

        data
    }
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
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available stco/co64 test data files
    #[test]
    fn test_stsc_roundtrip() {
        test_atom_roundtrip_sync::<SampleToChunkAtom>(STSC);
    }
}
