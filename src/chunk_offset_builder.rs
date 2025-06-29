use crate::atom::{SampleSizeAtom, SampleToChunkAtom};

#[derive(Debug)]
pub struct ChunkInfo {
    pub chunk_number: u32,
    pub samples_per_chunk: u32,
    pub chunk_size: u64,
    pub sample_indices: Vec<u32>,
}

pub struct ChunkOffsetBuilder {
    stsc: SampleToChunkAtom,
    stsz: SampleSizeAtom,
}

impl ChunkOffsetBuilder {
    pub fn new(stsc: SampleToChunkAtom, stsz: SampleSizeAtom) -> Self {
        Self { stsc, stsz }
    }

    /// Build chunk information including sizes and sample mappings
    pub fn build_chunk_info(&self) -> Result<Vec<ChunkInfo>, String> {
        let mut chunks = Vec::new();
        let mut sample_index = 0u32;

        // Sort entries by first_chunk to ensure proper order
        let mut entries = self.stsc.entries.to_vec();
        entries.sort_by_key(|e| e.first_chunk);

        for (i, entry) in entries.iter().enumerate() {
            let next_first_chunk = if i + 1 < entries.len() {
                entries[i + 1].first_chunk
            } else {
                // For the last entry, we need to calculate how many chunks remain
                let remaining_samples = self.stsz.sample_count - sample_index;
                entry.first_chunk + remaining_samples.div_ceil(entry.samples_per_chunk)
            };

            // Process all chunks for this entry
            for chunk_num in entry.first_chunk..next_first_chunk {
                if sample_index >= self.stsz.sample_count {
                    break;
                }

                let samples_in_this_chunk = std::cmp::min(
                    entry.samples_per_chunk,
                    self.stsz.sample_count - sample_index,
                );

                let mut chunk_size = 0u64;
                let mut sample_indices = Vec::new();

                // Calculate chunk size by summing sample sizes
                for _ in 0..samples_in_this_chunk {
                    if sample_index as usize >= self.stsz.entry_sizes.len() {
                        return Err(format!(
                            "Sample index {} exceeds available sample sizes",
                            sample_index
                        ));
                    }

                    chunk_size += self.stsz.entry_sizes[sample_index as usize] as u64;
                    sample_indices.push(sample_index);
                    sample_index += 1;
                }

                chunks.push(ChunkInfo {
                    chunk_number: chunk_num,
                    samples_per_chunk: samples_in_this_chunk,
                    chunk_size,
                    sample_indices,
                });
            }
        }

        Ok(chunks)
    }

    /// Build actual chunk offsets given a starting offset
    pub fn build_chunk_offsets(&self, start_offset: u64) -> Result<Vec<u64>, String> {
        let chunk_info = self.build_chunk_info()?;
        let mut offsets = Vec::with_capacity(chunk_info.len());
        let mut current_offset = start_offset;

        for chunk in chunk_info {
            offsets.push(current_offset);
            current_offset += chunk.chunk_size;
        }

        Ok(offsets)
    }
}
