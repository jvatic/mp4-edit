use crate::atom::{SampleSizeAtom, SampleToChunkAtom};

#[derive(Debug)]
pub struct ChunkInfo {
    pub chunk_number: u32,
    pub samples_per_chunk: u32,
    pub chunk_size: u64,
    pub sample_indices: Vec<usize>,
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
    pub fn build_chunk_info(&self) -> impl Iterator<Item = ChunkInfo> + '_ {
        let mut sample_index = 0u32;

        self.stsc
            .entries
            .iter()
            .zip(
                self.stsc
                    .entries
                    .iter()
                    .skip(1)
                    .map(Some)
                    .chain(std::iter::once(None)),
            )
            .flat_map(move |(entry, next_entry)| {
                let next_first_chunk = if let Some(next_entry) = next_entry {
                    next_entry.first_chunk
                } else {
                    let remaining_samples = self.stsz.sample_count - sample_index;
                    entry.first_chunk + remaining_samples.div_ceil(entry.samples_per_chunk)
                };

                let start_sample_index = sample_index;
                sample_index += (next_first_chunk - entry.first_chunk) * entry.samples_per_chunk;

                // Process all chunks for this entry
                (entry.first_chunk..next_first_chunk).scan(
                    start_sample_index,
                    |sample_index, chunk_num| {
                        let (sample_indices, chunk_size) = self
                            .stsz
                            .sample_sizes()
                            .enumerate()
                            .skip(*sample_index as usize)
                            .take(entry.samples_per_chunk as usize)
                            .fold(
                                (Vec::with_capacity(entry.samples_per_chunk as usize), 0u64),
                                |(mut sample_indices, mut chunk_size), (i, size)| {
                                    sample_indices.push(i);
                                    chunk_size += *size as u64;
                                    (sample_indices, chunk_size)
                                },
                            );
                        *sample_index += entry.samples_per_chunk;

                        Some(ChunkInfo {
                            chunk_number: chunk_num,
                            samples_per_chunk: entry.samples_per_chunk,
                            chunk_size,
                            sample_indices,
                        })
                    },
                )
            })
    }

    /// Build actual chunk offsets given a starting offset
    pub fn build_chunk_offsets(&self, start_offset: u64) -> impl Iterator<Item = u64> + '_ {
        self.build_chunk_info()
            .scan(start_offset, |current_offset, chunk| {
                let chunk_offset = *current_offset;
                *current_offset += chunk.chunk_size;
                Some(chunk_offset)
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::stsc::SampleToChunkEntry;

    use super::*;

    #[test]
    fn test_chunk_offset_calculation() {
        let stsc_entries = vec![
            SampleToChunkEntry {
                first_chunk: 1,
                samples_per_chunk: 2,
                sample_description_index: 1,
            },
            SampleToChunkEntry {
                first_chunk: 3,
                samples_per_chunk: 3,
                sample_description_index: 1,
            },
        ];

        let stsc = SampleToChunkAtom {
            version: 0,
            flags: [0, 0, 0],
            entries: stsc_entries.into(),
        };

        let stsz = SampleSizeAtom {
            version: 0,
            flags: 0,
            sample_size: 0,
            sample_count: 7,
            entry_sizes: vec![100, 200, 150, 250, 300, 400, 500].into(),
        };

        let builder = ChunkOffsetBuilder::new(stsc, stsz);
        let offsets = builder.build_chunk_offsets(0).collect::<Vec<_>>();

        // Expected:
        // Chunk 1: samples 0,1 (100+200=300 bytes) -> offset 0
        // Chunk 2: samples 2,3 (150+250=400 bytes) -> offset 300
        // Chunk 3: samples 4,5,6 (300+400+500=1200 bytes) -> offset 700

        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], 0); // Chunk 1 starts at 0
        assert_eq!(offsets[1], 300); // Chunk 2 starts at 300
        assert_eq!(offsets[2], 700); // Chunk 3 starts at 700
    }

    #[test]
    fn test_chunk_info_generation() {
        let stsc_entries = vec![
            SampleToChunkEntry {
                first_chunk: 1,
                samples_per_chunk: 2,
                sample_description_index: 1,
            },
            SampleToChunkEntry {
                first_chunk: 3,
                samples_per_chunk: 1,
                sample_description_index: 1,
            },
        ];

        let stsc = SampleToChunkAtom {
            version: 0,
            flags: [0, 0, 0],
            entries: stsc_entries.into(),
        };

        let stsz = SampleSizeAtom {
            version: 0,
            flags: 0,
            sample_size: 0,
            sample_count: 5,
            entry_sizes: vec![100, 200, 300, 400, 500].into(),
        };

        let builder = ChunkOffsetBuilder::new(stsc, stsz);
        let chunk_info = builder.build_chunk_info().collect::<Vec<_>>();

        assert_eq!(chunk_info.len(), 3);

        // Chunk 1: 2 samples (0, 1)
        assert_eq!(chunk_info[0].chunk_number, 1);
        assert_eq!(chunk_info[0].samples_per_chunk, 2);
        assert_eq!(chunk_info[0].chunk_size, 300); // 100 + 200
        assert_eq!(chunk_info[0].sample_indices, vec![0, 1]);

        // Chunk 2: 2 samples (2, 3)
        assert_eq!(chunk_info[1].chunk_number, 2);
        assert_eq!(chunk_info[1].samples_per_chunk, 2);
        assert_eq!(chunk_info[1].chunk_size, 700); // 300 + 400
        assert_eq!(chunk_info[1].sample_indices, vec![2, 3]);

        // Chunk 3: 1 sample (4)
        assert_eq!(chunk_info[2].chunk_number, 3);
        assert_eq!(chunk_info[2].samples_per_chunk, 1);
        assert_eq!(chunk_info[2].chunk_size, 500); // 500
        assert_eq!(chunk_info[2].sample_indices, vec![4]);
    }

    #[test]
    fn test_edge_case_empty_samples() {
        let stsc_entries = vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: 1,
            sample_description_index: 1,
        }];

        let stsc = SampleToChunkAtom {
            version: 0,
            flags: [0, 0, 0],
            entries: stsc_entries.into(),
        };

        let stsz = SampleSizeAtom {
            version: 0,
            flags: 0,
            sample_size: 0,
            sample_count: 0,
            entry_sizes: vec![].into(),
        };

        let builder = ChunkOffsetBuilder::new(stsc, stsz);
        let chunk_info = builder.build_chunk_info().collect::<Vec<_>>();

        assert_eq!(chunk_info.len(), 0);
    }
}
