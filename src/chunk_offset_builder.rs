use std::collections::VecDeque;

use crate::atom::{SampleSizeAtom, SampleToChunkAtom};

#[derive(Debug)]
pub struct ChunkInfo {
    pub track_index: usize,
    pub chunk_number: u32,
    pub samples_per_chunk: u32,
    pub chunk_size: u64,
    pub sample_indices: Vec<usize>,
    pub sample_sizes: Vec<u32>,
}

#[derive(Debug)]
pub struct ChunkOffset {
    pub track_index: usize,
    pub offset: u64,
}

#[derive(Clone)]
pub struct ChunkOffsetBuilderTrack<'a> {
    stsc: &'a SampleToChunkAtom,
    stsz: &'a SampleSizeAtom,
}

impl<'a> ChunkOffsetBuilderTrack<'a> {
    /// Build chunk information including sizes and sample mappings
    pub fn build_chunk_info(&self, track_index: usize) -> impl Iterator<Item = ChunkInfo> + 'a {
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
            .scan(
                (track_index, 0u32),
                |(track_index, sample_index), (entry, next_entry)| {
                    let next_first_chunk = if let Some(next_entry) = next_entry {
                        next_entry.first_chunk
                    } else {
                        let remaining_samples =
                            self.stsz.sample_count.saturating_sub(*sample_index);
                        entry.first_chunk + remaining_samples.div_ceil(entry.samples_per_chunk)
                    };

                    let start_sample_index = *sample_index;
                    *sample_index +=
                        (next_first_chunk - entry.first_chunk) * entry.samples_per_chunk;

                    // Process all chunks for this entry
                    let track_index = *track_index;
                    Some((entry.first_chunk..next_first_chunk).scan(
                        (track_index, start_sample_index),
                        |(track_index, sample_index), chunk_num| {
                            let (sample_indices, sample_sizes, chunk_size) = self
                                .stsz
                                .sample_sizes()
                                .enumerate()
                                .skip(*sample_index as usize)
                                .take(entry.samples_per_chunk as usize)
                                .fold(
                                    (
                                        Vec::with_capacity(entry.samples_per_chunk as usize),
                                        Vec::with_capacity(entry.samples_per_chunk as usize),
                                        0u64,
                                    ),
                                    |(mut sample_indices, mut sample_sizes, mut chunk_size),
                                     (i, size)| {
                                        sample_indices.push(i);
                                        sample_sizes.push(*size);
                                        chunk_size += *size as u64;
                                        (sample_indices, sample_sizes, chunk_size)
                                    },
                                );
                            *sample_index += entry.samples_per_chunk;

                            let track_index = *track_index;
                            Some(ChunkInfo {
                                track_index,
                                chunk_number: chunk_num,
                                samples_per_chunk: entry.samples_per_chunk,
                                chunk_size,
                                sample_indices,
                                sample_sizes,
                            })
                        },
                    ))
                },
            )
            .flatten()
    }
}

pub struct ChunkOffsetBuilder<'a> {
    tracks: Vec<ChunkOffsetBuilderTrack<'a>>,
}

impl<'a> Default for ChunkOffsetBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ChunkOffsetBuilder<'a> {
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tracks: Vec::with_capacity(capacity),
        }
    }

    pub fn add_track(&mut self, stsc: &'a SampleToChunkAtom, stsz: &'a SampleSizeAtom) {
        self.tracks.push(ChunkOffsetBuilderTrack { stsc, stsz })
    }

    /// Build interleaved chunk information including sizes and sample mappings
    pub fn build_chunk_info(&self) -> impl Iterator<Item = ChunkInfo> + 'a {
        let mut iters = VecDeque::from_iter(
            self.tracks
                .clone()
                .into_iter()
                .enumerate()
                .map(|(track_index, track)| track.build_chunk_info(track_index)),
        );

        // round-robin chunks from each track
        std::iter::from_fn(move || {
            while let Some(mut it) = iters.pop_front() {
                if let Some(item) = it.next() {
                    iters.push_back(it);
                    return Some(item);
                }
            }
            None
        })
    }

    /// Build interleaved chunk offsets for each track given a starting offset
    pub fn build_chunk_offsets(&self, start_offset: u64) -> Vec<Vec<u64>> {
        let tracks: Vec<Vec<u64>> = (0..self.tracks.len()).map(|_| Vec::new()).collect();

        let (_, chunk_offsets) = self.build_chunk_info().fold(
            (start_offset, tracks),
            |(mut current_offset, mut tracks), chunk| {
                let chunk_offset = current_offset;
                current_offset += chunk.chunk_size;
                tracks[chunk.track_index].push(chunk_offset);
                (current_offset, tracks)
            },
        );

        chunk_offsets
    }

    /// Build interleaved chunk offsets preserving the original order based on input chunk offsets
    pub fn build_chunk_offsets_ordered(
        &self,
        original_chunk_offsets: Vec<&[u64]>,
        start_offset: u64,
    ) -> Vec<Vec<u64>> {
        let mut all_chunks: Vec<(u64, ChunkInfo)> = Vec::new();

        for (track_index, track) in self.tracks.iter().enumerate() {
            let chunks: Vec<_> = track.build_chunk_info(track_index).collect();
            for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                let original_offset = original_chunk_offsets[track_index][chunk_idx];
                all_chunks.push((original_offset, chunk));
            }
        }

        all_chunks.sort_by_key(|(original_offset, _)| *original_offset);

        let mut tracks: Vec<Vec<u64>> = (0..self.tracks.len()).map(|_| Vec::new()).collect();
        let mut current_offset = start_offset;

        for (_, chunk) in all_chunks {
            tracks[chunk.track_index].push(current_offset);
            current_offset += chunk.chunk_size;
        }

        tracks
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

        let mut builder = ChunkOffsetBuilder::with_capacity(1);
        builder.add_track(&stsc, &stsz);
        let offsets = builder.build_chunk_offsets(0);
        let offsets = &offsets[0];

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

        let mut builder = ChunkOffsetBuilder::with_capacity(1);
        builder.add_track(&stsc, &stsz);
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

        let mut builder = ChunkOffsetBuilder::with_capacity(1);
        builder.add_track(&stsc, &stsz);
        let chunk_info = builder.build_chunk_info().collect::<Vec<_>>();

        assert_eq!(chunk_info.len(), 0);
    }

    #[test]
    fn test_track_interleaving() {
        // Track 1: 2 chunks with 2 samples each
        let stsc_entries_1 = vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: 2,
            sample_description_index: 1,
        }];

        let stsc_1 = SampleToChunkAtom {
            version: 0,
            flags: [0, 0, 0],
            entries: stsc_entries_1.into(),
        };

        let stsz_1 = SampleSizeAtom {
            version: 0,
            flags: 0,
            sample_size: 0,
            sample_count: 4,
            entry_sizes: vec![100, 200, 150, 250].into(),
        };

        // Track 2: 3 chunks with 1 sample each
        let stsc_entries_2 = vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: 1,
            sample_description_index: 1,
        }];

        let stsc_2 = SampleToChunkAtom {
            version: 0,
            flags: [0, 0, 0],
            entries: stsc_entries_2.into(),
        };

        let stsz_2 = SampleSizeAtom {
            version: 0,
            flags: 0,
            sample_size: 0,
            sample_count: 3,
            entry_sizes: vec![300, 400, 500].into(),
        };

        let mut builder = ChunkOffsetBuilder::with_capacity(2);
        builder.add_track(&stsc_1, &stsz_1);
        builder.add_track(&stsc_2, &stsz_2);

        let chunk_info = builder.build_chunk_info().collect::<Vec<_>>();

        // Expected interleaving: T1C1, T2C1, T1C2, T2C2, T2C3
        assert_eq!(chunk_info.len(), 5);

        // Track 1, Chunk 1: samples 0,1 (track_index=0)
        assert_eq!(chunk_info[0].track_index, 0);
        assert_eq!(chunk_info[0].chunk_number, 1);
        assert_eq!(chunk_info[0].samples_per_chunk, 2);
        assert_eq!(chunk_info[0].chunk_size, 300); // 100 + 200
        assert_eq!(chunk_info[0].sample_indices, vec![0, 1]);

        // Track 2, Chunk 1: sample 0 (track_index=1)
        assert_eq!(chunk_info[1].track_index, 1);
        assert_eq!(chunk_info[1].chunk_number, 1);
        assert_eq!(chunk_info[1].samples_per_chunk, 1);
        assert_eq!(chunk_info[1].chunk_size, 300); // 300
        assert_eq!(chunk_info[1].sample_indices, vec![0]);

        // Track 1, Chunk 2: samples 2,3 (track_index=0)
        assert_eq!(chunk_info[2].track_index, 0);
        assert_eq!(chunk_info[2].chunk_number, 2);
        assert_eq!(chunk_info[2].samples_per_chunk, 2);
        assert_eq!(chunk_info[2].chunk_size, 400); // 150 + 250
        assert_eq!(chunk_info[2].sample_indices, vec![2, 3]);

        // Track 2, Chunk 2: sample 1 (track_index=1)
        assert_eq!(chunk_info[3].track_index, 1);
        assert_eq!(chunk_info[3].chunk_number, 2);
        assert_eq!(chunk_info[3].samples_per_chunk, 1);
        assert_eq!(chunk_info[3].chunk_size, 400); // 400
        assert_eq!(chunk_info[3].sample_indices, vec![1]);

        // Track 2, Chunk 3: sample 2 (track_index=1)
        assert_eq!(chunk_info[4].track_index, 1);
        assert_eq!(chunk_info[4].chunk_number, 3);
        assert_eq!(chunk_info[4].samples_per_chunk, 1);
        assert_eq!(chunk_info[4].chunk_size, 500); // 500
        assert_eq!(chunk_info[4].sample_indices, vec![2]);

        // Test chunk offsets are calculated correctly with interleaving
        let offsets = builder.build_chunk_offsets(0);

        // Track 1 offsets: [0, 600] (T1C1 at 0, T1C2 at 0+300+300=600)
        assert_eq!(offsets[0], vec![0, 600]);

        // Track 2 offsets: [300, 1000, 1400] (T2C1 at 300, T2C2 at 300+300+400=1000, T2C3 at 1000+400=1400)
        assert_eq!(offsets[1], vec![300, 1000, 1400]);
    }

    #[test]
    fn test_build_chunk_offsets_ordered() {
        // Track 1: 2 chunks
        let stsc_entries_1 = vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: 2,
            sample_description_index: 1,
        }];

        let stsc_1 = SampleToChunkAtom {
            version: 0,
            flags: [0, 0, 0],
            entries: stsc_entries_1.into(),
        };

        let stsz_1 = SampleSizeAtom {
            version: 0,
            flags: 0,
            sample_size: 0,
            sample_count: 4,
            entry_sizes: vec![100, 200, 150, 250].into(),
        };

        // Track 2: 2 chunks
        let stsc_entries_2 = vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: 1,
            sample_description_index: 1,
        }];

        let stsc_2 = SampleToChunkAtom {
            version: 0,
            flags: [0, 0, 0],
            entries: stsc_entries_2.into(),
        };

        let stsz_2 = SampleSizeAtom {
            version: 0,
            flags: 0,
            sample_size: 0,
            sample_count: 2,
            entry_sizes: vec![300, 400].into(),
        };

        let mut builder = ChunkOffsetBuilder::with_capacity(2);
        builder.add_track(&stsc_1, &stsz_1);
        builder.add_track(&stsc_2, &stsz_2);

        // Original chunk offsets in a specific order:
        // Track 1: chunks at offsets [1000, 2000]
        // Track 2: chunks at offsets [500, 1500]
        // This means the order should be: T2C1 (500), T1C1 (1000), T2C2 (1500), T1C2 (2000)
        let original_offsets_track_1 = vec![1000u64, 2000u64];
        let original_offsets_track_2 = vec![500u64, 1500u64];
        let original_offsets = vec![
            original_offsets_track_1.as_slice(),
            original_offsets_track_2.as_slice(),
        ];

        let new_offsets = builder.build_chunk_offsets_ordered(original_offsets, 0);

        // With the original ordering (T2C1, T1C1, T2C2, T1C2) and chunk sizes:
        // T2C1: 300 bytes -> offset 0
        // T1C1: 300 bytes -> offset 300
        // T2C2: 400 bytes -> offset 600
        // T1C2: 400 bytes -> offset 1000

        // Track 1 chunks should be at offsets [300, 1000]
        assert_eq!(new_offsets[0], vec![300, 1000]);

        // Track 2 chunks should be at offsets [0, 600]
        assert_eq!(new_offsets[1], vec![0, 600]);
    }
}
