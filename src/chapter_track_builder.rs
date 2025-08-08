use std::time::Duration;

use bon::bon;

use crate::atom::{
    chpl::ChapterEntries,
    containers::*,
    dref::{DataReferenceEntry, DataReferenceEntryInner},
    elst::{EditEntry, ELST},
    gmhd::GenericMediaHeaderAtom,
    hdlr::{HandlerReferenceAtom, HandlerType},
    mdhd::{LanguageCode, MediaHeaderAtom},
    stco_co64::ChunkOffsetAtom,
    stsc::{SampleToChunkAtom, SampleToChunkEntry},
    stsd::{
        SampleDescriptionTableAtom, SampleEntry, SampleEntryData, SampleEntryType, TextSampleEntry,
    },
    stsz::SampleSizeAtom,
    stts::{TimeToSampleAtom, TimeToSampleEntry},
    tkhd::TrackHeaderAtom,
    util::time::{mp4_timestamp_now, scaled_duration},
    Atom, AtomData, AtomHeader, DataReferenceAtom, EditListAtom, FourCC,
};
use crate::writer::SerializeAtom;

/// Represents a chapter track that can generate TRAK atoms and sample data
pub struct ChapterTrack {
    language: LanguageCode,
    timescale: u32,
    track_id: u32,
    creation_time: u64,
    modification_time: u64,
    sample_data: Vec<Vec<u8>>,
    sample_durations: Vec<u32>,
    media_duration: u64,
    movie_duration: u64,
}

#[bon]
impl ChapterTrack {
    #[builder]
    pub fn new(
        #[builder(finish_fn, into)] chapters: ChapterEntries,
        #[builder(default = LanguageCode::Undetermined)] language: LanguageCode,
        #[builder(default = 1000)] timescale: u32,
        #[builder(default = 600)] movie_timescale: u32,
        #[builder(default = 2)] track_id: u32,
        #[builder(into)] total_duration: Duration,
        #[builder(default = mp4_timestamp_now())] creation_time: u64,
        #[builder(default = mp4_timestamp_now())] modification_time: u64,
    ) -> Self {
        let (sample_data, sample_durations) =
            Self::create_samples_and_durations(&chapters, total_duration, timescale);
        Self {
            language,
            timescale,
            track_id,
            creation_time,
            modification_time,
            sample_data,
            sample_durations,
            // Calculate duration in media timescale (for MDHD and STTS)
            media_duration: scaled_duration(total_duration, timescale as u64),
            // Calculate duration in movie timescale (for TKHD)
            movie_duration: scaled_duration(total_duration, movie_timescale as u64),
        }
    }

    /// Create chapter marker samples with fixed-size data
    fn create_samples_and_durations(
        chapters: &ChapterEntries,
        total_duration: Duration,
        timescale: u32,
    ) -> (Vec<Vec<u8>>, Vec<u32>) {
        if chapters.is_empty() {
            return (vec![], vec![]);
        }

        let total_duration_100ns = total_duration.as_nanos() / 100;
        let mut samples = Vec::new();
        let mut durations = Vec::new();

        for (i, chapter) in chapters.iter().enumerate() {
            // Create minimal chapter marker data (similar to working file's 45-byte samples)
            // This creates a simple text sample with the chapter title
            let chapter_marker = Self::create_chapter_marker_data(&chapter.title);
            samples.push(chapter_marker);

            // Calculate duration for this chapter
            let end_time = if i + 1 < chapters.len() {
                chapters[i + 1].start_time
            } else {
                total_duration_100ns as u64
            };

            let chapter_duration_100ns = end_time - chapter.start_time;
            let chapter_duration_seconds = chapter_duration_100ns as f64 / 10_000_000.0;
            let chapter_duration_scaled = (chapter_duration_seconds * timescale as f64) as u32;

            durations.push(chapter_duration_scaled);
        }

        (samples, durations)
    }

    /// Create standardized chapter marker data (45 bytes like working file)
    fn create_chapter_marker_data(title: &str) -> Vec<u8> {
        // Create a simple text sample similar to QuickTime chapter markers
        // Using a fixed format that matches expected chapter track structure
        let mut data = Vec::new();

        // Add text length as 16-bit big-endian (QuickTime text sample format)
        let text_bytes = title.as_bytes();
        let text_len = text_bytes.len().min(65535) as u16;
        data.extend_from_slice(&text_len.to_be_bytes());

        // Add the actual text
        data.extend_from_slice(&text_bytes[..text_len as usize]);

        // Pad to consistent size (45 bytes like working file)
        // This ensures all samples have the same size for better compatibility
        const TARGET_SIZE: usize = 45;
        if data.len() < TARGET_SIZE {
            data.resize(TARGET_SIZE, 0);
        } else if data.len() > TARGET_SIZE {
            data.truncate(TARGET_SIZE);
        }

        data
    }

    /// Returns all sample data concatenated for writing to mdat
    pub fn sample_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();
        for sample in &self.sample_data {
            result.extend_from_slice(sample);
        }
        result
    }

    /// Returns the individual sample data
    pub fn individual_samples(&self) -> &[Vec<u8>] {
        &self.sample_data
    }

    /// Returns the total size of all samples combined
    pub fn total_sample_size(&self) -> usize {
        self.sample_data.iter().map(|s| s.len()).sum()
    }

    /// Creates a TRAK atom for the chapter track at the specified chunk offset
    pub fn create_trak_atom(&self, chunk_offset: u64) -> Atom {
        let track_header = self.create_track_header();
        let edit_list = self.create_edit_list_atom();
        let media_atom = self.create_media_atom(chunk_offset);

        Atom {
            header: AtomHeader::new(FourCC(*TRAK)),
            data: None,
            children: vec![track_header, edit_list, media_atom],
        }
    }

    fn create_track_header(&self) -> Atom {
        let tkhd = TrackHeaderAtom {
            version: 0,
            flags: [0, 0, 2], // Enabled in movie, disabled in preview
            creation_time: self.creation_time,
            modification_time: self.modification_time,
            track_id: self.track_id,
            duration: self.movie_duration, // Use movie timescale duration
            layer: 0,
            alternate_group: 0,
            volume: 0.0,  // Text track has no volume
            matrix: None, // Use default identity matrix
            width: 0.0,   // Text track dimensions
            height: 0.0,
        };

        Atom {
            header: AtomHeader::new(tkhd.atom_type()),
            data: Some(AtomData::TrackHeader(tkhd)),
            children: vec![],
        }
    }

    fn create_edit_list_atom(&self) -> Atom {
        Atom {
            header: AtomHeader::new(FourCC(*EDTS)),
            data: None,
            children: vec![Atom {
                header: AtomHeader::new(FourCC(*ELST)),
                data: Some(AtomData::EditList(EditListAtom {
                    version: 0,
                    flags: [0u8; 3],
                    entries: vec![EditEntry {
                        segment_duration: self.movie_duration,
                        media_time: 0,
                        media_rate: 1.0,
                    }],
                })),
                children: Vec::new(),
            }],
        }
    }

    fn create_media_atom(&self, chunk_offset: u64) -> Atom {
        let mdhd = self.create_media_header();
        let hdlr = self.create_handler_reference();
        let minf = self.create_media_information(chunk_offset);

        Atom {
            header: AtomHeader::new(FourCC(*MDIA)),
            data: None,
            children: vec![mdhd, hdlr, minf],
        }
    }

    fn create_media_header(&self) -> Atom {
        let mdhd = MediaHeaderAtom::builder()
            .creation_time(self.creation_time)
            .modification_time(self.modification_time)
            .timescale(self.timescale)
            .duration(self.media_duration)
            .language(self.language)
            .build();

        Atom {
            header: AtomHeader::new(mdhd.atom_type()),
            data: Some(AtomData::MediaHeader(mdhd)),
            children: vec![],
        }
    }

    fn create_handler_reference(&self) -> Atom {
        let hdlr = HandlerReferenceAtom::builder()
            .handler_type(HandlerType::Text)
            .name("SubtitleHandler")
            .build();

        Atom {
            header: AtomHeader::new(hdlr.atom_type()),
            data: Some(AtomData::HandlerReference(hdlr)),
            children: vec![],
        }
    }

    fn create_media_information(&self, chunk_offset: u64) -> Atom {
        let stbl = self.create_sample_table(chunk_offset);

        // For text tracks, we need a generic media information header
        let gmhd = GenericMediaHeaderAtom::new();

        // Create DINF (Data Information) with proper data reference
        let dref = DataReferenceAtom {
            version: 0,
            flags: [0u8; 3],
            entry_count: 1,
            entries: vec![DataReferenceEntry {
                inner: DataReferenceEntryInner::Url("".to_string()),
                version: 0,
                flags: [0, 0, 1], // Self-contained flag
            }],
        };

        let dinf = Atom {
            header: AtomHeader::new(FourCC(*DINF)),
            data: None,
            children: vec![Atom {
                header: AtomHeader::new(dref.atom_type()),
                data: Some(AtomData::DataReference(dref)),
                children: vec![],
            }],
        };

        Atom {
            header: AtomHeader::new(FourCC(*MINF)),
            data: None,
            children: vec![
                Atom {
                    header: AtomHeader::new(gmhd.atom_type()),
                    data: Some(AtomData::GenericMediaHeader(gmhd)),
                    children: vec![],
                },
                dinf,
                stbl,
            ],
        }
    }

    fn create_sample_table(&self, chunk_offset: u64) -> Atom {
        let stsd = self.create_sample_description();
        let stts = self.create_time_to_sample();
        let stsc = self.create_sample_to_chunk();
        let stsz = self.create_sample_size();
        let stco = self.create_chunk_offset(chunk_offset);

        Atom {
            header: AtomHeader::new(FourCC(*STBL)),
            data: None,
            children: vec![stsd, stts, stsc, stsz, stco],
        }
    }

    fn create_sample_description(&self) -> Atom {
        let text_sample_entry = SampleEntry {
            entry_type: SampleEntryType::Text,
            data_reference_index: 1, // References the data reference entry we created
            data: SampleEntryData::Text(TextSampleEntry {
                version: 0,
                revision_level: 1,
                vendor: [0u8; 4],
                display_flags: 0,
                text_justification: 0,
                background_color: [0u16; 3], // Black background (RGB)
                default_text_box: [0, 0, 256, 0],
                default_style: Some(vec![
                    0, 0, // Start character (0)
                    0, 0, // End character (0)
                    0, 0,  // Font ID (0)
                    13, // Font size (13pt)
                    102, 116, 97, 98, // Font name: "ftab" (first 4 chars)
                    0,  // Font face flags
                    1,  // Red color component
                    0,  // Green color component
                    1,  // Blue color component
                    0,  // Alpha/padding
                ]),
                font_table: None,
                extensions: Vec::new(),
            }),
        };

        let stsd = SampleDescriptionTableAtom::from(vec![text_sample_entry]);

        Atom {
            header: AtomHeader::new(stsd.atom_type()),
            data: Some(AtomData::SampleDescriptionTable(stsd)),
            children: vec![],
        }
    }

    fn create_time_to_sample(&self) -> Atom {
        let entries: Vec<TimeToSampleEntry> = self
            .sample_durations
            .iter()
            .map(|&duration| TimeToSampleEntry {
                sample_count: 1,
                sample_duration: duration,
            })
            .collect();

        let stts = TimeToSampleAtom::from(entries);

        Atom {
            header: AtomHeader::new(stts.atom_type()),
            data: Some(AtomData::TimeToSample(stts)),
            children: vec![],
        }
    }

    fn create_sample_to_chunk(&self) -> Atom {
        // Single chunk with all samples (like the working file)
        let stsc = SampleToChunkAtom::from(vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: self.sample_data.len() as u32,
            sample_description_index: 1,
        }]);

        Atom {
            header: AtomHeader::new(stsc.atom_type()),
            data: Some(AtomData::SampleToChunk(stsc)),
            children: vec![],
        }
    }

    fn create_sample_size(&self) -> Atom {
        // Since we're using fixed 45-byte samples, use constant size format
        let stsz = SampleSizeAtom::builder()
            .sample_size(45) // Fixed size like working file
            .sample_count(self.sample_data.len() as u32)
            .build();

        Atom {
            header: AtomHeader::new(stsz.atom_type()),
            data: Some(AtomData::SampleSize(stsz)),
            children: vec![],
        }
    }

    fn create_chunk_offset(&self, chunk_offset: u64) -> Atom {
        let stco = ChunkOffsetAtom::new(vec![chunk_offset]);

        Atom {
            header: AtomHeader::new(stco.atom_type()),
            data: Some(AtomData::ChunkOffset(stco)),
            children: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::chpl::ChapterEntry;

    #[test]
    fn test_chapter_track_creation() {
        let chapters = vec![
            ChapterEntry {
                start_time: 0,
                title: "Introduction".to_string(),
            },
            ChapterEntry {
                start_time: 300_000_000, // 30 seconds in 100ns units
                title: "Chapter 1".to_string(),
            },
            ChapterEntry {
                start_time: 600_000_000, // 60 seconds in 100ns units
                title: "Chapter 2".to_string(),
            },
        ];

        let track = ChapterTrack::builder()
            .track_id(2)
            .timescale(1000)
            .movie_timescale(600)
            .total_duration(Duration::from_secs(900))
            .build(chapters);

        // Test that all samples are exactly 45 bytes
        let individual_samples = track.individual_samples();
        assert_eq!(individual_samples.len(), 3);
        for sample in individual_samples {
            assert_eq!(sample.len(), 45);
        }

        // Test TRAK atom creation
        let trak_atom = track.create_trak_atom(1024);
        assert_eq!(trak_atom.header.atom_type, FourCC(*TRAK));
        assert_eq!(trak_atom.children.len(), 3); // TKHD, EDTS and MDIA
    }

    #[test]
    fn test_empty_chapters() {
        let track = ChapterTrack::builder()
            .total_duration(Duration::from_secs(0))
            .movie_timescale(600)
            .build(vec![]);

        assert!(track.sample_bytes().is_empty());
        assert!(track.individual_samples().is_empty());
    }

    #[test]
    fn test_chapter_marker_data_format() {
        let marker_data = ChapterTrack::create_chapter_marker_data("Test Chapter");

        // Should be exactly 45 bytes
        assert_eq!(marker_data.len(), 45);

        // First 2 bytes should be the text length in big-endian
        let text_len = u16::from_be_bytes([marker_data[0], marker_data[1]]);
        assert_eq!(text_len, "Test Chapter".len() as u16);

        // Next bytes should be the text
        let text_bytes = &marker_data[2..2 + text_len as usize];
        assert_eq!(text_bytes, "Test Chapter".as_bytes());
    }

    #[test]
    fn test_long_chapter_title_truncation() {
        let long_title = "This is a very long chapter title that exceeds the normal expected length and should be truncated";
        let marker_data = ChapterTrack::create_chapter_marker_data(long_title);

        // Should still be exactly 45 bytes
        assert_eq!(marker_data.len(), 45);

        // Text length should be truncated to fit
        let text_len = u16::from_be_bytes([marker_data[0], marker_data[1]]);
        assert!(text_len <= 43); // 45 - 2 bytes for length
    }

    #[test]
    fn test_duration_calculations() {
        let chapters = vec![
            ChapterEntry {
                start_time: 0,
                title: "Chapter 1".to_string(),
            },
            ChapterEntry {
                start_time: 300_000_000, // 30 seconds in 100ns units
                title: "Chapter 2".to_string(),
            },
        ];

        let track = ChapterTrack::builder()
            .timescale(1000) // 1000 units per second
            .total_duration(Duration::from_secs(120)) // 2 minutes total
            .movie_timescale(600)
            .build(chapters);

        // Verify we have 2 samples with correct durations
        assert_eq!(track.sample_durations.len(), 2);

        // Chapter 1: 0s to 30s = 30 seconds = 30,000 units at timescale 1000
        assert_eq!(track.sample_durations[0], 30_000);

        // Chapter 2: 30s to 120s = 90 seconds = 90,000 units at timescale 1000
        assert_eq!(track.sample_durations[1], 90_000);

        // Total should match media duration
        let total_sample_duration: u32 = track.sample_durations.iter().sum();
        assert_eq!(total_sample_duration, track.media_duration as u32);
    }
}
