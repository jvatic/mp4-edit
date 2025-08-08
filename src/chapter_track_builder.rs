use std::time::Duration;

use bon::bon;

use crate::atom::{
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

/// Represents a chapter from input JSON (milliseconds-based)
#[derive(Debug, Clone)]
pub struct InputChapter {
    pub title: String,
    pub offset_ms: u64,
    pub duration_ms: u64,
}

#[bon]
impl InputChapter {
    #[builder]
    pub fn new(
        #[builder(into, finish_fn)] title: String,
        start_offset: Duration,
        duration: Duration,
    ) -> Self {
        Self {
            title,
            offset_ms: start_offset.as_millis() as u64,
            duration_ms: duration.as_millis() as u64,
        }
    }
}

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
        #[builder(finish_fn, into)] chapters: Vec<InputChapter>,
        #[builder(default = LanguageCode::Undetermined)] language: LanguageCode,
        #[builder(default = 44100)] timescale: u32, // Match audio timescale from dump
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
    /// This creates a contiguous timeline by extending chapters to fill gaps
    fn create_samples_and_durations(
        chapters: &[InputChapter],
        total_duration: Duration,
        timescale: u32,
    ) -> (Vec<Vec<u8>>, Vec<u32>) {
        if chapters.is_empty() {
            return (vec![], vec![]);
        }

        let total_duration_ms = total_duration.as_millis() as u64;
        let mut samples = Vec::new();
        let mut durations = Vec::new();

        for (i, chapter) in chapters.iter().enumerate() {
            // Create chapter marker data (45 bytes like working file)
            let chapter_marker = Self::create_chapter_marker_data(&chapter.title);
            samples.push(chapter_marker);

            // Calculate the end time for this chapter
            let chapter_end_ms = if i + 1 < chapters.len() {
                // Extend this chapter until the next chapter starts
                chapters[i + 1].offset_ms
            } else {
                // Last chapter extends to the end of the media
                total_duration_ms
            };

            // Calculate the actual duration this chapter should span
            let actual_duration_ms = chapter_end_ms - chapter.offset_ms;
            let chapter_duration_seconds = actual_duration_ms as f64 / 1000.0;
            let chapter_duration_scaled = (chapter_duration_seconds * timescale as f64) as u32;

            durations.push(chapter_duration_scaled);
        }

        (samples, durations)
    }

    /// Create standardized chapter marker data matching QuickTime format
    fn create_chapter_marker_data(title: &str) -> Vec<u8> {
        let mut data = Vec::with_capacity(45);

        // QuickTime text sample format:
        // - 2 bytes: text length (big-endian)
        // - N bytes: UTF-8 text
        // - Padding to reach 45 bytes total

        let title_bytes = title.as_bytes();
        let text_len = title_bytes.len().min(43) as u16; // Leave 2 bytes for length

        // Write text length
        data.extend_from_slice(&text_len.to_be_bytes());

        // Write text data
        data.extend_from_slice(&title_bytes[..text_len as usize]);

        // Pad to exactly 45 bytes (matching the working file)
        while data.len() < 45 {
            data.push(0);
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
            flags: [0, 0, 2], // Track enabled, not in movie (matches working file)
            creation_time: self.creation_time,
            modification_time: self.modification_time,
            track_id: self.track_id,
            duration: self.movie_duration,
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
            .name("SubtitleHandler") // Matches working file
            .build();

        Atom {
            header: AtomHeader::new(hdlr.atom_type()),
            data: Some(AtomData::HandlerReference(hdlr)),
            children: vec![],
        }
    }

    fn create_media_information(&self, chunk_offset: u64) -> Atom {
        let stbl = self.create_sample_table(chunk_offset);

        // Generic media header for text tracks (matches working file)
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
        // Text sample entry matching the working file format
        let text_sample_entry = SampleEntry {
            entry_type: SampleEntryType::Text,
            data_reference_index: 1,
            data: SampleEntryData::Text(TextSampleEntry {
                version: 0,
                revision_level: 1, // Matches working file
                vendor: [0u8; 4],
                display_flags: 0,
                text_justification: 0,
                background_color: [0u16; 3],      // Black background
                default_text_box: [0, 0, 256, 0], // Matches working file
                default_style: Some(vec![
                    0, 0, 0, 0, // Start/end character indices
                    0, 0,  // Font ID
                    13, // Font size (matches working file)
                    102, 116, 97, 98, // "ftab" font name
                    0,  // Font face flags
                    1, 0, 1, 0, // Color: red=1, green=0, blue=1, alpha=0
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
        // Create individual entries for each sample (like working file)
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
        // Single chunk with all samples (matches working file pattern)
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
        // Fixed size of 45 bytes per sample (matches working file)
        let stsz = SampleSizeAtom::builder()
            .sample_size(45) // Constant size
            .sample_count(self.sample_data.len() as u32)
            .entry_sizes(vec![]) // Empty when using constant size
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

    #[test]
    fn test_chapter_track_with_millisecond_input() {
        let chapters = vec![
            InputChapter {
                title: "Opening Credits".to_string(),
                offset_ms: 0,
                duration_ms: 19758,
            },
            InputChapter {
                title: "Dedication".to_string(),
                offset_ms: 19758,
                duration_ms: 4510,
            },
            InputChapter {
                title: "Epigraph".to_string(),
                offset_ms: 24268,
                duration_ms: 12364,
            },
        ];

        let total_duration = Duration::from_millis(36632);

        let track = ChapterTrack::builder()
            .track_id(2)
            .timescale(44100) // Match audio timescale
            .movie_timescale(600)
            .total_duration(total_duration)
            .build(chapters);

        // Verify sample count and sizes
        assert_eq!(track.individual_samples().len(), 3);
        for sample in track.individual_samples() {
            assert_eq!(sample.len(), 45);
        }

        // Verify duration calculations
        let expected_durations = [
            (19.758 * 44100.0) as u32, // ~871 samples
            (4.510 * 44100.0) as u32,  // ~199 samples
            (12.364 * 44100.0) as u32, // ~545 samples
        ];

        for (i, &expected) in expected_durations.iter().enumerate() {
            assert!(
                (track.sample_durations[i] as f32 - expected as f32).abs() < 100.0,
                "Duration mismatch for chapter {}: expected ~{}, got {}",
                i,
                expected,
                track.sample_durations[i]
            );
        }
    }

    #[test]
    fn test_chapter_marker_format() {
        let title = "Test Chapter";
        let data = ChapterTrack::create_chapter_marker_data(title);

        assert_eq!(data.len(), 45);

        // Verify text length encoding
        let text_len = u16::from_be_bytes([data[0], data[1]]);
        assert_eq!(text_len, title.len() as u16);

        // Verify text content
        let text_data = &data[2..2 + text_len as usize];
        assert_eq!(text_data, title.as_bytes());

        // Verify padding
        let padding = &data[2 + text_len as usize..];
        assert!(padding.iter().all(|&b| b == 0));
    }
}
