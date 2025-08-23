use std::time::Duration;

use bon::bon;

use crate::atom::{
    containers::{DINF, EDTS, MDIA, MINF, STBL, TRAK},
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
    Atom, AtomData, AtomHeader, DataReferenceAtom, EditListAtom,
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

/// Configuration for QuickTime text sample entry formatting
#[derive(Debug, Clone)]
pub struct TextSampleConfig {
    /// Minimum padding bytes to add after text content (for compatibility)
    pub min_padding: u16,
    /// Font size for chapter text display
    pub font_size: u8,
    /// Default text box dimensions [top, left, bottom, right]
    pub text_box: [u16; 4],
    /// Text color as RGBA values [red, green, blue, alpha]
    pub text_color: [u8; 4],
    /// Font name bytes (e.g., b"ftab" for default)
    pub font_name: [u8; 4],
}

impl Default for TextSampleConfig {
    fn default() -> Self {
        Self {
            min_padding: 4, // Small padding for compatibility
            font_size: 13,
            text_box: [0, 0, 256, 0],
            text_color: [1, 0, 1, 0],      // Magenta with no alpha
            font_name: [102, 116, 97, 98], // "ftab"
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
    sample_sizes: Vec<u32>, // Individual sample sizes
    media_duration: u64,
    movie_duration: u64,
    text_config: TextSampleConfig,
    handler_name: String,
}

#[bon]
impl ChapterTrack {
    #[builder]
    pub fn new(
        #[builder(finish_fn, into)] chapters: Vec<InputChapter>,
        #[builder(default = LanguageCode::Undetermined)] language: LanguageCode,
        #[builder(default = 44100)] timescale: u32,
        #[builder(default = 600)] movie_timescale: u32,
        #[builder(default = 2)] track_id: u32,
        #[builder(into)] total_duration: Duration,
        #[builder(default = mp4_timestamp_now())] creation_time: u64,
        #[builder(default = mp4_timestamp_now())] modification_time: u64,
        #[builder(default = TextSampleConfig::default())] text_config: TextSampleConfig,
        #[builder(default = "SubtitleHandler".to_string(), into)] handler_name: String,
    ) -> Self {
        let (sample_data, sample_durations, sample_sizes) =
            Self::create_samples_durations_and_sizes(
                &chapters,
                total_duration,
                timescale,
                &text_config,
            );

        Self {
            language,
            timescale,
            track_id,
            creation_time,
            modification_time,
            sample_data,
            sample_durations,
            sample_sizes,
            // Calculate duration in media timescale (for MDHD and STTS)
            media_duration: scaled_duration(total_duration, u64::from(timescale)),
            // Calculate duration in movie timescale (for TKHD)
            movie_duration: scaled_duration(total_duration, u64::from(movie_timescale)),
            text_config,
            handler_name,
        }
    }

    /// Create chapter marker samples with variable sizes based on content
    /// This creates a contiguous timeline by extending chapters to fill gaps
    fn create_samples_durations_and_sizes(
        chapters: &[InputChapter],
        total_duration: Duration,
        timescale: u32,
        text_config: &TextSampleConfig,
    ) -> (Vec<Vec<u8>>, Vec<u32>, Vec<u32>) {
        if chapters.is_empty() {
            return (vec![], vec![], vec![]);
        }

        let total_duration_ms = total_duration.as_millis() as u64;
        let mut samples = Vec::new();
        let mut durations = Vec::new();
        let mut sizes = Vec::new();

        for (i, chapter) in chapters.iter().enumerate() {
            // Create chapter marker data with variable size based on content
            let chapter_marker = Self::create_variable_chapter_marker(&chapter.title, text_config);
            let sample_size = chapter_marker.len() as u32;

            samples.push(chapter_marker);
            sizes.push(sample_size);

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
            let chapter_duration_scaled = (chapter_duration_seconds * f64::from(timescale)) as u32;

            durations.push(chapter_duration_scaled);
        }

        (samples, durations, sizes)
    }

    /// Create variable-size chapter marker data based on actual text content
    fn create_variable_chapter_marker(title: &str, config: &TextSampleConfig) -> Vec<u8> {
        // QuickTime text sample format:
        // - 2 bytes: text length (big-endian)
        // - N bytes: UTF-8 text
        // - Optional padding for compatibility

        let title_bytes = title.as_bytes();
        let text_len = title_bytes.len() as u16;

        // Calculate total size: text length field + text content + minimum padding
        let total_size = 2 + title_bytes.len() + config.min_padding as usize;
        let mut data = Vec::with_capacity(total_size);

        // Write text length (big-endian)
        data.extend_from_slice(&text_len.to_be_bytes());

        // Write text data
        data.extend_from_slice(title_bytes);

        // Add minimum padding for compatibility
        data.resize(data.len() + config.min_padding as usize, 0);

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

    /// Returns the individual sample sizes
    pub fn sample_sizes(&self) -> &[u32] {
        &self.sample_sizes
    }

    /// Returns the total size of all samples combined
    pub fn total_sample_size(&self) -> usize {
        self.sample_sizes.iter().map(|&s| s as usize).sum()
    }

    /// Returns the number of chapter samples
    pub fn sample_count(&self) -> u32 {
        self.sample_data.len() as u32
    }

    /// Check if all samples have the same size (for STSZ optimization)
    pub fn has_uniform_sample_size(&self) -> bool {
        if self.sample_sizes.is_empty() {
            return true;
        }
        let first_size = self.sample_sizes[0];
        self.sample_sizes.iter().all(|&size| size == first_size)
    }

    /// Get the uniform sample size if all samples are the same size
    pub fn uniform_sample_size(&self) -> Option<u32> {
        if self.has_uniform_sample_size() {
            self.sample_sizes.first().copied()
        } else {
            None
        }
    }

    /// Creates a TRAK atom for the chapter track at the specified chunk offset
    pub fn create_trak_atom(&self, chunk_offset: u64) -> Atom {
        let track_header = self.create_track_header();
        let edit_list = self.create_edit_list_atom();
        let media_atom = self.create_media_atom(chunk_offset);

        Atom {
            header: AtomHeader::new(*TRAK),
            data: None,
            children: vec![track_header, edit_list, media_atom],
        }
    }

    fn create_track_header(&self) -> Atom {
        let tkhd = TrackHeaderAtom {
            version: 0,
            flags: [0, 0, 2], // Track enabled, not in movie - standard for chapter tracks
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
            header: AtomHeader::new(*EDTS),
            data: None,
            children: vec![Atom {
                header: AtomHeader::new(*ELST),
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
            header: AtomHeader::new(*MDIA),
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
            .name(&self.handler_name)
            .build();

        Atom {
            header: AtomHeader::new(hdlr.atom_type()),
            data: Some(AtomData::HandlerReference(hdlr)),
            children: vec![],
        }
    }

    fn create_media_information(&self, chunk_offset: u64) -> Atom {
        let stbl = self.create_sample_table(chunk_offset);

        // Generic media header for text tracks
        let gmhd = GenericMediaHeaderAtom::new();

        // Create DINF (Data Information) with proper data reference
        let dref = DataReferenceAtom {
            version: 0,
            flags: [0u8; 3],
            entry_count: 1,
            entries: vec![DataReferenceEntry {
                inner: DataReferenceEntryInner::Url(String::new()),
                version: 0,
                flags: [0, 0, 1], // Self-contained flag
            }],
        };

        let dinf = Atom {
            header: AtomHeader::new(*DINF),
            data: None,
            children: vec![Atom {
                header: AtomHeader::new(dref.atom_type()),
                data: Some(AtomData::DataReference(dref)),
                children: vec![],
            }],
        };

        Atom {
            header: AtomHeader::new(*MINF),
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
            header: AtomHeader::new(*STBL),
            data: None,
            children: vec![stsd, stts, stsc, stsz, stco],
        }
    }

    fn create_sample_description(&self) -> Atom {
        // Text sample entry with configurable parameters
        let text_sample_entry = SampleEntry {
            entry_type: SampleEntryType::Text,
            data_reference_index: 1,
            data: SampleEntryData::Text(TextSampleEntry {
                version: 0,
                revision_level: 1,
                vendor: [0u8; 4],
                display_flags: 0,
                text_justification: 0,
                background_color: [0u16; 3], // Black background
                default_text_box: self.text_config.text_box,
                default_style: Some({
                    let mut style = vec![
                        0,
                        0,
                        0,
                        0, // Start/end character indices
                        0,
                        0, // Font ID
                        self.text_config.font_size,
                    ];
                    style.extend_from_slice(&self.text_config.font_name);
                    style.push(0); // Font face flags
                    style.extend_from_slice(&self.text_config.text_color);
                    style
                }),
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
        // Create individual entries for each sample duration
        // This is essential for proper chapter timing recognition in audiobook players
        let entries: Vec<TimeToSampleEntry> = self
            .sample_durations
            .iter()
            .map(|&duration| TimeToSampleEntry {
                sample_count: 1, // Each entry represents exactly 1 sample
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
        // All samples in a single chunk
        let stsc = SampleToChunkAtom::from(vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: self.sample_count(),
            sample_description_index: 1,
        }]);

        Atom {
            header: AtomHeader::new(stsc.atom_type()),
            data: Some(AtomData::SampleToChunk(stsc)),
            children: vec![],
        }
    }

    fn create_sample_size(&self) -> Atom {
        // Use variable sample sizes - this is the key improvement
        let stsz = if let Some(uniform_size) = self.uniform_sample_size() {
            // Optimization: if all samples have the same size, use uniform mode
            SampleSizeAtom::builder()
                .sample_size(uniform_size)
                .sample_count(self.sample_count())
                .entry_sizes(vec![]) // Empty when using constant size
                .build()
        } else {
            // Use individual sample sizes for variable-length content
            SampleSizeAtom::builder()
                .sample_size(0) // 0 indicates variable sizes
                .sample_count(self.sample_count())
                .entry_sizes(self.sample_sizes.clone())
                .build()
        };

        Atom {
            header: AtomHeader::new(stsz.atom_type()),
            data: Some(AtomData::SampleSize(stsz)),
            children: vec![],
        }
    }

    fn create_chunk_offset(&self, chunk_offset: u64) -> Atom {
        // Single chunk containing all chapter marker samples
        let stco = ChunkOffsetAtom::builder()
            .chunk_offset(chunk_offset)
            .build();

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
    fn test_chapter_track_builder() {
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

        // Calculate total duration from the last chapter
        let last_chapter = chapters.last().unwrap();
        let total_duration =
            Duration::from_millis(last_chapter.offset_ms + last_chapter.duration_ms);

        let track = ChapterTrack::builder()
            .track_id(2)
            .timescale(44100) // Standard audio timescale
            .movie_timescale(600) // Standard movie timescale
            .total_duration(total_duration)
            .language(LanguageCode::Undetermined)
            .build(chapters);

        // Verify we have the correct number of samples
        assert_eq!(track.sample_count(), 3);

        // Verify sample sizes are based on chapter title length
        let samples = track.individual_samples();
        assert_eq!(samples.len(), 3);

        // Expected sizes: 2 bytes (length) + title length + 4 bytes (min_padding)
        // "Opening Credits" (15 chars) = 2 + 15 + 4 = 21 bytes
        // "Dedication" (10 chars) = 2 + 10 + 4 = 16 bytes
        // "Epigraph" (8 chars) = 2 + 8 + 4 = 14 bytes
        assert_eq!(samples[0].len(), 21);
        assert_eq!(samples[1].len(), 16);
        assert_eq!(samples[2].len(), 14);

        // Verify we have individual duration entries for each chapter
        assert_eq!(track.sample_durations.len(), 3);

        println!(
            "Generated {} chapter samples with durations: {:?}",
            track.sample_count(),
            track.sample_durations
        );
    }
}
