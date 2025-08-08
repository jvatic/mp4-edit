use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bon::bon;

use crate::atom::{
    chpl::ChapterEntries,
    containers::*,
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
    util::time::mp4_timestamp_now,
    Atom, AtomData, AtomHeader, DataReferenceAtom, FourCC, RawData,
};
use crate::writer::SerializeAtom;

/// Represents a chapter track that can generate TRAK atoms and sample data
pub struct ChapterTrack {
    language: LanguageCode,
    timescale: u32,
    track_id: u32,
    creation_time: u64,
    modification_time: u64,
    sample_data: Vec<u8>,
    total_duration: u64,
}

#[bon]
impl ChapterTrack {
    #[builder]
    pub fn new(
        #[builder(finish_fn, into)] chapters: ChapterEntries,
        #[builder(default = LanguageCode::Undetermined)] language: LanguageCode,
        #[builder(default = 1000)] timescale: u32,
        #[builder(default = 2)] track_id: u32,
        #[builder(into)] total_duration: Duration,
        #[builder(default = mp4_timestamp_now())] creation_time: u64,
        #[builder(default = mp4_timestamp_now())] modification_time: u64,
    ) -> Self {
        let sample_data = Self::create_sample_data(&chapters);

        const NANOS_PER_SECOND: u128 = 1_000_000_000;
        let total_duration_nanos = total_duration.as_nanos();
        let timescale_nanos = timescale as u128;
        let total_duration = (total_duration_nanos * timescale_nanos / NANOS_PER_SECOND) as u64;

        Self {
            language,
            timescale,
            track_id,
            creation_time,
            modification_time,
            sample_data,
            total_duration,
        }
    }

    /// Create sample data from chapters - using a simple text format
    fn create_sample_data(chapters: &ChapterEntries) -> Vec<u8> {
        let mut data = Vec::new();

        // Simple text format: each line is "start_time_ms:title"
        for (i, chapter) in chapters.iter().enumerate() {
            if i > 0 {
                data.push(b'\n');
            }

            // Convert start_time (in 100-nanosecond units) to milliseconds
            let start_time_ms = chapter.start_time / 10_000;
            let line = format!("{}:{}", start_time_ms, chapter.title);
            data.extend_from_slice(line.as_bytes());
        }

        data
    }

    /// Returns the sample data bytes
    pub fn sample_bytes(&self) -> &[u8] {
        self.sample_data.as_slice()
    }

    /// Creates a TRAK atom for the chapter track at the specified chunk offset
    pub fn create_trak_atom(&self, chunk_offset: u64) -> Atom {
        let track_header = self.create_track_header();
        let media_atom = self.create_media_atom(chunk_offset);

        Atom {
            header: AtomHeader {
                atom_type: FourCC(*TRAK),
                offset: 0,
                header_size: 8,
                data_size: 0, // Will be calculated when serialized
            },
            data: None,
            children: vec![track_header, media_atom],
        }
    }

    fn create_track_header(&self) -> Atom {
        let tkhd = TrackHeaderAtom {
            version: 0,
            flags: [0, 0, 0x07], // Track enabled, in movie, in preview
            creation_time: self.creation_time,
            modification_time: self.modification_time,
            track_id: self.track_id,
            duration: self.total_duration,
            layer: 0,
            alternate_group: 0,
            volume: 0.0,  // Text track has no volume
            matrix: None, // Use default identity matrix
            width: 0.0,   // Text track dimensions
            height: 0.0,
        };

        Atom {
            header: AtomHeader {
                atom_type: tkhd.atom_type(),
                offset: 0,
                header_size: 8,
                data_size: 0,
            },
            data: Some(AtomData::TrackHeader(tkhd)),
            children: vec![],
        }
    }

    fn create_media_atom(&self, chunk_offset: u64) -> Atom {
        let mdhd = self.create_media_header();
        let hdlr = self.create_handler_reference();
        let minf = self.create_media_information(chunk_offset);

        Atom {
            header: AtomHeader {
                atom_type: FourCC(*MDIA),
                offset: 0,
                header_size: 8,
                data_size: 0,
            },
            data: None,
            children: vec![mdhd, hdlr, minf],
        }
    }

    fn create_media_header(&self) -> Atom {
        let mdhd = MediaHeaderAtom::builder()
            .timescale(self.timescale)
            .duration(self.total_duration)
            .language(self.language)
            .build();

        Atom {
            header: AtomHeader {
                atom_type: mdhd.atom_type(),
                offset: 0,
                header_size: 8,
                data_size: 0,
            },
            data: Some(AtomData::MediaHeader(mdhd)),
            children: vec![],
        }
    }

    fn create_handler_reference(&self) -> Atom {
        let hdlr = HandlerReferenceAtom::builder()
            .handler_type(HandlerType::Text)
            .name("ChapterListHandler")
            .build();

        Atom {
            header: AtomHeader {
                atom_type: hdlr.atom_type(),
                offset: 0,
                header_size: 8,
                data_size: 0,
            },
            data: Some(AtomData::HandlerReference(hdlr)),
            children: vec![],
        }
    }

    fn create_media_information(&self, chunk_offset: u64) -> Atom {
        let stbl = self.create_sample_table(chunk_offset);

        // For text tracks, we need a null media information header
        let nmhd = RawData::new(FourCC(*b"nmhd"), vec![0u8; 8]);

        // Create DINF (Data Information)
        let dref = DataReferenceAtom {
            version: 0,
            flags: [0u8; 3],
            entry_count: 0,
            entries: vec![],
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
                    header: AtomHeader::new(nmhd.atom_type()),
                    data: Some(AtomData::RawData(nmhd)),
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
            header: AtomHeader {
                atom_type: FourCC(*STBL),
                offset: 0,
                header_size: 8,
                data_size: 0,
            },
            data: None,
            children: vec![stsd, stts, stsc, stsz, stco],
        }
    }

    fn create_sample_description(&self) -> Atom {
        let text_sample_entry = SampleEntry {
            entry_type: SampleEntryType::Text,
            data_reference_index: 1, // 1 == self-contained
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
        let stts = TimeToSampleAtom::from(vec![TimeToSampleEntry {
            sample_count: 1,
            sample_duration: self.total_duration as u32,
        }]);

        Atom {
            header: AtomHeader::new(stts.atom_type()),
            data: Some(AtomData::TimeToSample(stts)),
            children: vec![],
        }
    }

    fn create_sample_to_chunk(&self) -> Atom {
        // Single chunk with single sample
        let stsc = SampleToChunkAtom::from(vec![SampleToChunkEntry {
            first_chunk: 1,
            samples_per_chunk: 1,
            sample_description_index: 1,
        }]);

        Atom {
            header: AtomHeader::new(stsc.atom_type()),
            data: Some(AtomData::SampleToChunk(stsc)),
            children: vec![],
        }
    }

    fn create_sample_size(&self) -> Atom {
        let stsz = SampleSizeAtom::builder()
            .sample_count(1)
            .sample_size(self.sample_data.len() as u32)
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
            .total_duration(Duration::from_secs(900))
            .build(chapters);

        // Test sample data creation
        let sample_data = track.sample_bytes();
        let sample_text = String::from_utf8_lossy(sample_data);

        assert!(sample_text.contains("0:Introduction"));
        assert!(sample_text.contains("30000:Chapter 1"));
        assert!(sample_text.contains("60000:Chapter 2"));

        // Test TRAK atom creation
        let trak_atom = track.create_trak_atom(1024);
        assert_eq!(trak_atom.header.atom_type, FourCC(*TRAK));
        assert_eq!(trak_atom.children.len(), 2); // TKHD and MDIA
    }

    #[test]
    fn test_empty_chapters() {
        let track = ChapterTrack::builder()
            .total_duration(Duration::from_secs(0))
            .build(vec![]);

        assert!(track.sample_bytes().is_empty());
    }

    #[test]
    fn test_sample_data_format() {
        let chapters = vec![ChapterEntry {
            start_time: 0,
            title: "Test Chapter".to_string(),
        }];

        let track = ChapterTrack::builder()
            .total_duration(Duration::from_secs(300))
            .build(chapters);
        let sample_text = String::from_utf8_lossy(track.sample_bytes());

        assert_eq!(sample_text, "0:Test Chapter");
    }
}
