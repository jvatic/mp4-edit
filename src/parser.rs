use derive_more::Display;
use futures_io::AsyncRead;
use futures_util::io::{AsyncReadExt, Cursor};
use futures_util::stream::Stream;
use std::future::Future;
use std::ops::Deref;
use thiserror::Error;

use crate::{
    atom::{
        chpl::{ChapterListAtom, CHPL},
        dref::{DataReferenceAtom, DREF},
        elst::{EditListAtom, ELST},
        free::{FreeAtom, FREE, SKIP},
        ftyp::{FileTypeAtom, FTYP},
        gmhd::{GenericMediaHeaderAtom, GMHD},
        hdlr::{HandlerReferenceAtom, HDLR},
        ilst::{ItemListAtom, ILST},
        mdhd::{MediaHeaderAtom, MDHD},
        meta::{META, META_VERSION_FLAGS_SIZE},
        mvhd::{MovieHeaderAtom, MVHD},
        sbgp::{SampleToGroupAtom, SBGP},
        sgpd::{SampleGroupDescriptionAtom, SGPD},
        smhd::{SoundMediaHeaderAtom, SMHD},
        stco_co64::{ChunkOffsetAtom, CO64, STCO},
        stsc::{SampleToChunkAtom, STSC},
        stsd::{SampleDescriptionTableAtom, STSD},
        stsz::{SampleSizeAtom, STSZ},
        stts::{TimeToSampleAtom, STTS},
        tkhd::{TrackHeaderAtom, TKHD},
        tref::{TrackReferenceAtom, TREF},
        FourCC, RawData,
    },
    Atom, AtomData,
};

const MDAT: &[u8; 4] = b"mdat";

/// Async trait for parsing atoms from an AsyncRead stream
pub trait Parse: Sized {
    fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> impl Future<Output = Result<Self, anyhow::Error>> + Send;
}

#[derive(Debug, Error)]
#[error(
    "{kind}{}",
    self.location.map(|(offset, length)|
        format!(" at offset {offset} with length {length}")).unwrap_or_default()
)]
pub struct ParseError {
    /// The kind of error that occurred during parsing.
    kind: ParseErrorKind,
    /// location is the (offset, length) of the input data related to the error
    location: Option<(usize, usize)>,
    /// The source error that caused this error.
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

#[derive(Debug, Display)]
pub enum ParseErrorKind {
    #[display("I/O error")]
    Io,
    #[display("EOF error")]
    Eof,
    #[display("Invalid atom header")]
    InvalidHeader,
    #[display("Invalid atom size")]
    InvalidSize,
    #[display("Unsupported atom type")]
    UnsupportedAtom,
    #[display("Atom parsing failed")]
    AtomParsing,
    #[display("Insufficient data")]
    InsufficientData,
}

pub struct Sample {
    pub data: Vec<u8>,
    pub duration: u32,
    pub description_index: u32,
    pub sample_number: u32,
    pub timestamp: u64,
}

impl Sample {
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

pub struct Parser<R> {
    reader: R,
    state: ParserState,
    current_offset: usize,
    peek_buffer: Vec<u8>,
}

enum ParserState {
    NotStarted,
    MetaData,
    MediaData,
}

struct ParsedAtom {
    atom_type: FourCC,
    size: u64,
    offset: u64,
    content_size: usize,
}

impl<R: AsyncRead + Unpin + Send> Parser<R> {
    pub fn new(reader: R) -> Self {
        Parser {
            reader: reader,
            state: ParserState::NotStarted,
            current_offset: 0,
            peek_buffer: Vec::new(),
        }
    }

    pub async fn parse_metadata<'a>(&'a mut self) -> Result<Vec<Atom>, ParseError> {
        if !matches!(self.state, ParserState::NotStarted) {
            panic!("parser in an invalid state");
        }
        self.state = ParserState::MetaData;
        self.parse_metadata_inner(None).await
    }

    async fn parse_metadata_inner<'a>(
        &'a mut self,
        length_limit: Option<usize>,
    ) -> Result<Vec<Atom>, ParseError> {
        let start_offset = self.current_offset;

        let mut top_level_atoms = Vec::new();

        loop {
            // ensure we're respecting container bounds
            if length_limit.is_some_and(|limit| self.current_offset - start_offset >= limit) {
                break;
            }

            // only parse as far as the mdat atom (we're assuming mdat is the last atom)
            let next_atom_type = match self.peek_next_atom_type().await {
                Ok(next_atom_type) => Ok(next_atom_type),
                Err(err) => {
                    if matches!(err.kind, ParseErrorKind::Eof) {
                        // end of stream, this means there's no mdat atom
                        // TODO: rewrite the tests to always include an mdat atom so we can get rid of this check
                        break;
                    }
                    Err(err)
                }
            }?;
            if next_atom_type == MDAT {
                break;
            }

            let parsed_atom = self.parse_next_atom().await?;
            if is_container_atom(&parsed_atom.atom_type) {
                // META containers have additional header data
                let size = if parsed_atom.atom_type.deref() == META {
                    // Ignore META version and flags
                    self.read_data(META_VERSION_FLAGS_SIZE).await?;
                    parsed_atom.content_size - META_VERSION_FLAGS_SIZE
                } else {
                    parsed_atom.content_size
                };

                let container_atom = Atom {
                    atom_type: parsed_atom.atom_type,
                    size: parsed_atom.size,
                    offset: parsed_atom.offset,
                    data: None,
                    children: Box::pin(self.parse_metadata_inner(Some(size))).await?,
                };

                top_level_atoms.push(container_atom);
            } else {
                // Yield leaf atoms
                let atom_type = parsed_atom.atom_type;
                let offset = parsed_atom.offset;
                let size = parsed_atom.size;
                let atom_data = self.parse_atom_data(parsed_atom).await?;
                let atom = Atom {
                    atom_type,
                    offset,
                    size,
                    data: Some(atom_data),
                    children: Vec::new(),
                };
                top_level_atoms.push(atom);
            }
        }

        Ok(top_level_atoms)
    }

    pub fn stream_samples<'a>(
        &'a mut self,
        sample_sizes: SampleSizeAtom,
        sample_durations: TimeToSampleAtom,
        chunk_offsets: ChunkOffsetAtom,
        sample_to_chunk: SampleToChunkAtom,
    ) -> impl Stream<Item = Result<Sample, ParseError>> + 'a {
        if !matches!(self.state, ParserState::MetaData) {
            panic!("parser in an invalid state");
        }
        self.state = ParserState::MediaData;
        async_stream::stream! {
            let mdat_atom = self.parse_next_atom().await?;
            if mdat_atom.atom_type != MDAT {
                panic!("expected mdat to be the next atom");
            }

            let sample_count = sample_sizes.sample_count;
            let chunk_offsets = chunk_offsets.chunk_offsets.into_inner();

            // Calculate individual sample sizes
            let sample_sizes_vec: Vec<u32> = if sample_sizes.sample_size != 0 {
                vec![sample_sizes.sample_size; sample_count as usize]
            } else {
                sample_sizes.entry_sizes.to_vec()
            };

            // Expand time-to-sample entries into per-sample durations
            let mut sample_durations_vec = Vec::with_capacity(sample_count as usize);
            for entry in sample_durations.entries.iter() {
                for _ in 0..entry.sample_count {
                    sample_durations_vec.push(entry.sample_duration);
                }
            }

            // Build chunk-to-samples mapping
            let mut chunk_samples: Vec<Vec<(u32, u32)>> = vec![Vec::new(); chunk_offsets.len()];
            let mut current_sample = 0u32;

            for (i, entry) in sample_to_chunk.entries.iter().enumerate() {
                let first_chunk = (entry.first_chunk - 1) as usize; // Convert to 0-based
                let samples_per_chunk = entry.samples_per_chunk;
                let description_index = entry.sample_description_index;

                // Determine the last chunk that uses this entry
                let last_chunk = if i + 1 < sample_to_chunk.entries.len() {
                    (sample_to_chunk.entries[i + 1].first_chunk - 2) as usize // -1 for 0-based, -1 for exclusive
                } else {
                    chunk_offsets.len() - 1
                };

                for chunk_idx in first_chunk..=last_chunk {
                    if chunk_idx >= chunk_samples.len() {
                        break;
                    }
                    for _ in 0..samples_per_chunk {
                        if current_sample >= sample_count {
                            break;
                        }
                        chunk_samples[chunk_idx].push((current_sample, description_index));
                        current_sample += 1;
                    }
                    if current_sample >= sample_count {
                        break;
                    }
                }
            }

            // Process chunks in order and yield samples
            let mut timestamp = 0u64;
            for (chunk_idx, chunk_sample_list) in chunk_samples.iter().enumerate() {
                if chunk_sample_list.is_empty() {
                    continue;
                }

                // Calculate chunk size
                let chunk_start = chunk_offsets[chunk_idx];
                let chunk_end = if chunk_idx + 1 < chunk_offsets.len() {
                    chunk_offsets[chunk_idx + 1]
                } else {
                    mdat_atom.offset + mdat_atom.size
                };
                let chunk_size = chunk_end - chunk_start;

                // Read entire chunk
                let chunk_data = self.read_data(chunk_size as usize).await?;

                // Parse samples from chunk
                let mut offset_in_chunk = 0usize;
                for &(sample_number, description_index) in chunk_sample_list {
                    let sample_size = sample_sizes_vec[sample_number as usize];
                    let sample_duration = sample_durations_vec[sample_number as usize];

                    if offset_in_chunk + sample_size as usize > chunk_data.len() {
                        yield Err(ParseError {
                            kind: ParseErrorKind::InsufficientData,
                            location: Some((self.current_offset, sample_size as usize)),
                            source: None,
                        });
                        return;
                    }

                    let sample_data = chunk_data[offset_in_chunk..offset_in_chunk + sample_size as usize].to_vec();

                    yield Ok(Sample {
                        data: sample_data,
                        duration: sample_duration,
                        description_index,
                        sample_number,
                        timestamp,
                    });

                    timestamp += sample_duration as u64;
                    offset_in_chunk += sample_size as usize;
                }
            }
        }
    }

    async fn peek_exact(&mut self, buf: &mut [u8]) -> Result<(), ParseError> {
        let size = buf.len();
        if self.peek_buffer.len() < size {
            let mut temp_buf = vec![0u8; size - self.peek_buffer.len()];
            self.reader.read_exact(&mut temp_buf).await.map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    return ParseError {
                        kind: ParseErrorKind::Eof,
                        location: Some((self.current_offset, size)),
                        source: Some(Box::new(e)),
                    };
                }
                ParseError {
                    kind: ParseErrorKind::Io,
                    location: Some((self.current_offset, size)),
                    source: Some(Box::new(e)),
                }
            })?;
            self.peek_buffer.extend_from_slice(&temp_buf[..]);
        }
        buf.copy_from_slice(&self.peek_buffer[..size]);
        Ok(())
    }

    async fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ParseError> {
        self.peek_exact(buf).await?;
        self.peek_buffer.drain(..buf.len());
        self.current_offset += buf.len();
        Ok(())
    }

    async fn read_data(&mut self, size: usize) -> Result<Vec<u8>, ParseError> {
        let mut data = vec![0u8; size];
        self.read_exact(&mut data).await?;
        Ok(data)
    }

    async fn peek_next_atom_type(&mut self) -> Result<FourCC, ParseError> {
        let mut header = [0u8; 8];
        self.peek_exact(&mut header).await?;
        Ok(FourCC([header[4], header[5], header[6], header[7]]))
    }

    async fn parse_next_atom(&mut self) -> Result<ParsedAtom, ParseError> {
        let atom_offset = self.current_offset as u64;

        // Try to read the atom header (size and type)
        let mut header = [0u8; 8];
        self.read_exact(&mut header).await?;

        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
        let atom_type: [u8; 4] = header[4..8].try_into().unwrap();

        // Handle extended size (64-bit) if needed
        let (header_size, data_size) = if size == 1 {
            // Extended size format
            let mut extended_size = [0u8; 8];
            self.read_exact(&mut extended_size).await?;
            let full_size = u64::from_be_bytes(extended_size);
            if full_size < 16 {
                return Err(ParseError {
                    kind: ParseErrorKind::InvalidSize,
                    location: Some((atom_offset as usize, 16)),
                    source: None,
                });
            }
            (16u64, full_size - 16)
        } else if size == 0 {
            // Size extends to end of file - not supported in this context
            return Err(ParseError {
                kind: ParseErrorKind::InvalidSize,
                location: Some((atom_offset as usize, 8)),
                source: None,
            });
        } else {
            if size < 8 {
                return Err(ParseError {
                    kind: ParseErrorKind::InvalidSize,
                    location: Some((atom_offset as usize, 8)),
                    source: None,
                });
            }
            (8u64, size - 8)
        };

        let atom_type = FourCC(atom_type);

        let total_size = header_size + data_size;

        Ok(ParsedAtom {
            atom_type,
            size: total_size,
            offset: atom_offset,
            content_size: data_size as usize,
        })
    }

    async fn parse_atom_data(&mut self, parsed_atom: ParsedAtom) -> Result<AtomData, ParseError> {
        let content_data = self.read_data(parsed_atom.content_size).await?;
        let cursor = Cursor::new(content_data);
        let atom_type = parsed_atom.atom_type;
        let atom_data = match atom_type.deref() {
            FTYP => FileTypeAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            MVHD => MovieHeaderAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            MDHD => MediaHeaderAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            ELST => EditListAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            HDLR => HandlerReferenceAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            GMHD => GenericMediaHeaderAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            SMHD => SoundMediaHeaderAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            ILST => ItemListAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            TKHD => TrackHeaderAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            STSD => SampleDescriptionTableAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            TREF => TrackReferenceAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            DREF => DataReferenceAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            STSZ => SampleSizeAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            STCO | CO64 => ChunkOffsetAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            STTS => TimeToSampleAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            STSC => SampleToChunkAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            CHPL => ChapterListAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            SGPD => SampleGroupDescriptionAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            SBGP => SampleToGroupAtom::parse(atom_type, cursor)
                .await
                .map(AtomData::from),
            FREE | SKIP => FreeAtom::parse(atom_type, cursor).await.map(AtomData::from),
            _ => Ok(RawData(cursor.get_ref().clone()).into()),
        }
        .map_err(|e| ParseError {
            kind: ParseErrorKind::AtomParsing,
            location: Some((parsed_atom.offset as usize, parsed_atom.size as usize)),
            source: Some(e.context(atom_type).into()),
        })?;

        Ok(atom_data)
    }
}

/// Determines whether a given atom type (fourcc) should be treated as a container for other atoms.
fn is_container_atom(atom_type: &[u8; 4]) -> bool {
    // Common container types in MP4
    matches!(
        atom_type,
        b"moov"
            | b"mfra"
            | b"udta"
            | b"trak"
            | b"edts"
            | b"mdia"
            | b"minf"
            | b"dinf"
            | b"stbl"
            | b"moof"
            | b"traf"
            | b"sinf"
            | b"schi"
            | META
    )
}
