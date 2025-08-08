use anyhow::anyhow;
use derive_more::Display;
use futures_io::{AsyncRead, AsyncSeek};
use futures_util::io::{AsyncReadExt, AsyncSeekExt, Cursor};
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::io::SeekFrom;
use std::ops::{Deref, DerefMut};
use thiserror::Error;

use crate::atom::containers::{
    DINF, EDTS, MDIA, MFRA, MINF, MOOF, MOOV, SCHI, SINF, STBL, TRAF, TRAK, UDTA,
};
use crate::atom::elst::EditEntry;
use crate::atom::hdlr::HandlerType;
use crate::atom::stco_co64::ChunkOffsets;
use crate::atom::stsc::SampleToChunkEntry;
use crate::atom::stsd::{
    BtrtExtension, DecoderSpecificInfo, SampleEntryData, SampleEntryType, StsdExtension,
};
use crate::atom::stts::TimeToSampleEntry;
use crate::atom::tref::TrackReference;
use crate::atom::util::DebugEllipsis;
use crate::atom::AtomHeader;
use crate::chunk_offset_builder::{ChunkInfo, ChunkOffsetBuilder};
use crate::writer::SerializeAtom;
use crate::{
    atom::{
        chpl::{ChapterListAtom, CHPL},
        containers::{META, META_VERSION_FLAGS_SIZE},
        dref::{DataReferenceAtom, DREF},
        elst::{EditListAtom, ELST},
        free::{FreeAtom, FREE, SKIP},
        ftyp::{FileTypeAtom, FTYP},
        hdlr::{HandlerReferenceAtom, HDLR},
        ilst::{ItemListAtom, ILST},
        mdhd::{MediaHeaderAtom, MDHD},
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

pub const MDAT: &[u8; 4] = b"mdat";

/// Async trait for parsing atoms from an AsyncRead stream
pub trait Parse: Sized {
    fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> impl Future<Output = Result<Self, ParseError>> + Send;
}

#[derive(Debug, Error)]
#[error(
    "{kind}{}",
    self.location.map(|(offset, length)|
        format!(" at offset {offset} with length {length}")).unwrap_or_default()
)]
pub struct ParseError {
    /// The kind of error that occurred during parsing.
    pub(crate) kind: ParseErrorKind,
    /// location is the (offset, length) of the input data related to the error
    pub(crate) location: Option<(usize, usize)>,
    /// The source error that caused this error.
    #[source]
    pub(crate) source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

#[derive(Debug, Display)]
#[non_exhaustive]
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
    #[display("Unexpected atom type")]
    UnexpectedAtom,
    #[display("Atom parsing failed")]
    AtomParsing,
    #[display("Insufficient data")]
    InsufficientData,
    #[display("moov atom is missing")]
    MissingMoov,
}

impl ParseError {
    pub(crate) fn new_unexpected_atom(atom_type: FourCC, expected: &[u8; 4]) -> Self {
        let expected = FourCC::from(*expected);
        Self {
            kind: ParseErrorKind::UnexpectedAtom,
            location: Some((0, 4)),
            source: Some(anyhow!("expected {expected}, got {atom_type}").into_boxed_dyn_error()),
        }
    }

    pub(crate) fn new_atom_parse(source: anyhow::Error) -> Self {
        Self {
            kind: ParseErrorKind::AtomParsing,
            location: None,
            source: Some(source.into_boxed_dyn_error()),
        }
    }
}

pub struct Mp4Reader<R> {
    reader: R,
    current_offset: usize,
    peek_buffer: Vec<u8>,
}

impl<R: AsyncRead + Unpin + Send> Mp4Reader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            current_offset: 0,
            peek_buffer: Vec::new(),
        }
    }

    async fn seek(&mut self, pos: SeekFrom) -> Result<(), ParseError>
    where
        R: AsyncSeek,
    {
        match self.reader.seek(pos).await {
            Ok(offset) => {
                self.current_offset = offset as usize;
                self.peek_buffer = Vec::new();
                Ok(())
            }
            Err(err) => Err(ParseError {
                kind: ParseErrorKind::Io,
                location: None,
                source: Some(Box::new(err)),
            }),
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
}

pub struct Parser<R> {
    reader: Mp4Reader<R>,
    mdat: Option<AtomHeader>,
}

impl<R: AsyncRead + Unpin + Send> Parser<R> {
    pub fn new(reader: R) -> Self {
        Parser {
            reader: Mp4Reader::new(reader),
            mdat: None,
        }
    }

    /// parses metadata atoms, both before and after mdat if moov isn't found before
    pub async fn parse_metadata_seek(mut self) -> Result<MdatParser<R>, ParseError>
    where
        R: AsyncSeek,
    {
        let mut atoms = self.parse_metadata_inner(None).await?;
        let mdat = match self.mdat.take() {
            Some(mdat) if !atoms.iter().any(|a| a.header.atom_type == MOOV) => {
                // moov is likely after mdat, so skip to the end of the mdat atom and parse any atoms there
                self.reader
                    .seek(SeekFrom::Current(mdat.data_size as i64))
                    .await?;
                let end_atoms = self.parse_metadata_inner(None).await?;
                atoms.extend(end_atoms);
                // and then return to where we were
                self.reader
                    .seek(SeekFrom::Start((mdat.offset + mdat.header_size) as u64))
                    .await?;
                Some(mdat)
            }
            mdat => mdat,
        };
        Ok(MdatParser::new(self.reader, Metadata::new(atoms), mdat))
    }

    /// parses metadata atoms until mdat found
    pub async fn parse_metadata(mut self) -> Result<MdatParser<R>, ParseError> {
        let atoms = self.parse_metadata_inner(None).await?;
        Ok(MdatParser::new(
            self.reader,
            Metadata::new(atoms),
            self.mdat,
        ))
    }

    async fn parse_metadata_inner(
        &mut self,
        length_limit: Option<usize>,
    ) -> Result<Vec<Atom>, ParseError> {
        let start_offset = self.reader.current_offset;

        let mut top_level_atoms = Vec::new();

        loop {
            // ensure we're respecting container bounds
            if length_limit.is_some_and(|limit| self.reader.current_offset - start_offset >= limit)
            {
                break;
            }

            let header = match self.parse_next_atom().await {
                Ok(parsed_atom) => Ok(parsed_atom),
                Err(err) => {
                    if matches!(
                        err.kind,
                        ParseErrorKind::Eof | ParseErrorKind::InvalidHeader
                    ) {
                        // end of stream, this means there's no mdat atom
                        // TODO: rewrite the tests to always include an mdat atom so we can get rid of this check
                        break;
                    }
                    Err(err)
                }
            }?;

            // only parse as far as the mdat atom (we're assuming mdat is the last atom)
            if header.atom_type == MDAT {
                self.mdat = Some(header);
                break;
            }

            if is_container_atom(&header.atom_type) {
                // META containers have additional header data
                let (size, data) = if header.atom_type.deref() == META {
                    // Handle META version and flags as RawData
                    let version_flags = self.reader.read_data(META_VERSION_FLAGS_SIZE).await?;
                    (
                        header.data_size - META_VERSION_FLAGS_SIZE,
                        Some(AtomData::RawData(RawData::new(
                            FourCC(*META),
                            version_flags,
                        ))),
                    )
                } else {
                    (header.data_size, None)
                };

                let container_atom = Atom {
                    header,
                    data,
                    children: Box::pin(self.parse_metadata_inner(Some(size))).await?,
                };

                top_level_atoms.push(container_atom);
            } else {
                // Yield leaf atoms
                let atom_data = self.parse_atom_data(&header).await?;
                let atom = Atom {
                    header,
                    data: Some(atom_data),
                    children: Vec::new(),
                };
                top_level_atoms.push(atom);
            }
        }

        Ok(top_level_atoms)
    }

    async fn parse_next_atom(&mut self) -> Result<AtomHeader, ParseError> {
        let atom_offset = self.reader.current_offset as u64;

        // Try to read the atom header (size and type)
        let mut header = [0u8; 8];
        self.reader.read_exact(&mut header).await?;

        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
        let atom_type: [u8; 4] = header[4..8].try_into().unwrap();

        // Handle extended size (64-bit) if needed
        let (header_size, data_size) = if size == 1 {
            // Extended size format
            let mut extended_size = [0u8; 8];
            self.reader.read_exact(&mut extended_size).await?;
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

        Ok(AtomHeader {
            atom_type,
            offset: atom_offset as usize,
            header_size: header_size as usize,
            data_size: data_size as usize,
        })
    }

    async fn parse_atom_data(&mut self, header: &AtomHeader) -> Result<AtomData, ParseError> {
        let content_data = self.reader.read_data(header.data_size).await?;
        let cursor = Cursor::new(content_data);
        let atom_type = header.atom_type;
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
            fourcc => Ok(RawData::new(FourCC(*fourcc), cursor.get_ref().clone()).into()),
        }
        .map_err(|err| ParseError {
            kind: ParseErrorKind::AtomParsing,
            location: Some(err.location.map_or_else(
                || header.location(),
                |(offset, size)| {
                    let (header_offset, header_size) = header.location();
                    (header_offset + offset, size.min(header_size))
                },
            )),
            source: Some(anyhow::Error::from(err).context(atom_type).into()),
        })?;

        Ok(atom_data)
    }
}

/// Determines whether a given atom type (fourcc) should be treated as a container for other atoms.
fn is_container_atom(atom_type: &[u8; 4]) -> bool {
    // Common container types in MP4
    matches!(
        atom_type,
        MOOV | MFRA
            | UDTA
            | TRAK
            | EDTS
            | MDIA
            | MINF
            | DINF
            | STBL
            | MOOF
            | TRAF
            | SINF
            | SCHI
            | META
    )
}

pub struct MdatParser<R> {
    meta: Metadata,
    reader: Option<Mp4Reader<R>>,
    mdat: Option<AtomHeader>,
}

impl<R> Clone for MdatParser<R> {
    fn clone(&self) -> Self {
        Self {
            meta: self.meta.clone(),
            reader: None,
            mdat: None,
        }
    }
}

impl<R> Deref for MdatParser<R> {
    type Target = Metadata;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<R> DerefMut for MdatParser<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl<R> MdatParser<R> {
    fn new(reader: Mp4Reader<R>, meta: Metadata, mdat: Option<AtomHeader>) -> Self {
        Self {
            reader: Some(reader),
            meta,
            mdat,
        }
    }

    /// Discards the reader and returns just the metadata
    pub fn into_metadata(self) -> Metadata {
        self.meta
    }

    /// Retains only the metadata atoms that satisfy the predicate
    /// (applies to top level and nested atoms)
    pub fn atoms_flat_retain_mut<P>(mut self, pred: P) -> Self
    where
        P: FnMut(&mut Atom) -> bool,
    {
        self.meta = self.meta.atoms_flat_retain_mut(pred);
        self
    }

    /// Retains only the TRAK atoms specified by the predicate
    pub fn tracks_retain<P>(mut self, pred: P) -> Self
    where
        P: FnMut(TrakAtomRef) -> bool,
    {
        self.meta = self.meta.tracks_retain(pred);
        self
    }

    pub fn mdat_header(&self) -> Option<&AtomHeader> {
        self.mdat.as_ref()
    }

    /// Parse chunks along with related metadata
    pub fn chunks(&mut self) -> Result<ChunkParser<'_, R>, ParseError> {
        let _ = self.mdat.take().ok_or_else(|| ParseError {
            kind: ParseErrorKind::InsufficientData,
            location: None,
            source: Some(
                anyhow!("mdat atom is missing or has already been consumed").into_boxed_dyn_error(),
            ),
        })?;

        let reader = self.reader.take().ok_or_else(|| ParseError {
            kind: ParseErrorKind::Io,
            location: None,
            source: Some(anyhow!("reader has already been consumed").into_boxed_dyn_error()),
        })?;

        let mut parser = ChunkParser {
            reader,
            tracks: Vec::new(),
            chunk_offsets: Vec::new(),
            sample_to_chunk: Vec::new(),
            sample_sizes: Vec::new(),
            time_to_sample: Vec::new(),
            chunk_info: Vec::new(),
        };

        for trak in self.meta.tracks_iter() {
            if let Some((trak, stco, stsc, stsz, stts)) = (|| {
                let mdia = trak.media()?;
                let minf = mdia.media_information()?;
                let stbl = minf.sample_table()?;
                let chunk_offset = stbl.chunk_offset()?;
                let sample_entries = stbl.sample_to_chunk()?;
                let sample_sizes = stbl.sample_size()?;
                let time_to_sample = stbl.time_to_sample()?;
                Some((
                    trak,
                    chunk_offset.chunk_offsets.inner(),
                    sample_entries,
                    sample_sizes,
                    time_to_sample,
                ))
            })() {
                let mut builder = ChunkOffsetBuilder::with_capacity(1);
                builder.add_track(stsc, stsz);
                parser.tracks.push(trak);
                parser.chunk_offsets.push(stco);
                parser.sample_to_chunk.push(stsc.entries.inner());
                parser.sample_sizes.push(stsz.entry_sizes.inner());
                parser.time_to_sample.push(stts.entries.inner());
                parser
                    .chunk_info
                    .push(builder.build_chunk_info().collect::<VecDeque<_>>());
            }
        }

        Ok(parser)
    }
}

#[derive(Clone)]
pub struct Metadata {
    atoms: Vec<Atom>,
}

impl Metadata {
    fn new(atoms: Vec<Atom>) -> Self {
        Self { atoms }
    }

    /// Transforms into (reader, current_offset, atoms)
    pub fn into_atoms(self) -> Vec<Atom> {
        self.atoms
    }

    /// Iterates over the metadata atoms
    pub fn atoms_iter(&self) -> impl Iterator<Item = &Atom> {
        self.atoms.iter()
    }

    /// Mutably iterates over the metadata atoms
    pub fn atoms_iter_mut(&mut self) -> impl Iterator<Item = &mut Atom> {
        self.atoms.iter_mut()
    }

    /// Retains only the metadata atoms that satisfy the predicate
    /// (applies to top level and nested atoms)
    pub fn atoms_flat_retain_mut<P>(mut self, mut pred: P) -> Self
    where
        P: FnMut(&mut Atom) -> bool,
    {
        self.atoms.retain_mut(|a| pred(a));
        for atom in self.atoms.iter_mut() {
            atom.children_flat_retain_mut(|a| pred(a));
        }
        self
    }

    pub fn ftyp_mut<F>(&mut self, mut f: F) -> Option<()>
    where
        F: FnMut(&mut FileTypeAtom),
    {
        let ftyp = self
            .atoms
            .iter_mut()
            .find(|a| a.header.atom_type == FTYP)?
            .data
            .as_mut()?;
        if let AtomData::FileType(ftyp) = ftyp {
            f(ftyp);
        } else {
            return None;
        }
        Some(())
    }

    pub fn moov(&self) -> Option<MoovAtomRef<'_>> {
        self.atoms
            .iter()
            .find(|a| a.header.atom_type == MOOV)
            .map(MoovAtomRef)
    }

    pub fn moov_mut(&mut self) -> Option<MoovAtomRefMut<'_>> {
        self.atoms
            .iter_mut()
            .find(|a| a.header.atom_type == MOOV)
            .map(MoovAtomRefMut)
    }

    /// Iterate through TRAK atoms
    pub fn tracks_iter(&self) -> impl Iterator<Item = TrakAtomRef<'_>> {
        self.atoms
            .iter()
            .filter(|a| a.header.atom_type == MOOV)
            .flat_map(|a| a.children.iter().filter(|a| a.header.atom_type == TRAK))
            .map(TrakAtomRef)
    }

    /// Iterate through TRAK atoms with handler type Audio
    pub fn audio_track_iter(&self) -> impl Iterator<Item = TrakAtomRef<'_>> {
        self.tracks_iter().filter(|trak| {
            match trak
                .media()
                .and_then(|mdia| mdia.handler_reference())
                .and_then(|hdlr| Some(&hdlr.handler_type))
            {
                Some(HandlerType::Audio) => true,
                _ => false,
            }
        })
    }

    pub fn tracks_iter_mut(&mut self) -> impl Iterator<Item = TrakAtomRefMut<'_>> {
        self.atoms
            .iter_mut()
            .filter(|a| a.header.atom_type == MOOV)
            .flat_map(|a| a.children.iter_mut().filter(|a| a.header.atom_type == TRAK))
            .map(TrakAtomRefMut)
    }

    /// Iterate through TRAK atoms with handler type Audio
    pub fn audio_track_iter_mut(&mut self) -> impl Iterator<Item = TrakAtomRefMut<'_>> {
        self.tracks_iter_mut().filter(|trak| {
            match trak
                .as_ref()
                .media()
                .and_then(|mdia| mdia.handler_reference())
                .and_then(|hdlr| Some(&hdlr.handler_type))
            {
                Some(HandlerType::Audio) => true,
                _ => false,
            }
        })
    }

    /// Retains only the TRAK atoms specified by the predicate
    pub fn tracks_retain<P>(mut self, mut pred: P) -> Self
    where
        P: FnMut(TrakAtomRef) -> bool,
    {
        self.atoms
            .iter_mut()
            .filter(|a| a.header.atom_type == MOOV)
            .for_each(|a| {
                a.children
                    .retain(|a| a.header.atom_type != TRAK || pred(TrakAtomRef(a)));
            });
        self
    }

    /// Adds trak atom to moov
    ///
    /// Returns an error if moov isn't found
    pub fn add_track(&mut self, trak: Atom) -> Result<(), ParseError> {
        let moov = self
            .atoms
            .iter_mut()
            .find(|a| a.header.atom_type == MOOV)
            .ok_or_else(|| ParseError {
                kind: ParseErrorKind::MissingMoov,
                location: None,
                source: None,
            })?;

        moov.children.push(trak);

        Ok(())
    }

    /// Returns the sum of all metadata atom sizes in bytes
    pub fn metadata_size(&self) -> usize {
        self.atoms_iter()
            .cloned()
            .flat_map(SerializeAtom::into_bytes)
            .collect::<Vec<_>>()
            .len()
    }

    /// Returns the sum of all track sizes in bytes
    pub fn mdat_size(&self) -> usize {
        self.tracks_iter().map(|trak| trak.size()).sum::<usize>()
    }

    /// Returns the sum of metadata_size and mdat_size
    pub fn file_size(&self) -> usize {
        self.metadata_size() + self.mdat_size()
    }

    /// Adds or replaces the chapter list atom in the metadata
    pub fn add_or_replace_chpl(&mut self, chpl: ChapterListAtom) {
        let udta = self
            .atoms_iter_mut()
            .find(|atom| atom.header.atom_type == UDTA);
        let udta = match udta {
            Some(udta) => udta,
            None => {
                let udta = Atom {
                    header: AtomHeader {
                        atom_type: FourCC::from(*UDTA),
                        offset: 0,
                        header_size: 0,
                        data_size: 0,
                    },
                    data: None,
                    children: Vec::new(),
                };
                self.atoms.push(udta);
                self.atoms.last_mut().unwrap()
            }
        };

        udta.children
            .retain_mut(|atom| atom.header.atom_type != CHPL);
        udta.children.push(Atom {
            header: AtomHeader {
                atom_type: FourCC::from(*CHPL),
                offset: 0,
                header_size: 0,
                data_size: 0,
            },
            data: Some(AtomData::ChapterList(chpl)),
            children: Vec::new(),
        });
    }

    /// Updates chunk offsets for each track
    ///
    /// Call this before writing metadata to disk to avoid corruption
    pub fn update_chunk_offsets(&mut self) -> Result<(), UpdateChunkOffsetError> {
        // mdat is located directly after metadata atoms, so metadata size + 8 bytes for the mdat header
        let mdat_content_offset = self.metadata_size() + 8;

        let (chunk_offsets, original_chunk_offsets) =
            self.tracks_iter()
                .fold(Ok((ChunkOffsetBuilder::new(), Vec::new())), |acc, trak| {
                    let (mut builder, mut chunk_offsets) = acc?;
                    let stbl = trak
                        .media()
                        .and_then(|mdia| mdia.media_information())
                        .and_then(|minf| minf.sample_table())
                        .ok_or_else(|| UpdateChunkOffsetError::SampleTableNotFound)?;
                    let stsz = stbl
                        .sample_size()
                        .ok_or_else(|| UpdateChunkOffsetError::SampleSizeAtomNotFound)?;
                    let stsc = stbl
                        .sample_to_chunk()
                        .ok_or_else(|| UpdateChunkOffsetError::SampleToChunkAtomNotFound)?;
                    let stco = stbl
                        .chunk_offset()
                        .ok_or_else(|| UpdateChunkOffsetError::ChunkOffsetAtomNotFound)?;
                    builder.add_track(stsc, stsz);
                    chunk_offsets.push(stco.chunk_offsets.inner());
                    Ok((builder, chunk_offsets))
                })?;

        let mut chunk_offsets = chunk_offsets
            .build_chunk_offsets_ordered(original_chunk_offsets, mdat_content_offset as u64);

        for (track_idx, trak) in self.tracks_iter_mut().enumerate() {
            let mut stbl = trak
                .media()
                .and_then(|mdia| mdia.media_information())
                .and_then(|minf| minf.sample_table())
                .ok_or_else(|| UpdateChunkOffsetError::SampleTableNotFound)?;
            let stco = stbl
                .chunk_offset()
                .ok_or_else(|| UpdateChunkOffsetError::ChunkOffsetAtomNotFound)?;
            let chunk_offsets = std::mem::take(&mut chunk_offsets[track_idx]);
            stco.chunk_offsets = ChunkOffsets::from(chunk_offsets);
        }

        Ok(())
    }

    /// Updates bitrate for each track
    pub fn update_bitrate(&mut self) {
        self.tracks_iter_mut().for_each(|mut trak| {
            if let Some(bitrate) = trak.as_ref().bitrate() {
                trak.update_bitrate(bitrate);
            }
        });
    }
}

#[derive(Debug, Error)]
pub enum UpdateChunkOffsetError {
    #[error("sample table atom not found")]
    SampleTableNotFound,
    #[error("sample size atom not found")]
    SampleSizeAtomNotFound,
    #[error("sample to chunk atom not found")]
    SampleToChunkAtomNotFound,
    #[error("chunk offset atom not found")]
    ChunkOffsetAtomNotFound,
}

pub struct MoovAtomRef<'a>(&'a Atom);

impl<'a> MoovAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children.iter()
    }

    pub fn header(&self) -> Option<&'a MovieHeaderAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == MVHD)?;
        match atom.data.as_ref()? {
            AtomData::MovieHeader(data) => Some(data),
            _ => None,
        }
    }
}

pub struct MoovAtomRefMut<'a>(&'a mut Atom);

impl<'a> MoovAtomRefMut<'a> {
    pub fn as_ref(&self) -> MoovAtomRef<'_> {
        MoovAtomRef(self.0)
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children.iter_mut()
    }

    pub fn header(&mut self) -> Option<&'_ mut MovieHeaderAtom> {
        self.children()
            .find(|a| a.header.atom_type == MDHD)
            .and_then(|a| a.data.as_mut())
            .and_then(|data| match data {
                AtomData::MovieHeader(data) => Some(data),
                _ => None,
            })
    }
}

pub struct TrakAtomRef<'a>(&'a Atom);

impl<'a> fmt::Debug for TrakAtomRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrakAtomRef")
            .field("track_id", &self.header().unwrap().track_id)
            .finish()
    }
}

impl<'a> TrakAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children.iter()
    }

    /// Finds the TKHD atom
    pub fn header(&self) -> Option<&'a TrackHeaderAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == TKHD)?;
        match atom.data.as_ref()? {
            AtomData::TrackHeader(data) => Some(data),
            _ => None,
        }
    }

    pub fn media(&self) -> Option<MdiaAtomRef<'a>> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == MDIA)?;
        Some(MdiaAtomRef(atom))
    }

    pub fn track_id(&self) -> Option<u32> {
        let tkhd = self.header()?;
        Some(tkhd.track_id)
    }

    /// Returns the sum of all sample sizes
    pub fn size(&self) -> usize {
        self.media()
            .and_then(|m| m.media_information())
            .and_then(|m| m.sample_table())
            .and_then(|st| st.sample_size())
            .map(|s| {
                if s.entry_sizes.is_empty() {
                    s.sample_size * s.sample_count
                } else {
                    s.entry_sizes.iter().sum::<u32>()
                }
            })
            .unwrap_or(0) as usize
    }

    /// Calculates the track's bitrate
    ///
    /// Returns None if either stsz or mdhd atoms can't be found
    pub fn bitrate(&self) -> Option<u32> {
        let duration_secds = self
            .media()
            .and_then(|m| m.header())
            .map(|mdhd| (mdhd.duration as f64) / (mdhd.timescale as f64))?;

        self.media()
            .and_then(|mdia| mdia.media_information())
            .and_then(|minf| minf.sample_table())
            .and_then(|stbl| stbl.sample_size())
            .map(|s| {
                let num_bits = s
                    .entry_sizes
                    .iter()
                    .map(|s| s.clone() as usize)
                    .sum::<usize>()
                    .saturating_mul(8);

                let bitrate = (num_bits as f64) / duration_secds;
                bitrate.round() as u32
            })
    }
}

pub struct TrakAtomRefMut<'a>(&'a mut Atom);

impl<'a> TrakAtomRefMut<'a> {
    pub fn as_ref(&self) -> TrakAtomRef<'_> {
        TrakAtomRef(&self.0)
    }

    pub fn into_ref(self) -> TrakAtomRef<'a> {
        TrakAtomRef(self.0)
    }

    pub fn header(&mut self) -> Option<&mut TrackHeaderAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == TKHD)?;
        match atom.data.as_mut()? {
            AtomData::TrackHeader(data) => Some(data),
            _ => None,
        }
    }

    pub fn media(self) -> Option<MdiaAtomRefMut<'a>> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == MDIA)?;
        Some(MdiaAtomRefMut(atom))
    }

    pub fn add_track_reference(&mut self, references: impl Into<Vec<TrackReference>>) {
        self.0.children.insert(
            // try and insert after track header but before mdia
            1.min(self.0.children.len()),
            Atom {
                header: AtomHeader::new(FourCC(*TREF)),
                data: Some(AtomData::TrackReference(TrackReferenceAtom {
                    references: references.into(),
                })),
                children: Vec::new(),
            },
        );
    }

    pub fn add_edit_list(&mut self, entries: impl Into<Vec<EditEntry>>) {
        self.0.children.insert(
            // try and insert after track header but before mdia
            1.min(self.0.children.len()),
            Atom {
                header: AtomHeader::new(FourCC(*EDTS)),
                data: None,
                children: vec![Atom {
                    header: AtomHeader::new(FourCC(*ELST)),
                    data: Some(AtomData::EditList(EditListAtom {
                        version: 0,
                        flags: [0u8; 3],
                        entries: entries.into(),
                    })),
                    children: Vec::new(),
                }],
            },
        );
    }

    /// Updates track metadata with the new bitrate
    pub fn update_bitrate(&mut self, bitrate: u32) {
        let mdia = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == MDIA)
            .map(|atom| MdiaAtomRefMut(atom));
        let stbl = mdia
            .and_then(|mdia| mdia.media_information())
            .and_then(|minf| minf.sample_table());
        if let Some(mut stbl) = stbl {
            if let Some(stsd) = stbl.sample_description() {
                stsd.entries.retain_mut(|entry| {
                    if !matches!(entry.data, SampleEntryData::Audio(_)) {
                        return true;
                    }

                    entry.entry_type = SampleEntryType::Mp4a;

                    if let SampleEntryData::Audio(audio) = &mut entry.data {
                        let mut sample_frequency = None;
                        audio.extensions.retain_mut(|ext| match ext {
                            StsdExtension::Esds(esds) => {
                                if let Some(c) =
                                    esds.es_descriptor.decoder_config_descriptor.as_mut()
                                {
                                    c.avg_bitrate = bitrate;
                                    c.max_bitrate = bitrate;
                                    if let Some(DecoderSpecificInfo::Audio(a, _)) =
                                        c.decoder_specific_info.as_ref()
                                    {
                                        sample_frequency = Some(a.sampling_frequency.as_hz());
                                    }
                                };
                                true
                            }
                            StsdExtension::Btrt(_) => false,
                            StsdExtension::Unknown { .. } => false,
                        });
                        audio.extensions.push(StsdExtension::Btrt(BtrtExtension {
                            buffer_size_db: 0,
                            avg_bitrate: bitrate,
                            max_bitrate: bitrate,
                        }));

                        if let Some(hz) = sample_frequency {
                            audio.sample_rate = hz as f32;
                        }
                    }

                    true
                });
            }
        }
    }
}

pub struct MdiaAtomRef<'a>(&'a Atom);

impl<'a> MdiaAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children.iter()
    }

    /// Finds the MDHD atom
    pub fn header(&self) -> Option<&'a MediaHeaderAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == MDHD)?;
        match atom.data.as_ref()? {
            AtomData::MediaHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the HDLR atom
    pub fn handler_reference(&self) -> Option<&'a HandlerReferenceAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == HDLR)?;
        match atom.data.as_ref()? {
            AtomData::HandlerReference(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the MINF atom
    pub fn media_information(&self) -> Option<MinfAtomRef<'a>> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == MINF)?;
        Some(MinfAtomRef(atom))
    }
}

pub struct MdiaAtomRefMut<'a>(&'a mut Atom);

impl<'a> MdiaAtomRefMut<'a> {
    pub fn as_ref(&self) -> MdiaAtomRef<'_> {
        MdiaAtomRef(&self.0)
    }

    pub fn into_ref(self) -> MdiaAtomRef<'a> {
        MdiaAtomRef(self.0)
    }

    pub fn children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.children.iter_mut()
    }

    /// Finds the MDHD atom
    pub fn header(&mut self) -> Option<&mut MediaHeaderAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == MDHD)?;
        match atom.data.as_mut()? {
            AtomData::MediaHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the HDLR atom
    pub fn handler_reference(&mut self) -> Option<&mut HandlerReferenceAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == HDLR)?;
        match atom.data.as_mut()? {
            AtomData::HandlerReference(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the MINF atom
    pub fn media_information(self) -> Option<MinfAtomRefMut<'a>> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == MINF)?;
        Some(MinfAtomRefMut(atom))
    }
}

pub struct MinfAtomRef<'a>(&'a Atom);

impl<'a> MinfAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children.iter()
    }

    /// Finds the STBL atom
    pub fn sample_table(&self) -> Option<StblAtomRef<'a>> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == STBL)?;
        Some(StblAtomRef(atom))
    }
}

pub struct MinfAtomRefMut<'a>(&'a mut Atom);

impl<'a> MinfAtomRefMut<'a> {
    pub fn as_ref(&self) -> MinfAtomRef<'_> {
        MinfAtomRef(&self.0)
    }

    pub fn into_ref(self) -> MinfAtomRef<'a> {
        MinfAtomRef(self.0)
    }

    pub fn children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.children.iter_mut()
    }

    /// Finds the STBL atom
    pub fn sample_table(self) -> Option<StblAtomRefMut<'a>> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == STBL)?;
        Some(StblAtomRefMut(atom))
    }
}

pub struct StblAtomRef<'a>(&'a Atom);

impl<'a> StblAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children.iter()
    }

    /// Finds the STSD atom
    pub fn sample_description(&self) -> Option<&'a SampleDescriptionTableAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == STSD)?;
        match atom.data.as_ref()? {
            AtomData::SampleDescriptionTable(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STTS atom
    pub fn time_to_sample(&self) -> Option<&'a TimeToSampleAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == STTS)?;
        match atom.data.as_ref()? {
            AtomData::TimeToSample(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSC atom
    pub fn sample_to_chunk(&self) -> Option<&'a SampleToChunkAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == STSC)?;
        match atom.data.as_ref()? {
            AtomData::SampleToChunk(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSZ atom
    pub fn sample_size(&self) -> Option<&'a SampleSizeAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == STSZ)?;
        match atom.data.as_ref()? {
            AtomData::SampleSize(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STCO atom
    pub fn chunk_offset(&self) -> Option<&'a ChunkOffsetAtom> {
        let atom = self
            .0
            .children
            .iter()
            .find(|a| a.header.atom_type == STCO)?;
        match atom.data.as_ref()? {
            AtomData::ChunkOffset(data) => Some(data),
            _ => None,
        }
    }
}

pub struct StblAtomRefMut<'a>(&'a mut Atom);

impl<'a> StblAtomRefMut<'a> {
    pub fn as_ref(&self) -> StblAtomRef<'_> {
        StblAtomRef(&self.0)
    }

    pub fn into_ref(self) -> StblAtomRef<'a> {
        StblAtomRef(self.0)
    }

    pub fn children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.children.iter_mut()
    }

    /// Finds the STSD atom
    pub fn sample_description(&mut self) -> Option<&mut SampleDescriptionTableAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == STSD)?;
        match atom.data.as_mut()? {
            AtomData::SampleDescriptionTable(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STTS atom
    pub fn time_to_sample(&mut self) -> Option<&mut TimeToSampleAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == STTS)?;
        match atom.data.as_mut()? {
            AtomData::TimeToSample(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSC atom
    pub fn sample_to_chunk(&mut self) -> Option<&mut SampleToChunkAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == STSC)?;
        match atom.data.as_mut()? {
            AtomData::SampleToChunk(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSZ atom
    pub fn sample_size(&mut self) -> Option<&mut SampleSizeAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == STSZ)?;
        match atom.data.as_mut()? {
            AtomData::SampleSize(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STCO atom
    pub fn chunk_offset(&mut self) -> Option<&mut ChunkOffsetAtom> {
        let atom = self
            .0
            .children
            .iter_mut()
            .find(|a| a.header.atom_type == STCO)?;
        match atom.data.as_mut()? {
            AtomData::ChunkOffset(data) => Some(data),
            _ => None,
        }
    }
}

pub struct ChunkParser<'a, R> {
    reader: Mp4Reader<R>,
    /// Reference to each track's metadata
    tracks: Vec<TrakAtomRef<'a>>,
    /// Chunk offsets for each track
    chunk_offsets: Vec<&'a [u64]>,
    /// [SampleToChunkEntry]s for each track
    sample_to_chunk: Vec<&'a [SampleToChunkEntry]>,
    /// Sample sizes for each track
    sample_sizes: Vec<&'a [u32]>,
    /// [TimeToSampleEntry]s for each track
    time_to_sample: Vec<&'a [TimeToSampleEntry]>,
    /// [ChunkInfo]s for each track
    chunk_info: Vec<VecDeque<ChunkInfo>>,
}

impl<'a, R: AsyncRead + Unpin + Send> ChunkParser<'a, R> {
    pub async fn read_next_chunk(&mut self) -> Result<Option<Chunk<'a>>, ParseError> {
        let current_offset = self.reader.current_offset as u64;

        let mut next_offset = None;
        let mut next_track_idx = 0;
        let mut next_chunk_idx = 0;

        for track_idx in 0..self.tracks.len() {
            let chunk_info = self.chunk_info[track_idx].front();
            if let Some(chunk_info) = chunk_info {
                let chunk_idx = chunk_info.chunk_number as usize - 1;
                let offset = self.chunk_offsets[track_idx][chunk_idx];
                if offset >= current_offset
                    && next_offset.is_none_or(|next_offset| offset < next_offset)
                {
                    next_offset = Some(offset);
                    next_track_idx = track_idx;
                    next_chunk_idx = chunk_idx;
                }
            }
        }

        if let Some(offset) = next_offset {
            // Skip to the next chunk
            let bytes_to_skip = offset - current_offset;
            if bytes_to_skip > 0 {
                self.reader.read_data(bytes_to_skip as usize).await?;
            }

            let chunk_info = self.chunk_info[next_track_idx].pop_front().unwrap();

            // Read the chunk
            self.read_chunk(next_track_idx, next_chunk_idx, chunk_info)
                .await
                .map(Some)
        } else {
            // No more chunks
            Ok(None)
        }
    }

    async fn read_chunk(
        &mut self,
        track_idx: usize,
        chunk_idx: usize,
        chunk_info: ChunkInfo,
    ) -> Result<Chunk<'a>, ParseError> {
        let time_to_sample = self.time_to_sample[track_idx];

        let sample_start_idx =
            chunk_info
                .sample_indices
                .first()
                .copied()
                .ok_or_else(|| ParseError {
                    kind: ParseErrorKind::InsufficientData,
                    location: None,
                    source: Some(
                        anyhow!("no samples indicies in chunk at index {chunk_idx}")
                            .into_boxed_dyn_error(),
                    ),
                })?;

        // Calculate total chunk size
        let chunk_size = chunk_info.chunk_size;
        let chunk_sample_sizes = chunk_info.sample_sizes.clone();

        // Read the chunk data
        let data = self.reader.read_data(chunk_size as usize).await?;

        // Get the sample durations slice for this chunk
        let sample_durations: Vec<u32> = time_to_sample
            .iter()
            .flat_map(|entry| {
                std::iter::repeat_n(entry.sample_duration, entry.sample_count as usize)
            })
            .skip(sample_start_idx)
            .take(chunk_sample_sizes.len())
            .collect();
        assert_eq!(chunk_sample_sizes.len(), sample_durations.len());

        // Create the chunk
        Ok(Chunk {
            trak_idx: track_idx,
            trak: TrakAtomRef(self.tracks[track_idx].0),
            sample_sizes: chunk_sample_sizes,
            sample_durations,
            data,
        })
    }
}

impl<'a> fmt::Debug for Chunk<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chunk")
            .field("trak", &self.trak)
            .field(
                "sample_sizes",
                &DebugEllipsis(Some(self.sample_sizes.len())),
            )
            .field(
                "time_to_sample",
                &DebugEllipsis(Some(self.sample_durations.len())),
            )
            .field("data", &DebugEllipsis(Some(self.data.len())))
            .finish()
    }
}

pub struct Chunk<'a> {
    /// Index of the trak in the file
    pub trak_idx: usize,
    /// Reference to the track the sample is in
    pub trak: TrakAtomRef<'a>,
    /// Slice of sample sizes within this chunk
    pub sample_sizes: Vec<u32>,
    /// Timescale duration of each sample indexed reletive to `sample_sizes`
    pub sample_durations: Vec<u32>,
    /// Bytes in the chunk
    pub data: Vec<u8>,
}

impl<'a> Chunk<'a> {
    pub fn samples(&'a self) -> impl Iterator<Item = Sample<'a>> {
        let timescale = self
            .trak
            .media()
            .and_then(|h| h.header())
            .map(|h| h.timescale)
            .expect("trak.mdia.mvhd is missing");
        self.sample_sizes
            .iter()
            .zip(self.sample_durations.iter())
            .scan(0usize, move |offset, (size, duration)| {
                let sample_offset = *offset;
                *offset += *size as usize;
                let data = &self.data[sample_offset..sample_offset + (*size as usize)];
                Some(Sample {
                    size: *size,
                    duration: *duration,
                    timescale,
                    data,
                })
            })
    }
}

pub struct Sample<'a> {
    pub size: u32,
    pub duration: u32,
    pub timescale: u32,
    pub data: &'a [u8],
}

impl<'a> fmt::Debug for Sample<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sample")
            .field("size", &self.size)
            .field("duration", &self.duration)
            .field("timescale", &self.timescale)
            .finish_non_exhaustive()
    }
}
