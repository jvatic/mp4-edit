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
    AudioSampleEntry, BtrtExtension, DecoderSpecificInfo, EsdsExtension, SampleEntry,
    SampleEntryData, SampleEntryType, StsdExtension,
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

        for trak in self.meta.moov().into_tracks_iter() {
            if let Some((trak, stco, stsc, stsz, stts)) = (|| {
                let stbl = trak.media().media_information().sample_table();
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
    pub fn atoms_flat_retain_mut<P>(&mut self, mut pred: P)
    where
        P: FnMut(&mut Atom) -> bool,
    {
        self.atoms.retain_mut(|a| pred(a));
        for atom in self.atoms.iter_mut() {
            atom.children_flat_retain_mut(|a| pred(a));
        }
    }

    fn atom_position(&self, typ: &[u8; 4]) -> Option<usize> {
        self.atoms.iter().position(|a| a.header.atom_type == typ)
    }

    fn find_atom(&self, typ: &[u8; 4]) -> AtomRef<'_> {
        AtomRef(self.atoms.iter().find(|a| a.header.atom_type == typ))
    }

    pub fn ftyp(&mut self) -> FtypAtomRef<'_> {
        FtypAtomRef(self.find_atom(FTYP))
    }

    pub fn ftyp_mut(&mut self) -> FtypAtomRefMut<'_> {
        if let Some(index) = self.atom_position(FTYP) {
            FtypAtomRefMut(AtomRefMut(&mut self.atoms[index]))
        } else {
            let index = 0;
            self.atoms.insert(
                index,
                Atom::builder()
                    .header(AtomHeader::new(*FTYP))
                    .data(FileTypeAtom::default())
                    .build(),
            );
            FtypAtomRefMut(AtomRefMut(&mut self.atoms[index]))
        }
    }

    pub fn moov(&self) -> MoovAtomRef<'_> {
        MoovAtomRef(self.find_atom(MOOV))
    }

    pub fn moov_mut(&mut self) -> MoovAtomRefMut<'_> {
        if let Some(index) = self.atom_position(MOOV) {
            MoovAtomRefMut(AtomRefMut(&mut self.atoms[index]))
        } else {
            let index = self.atom_position(FTYP).map(|i| i + 1).unwrap_or_default();
            self.atoms.insert(
                index,
                Atom::builder().header(AtomHeader::new(*MOOV)).build(),
            );
            MoovAtomRefMut(AtomRefMut(&mut self.atoms[index]))
        }
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
        self.moov()
            .into_tracks_iter()
            .map(|trak| trak.size())
            .sum::<usize>()
    }

    /// Returns the sum of metadata_size and mdat_size
    pub fn file_size(&self) -> usize {
        self.metadata_size() + self.mdat_size()
    }

    /// Updates chunk offsets for each track
    ///
    /// Call this before writing metadata to disk to avoid corruption
    pub fn update_chunk_offsets(&mut self) -> Result<(), UpdateChunkOffsetError> {
        // mdat is located directly after metadata atoms, so metadata size + 8 bytes for the mdat header
        let mdat_content_offset = self.metadata_size() + 8;

        let (chunk_offsets, original_chunk_offsets) = self.moov().into_tracks_iter().try_fold(
            (ChunkOffsetBuilder::new(), Vec::new()),
            |(mut builder, mut chunk_offsets), trak| {
                let stbl = trak.media().media_information().sample_table();
                let stsz = stbl
                    .sample_size()
                    .ok_or(UpdateChunkOffsetError::SampleSizeAtomNotFound)?;
                let stsc = stbl
                    .sample_to_chunk()
                    .ok_or(UpdateChunkOffsetError::SampleToChunkAtomNotFound)?;
                let stco = stbl
                    .chunk_offset()
                    .ok_or(UpdateChunkOffsetError::ChunkOffsetAtomNotFound)?;
                builder.add_track(stsc, stsz);
                chunk_offsets.push(stco.chunk_offsets.inner());
                Ok((builder, chunk_offsets))
            },
        )?;

        let mut chunk_offsets = chunk_offsets
            .build_chunk_offsets_ordered(original_chunk_offsets, mdat_content_offset as u64);

        for (track_idx, trak) in self.moov_mut().tracks().enumerate() {
            let mut stbl = trak
                .into_media()
                .and_then(|mdia| mdia.into_media_information())
                .and_then(|minf| minf.into_sample_table())
                .ok_or(UpdateChunkOffsetError::SampleTableNotFound)?;
            let stco = stbl
                .chunk_offset()
                .ok_or(UpdateChunkOffsetError::ChunkOffsetAtomNotFound)?;
            let chunk_offsets = std::mem::take(&mut chunk_offsets[track_idx]);
            stco.chunk_offsets = ChunkOffsets::from(chunk_offsets);
        }

        Ok(())
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

pub struct AtomIter<'a> {
    iter: Option<std::slice::Iter<'a, Atom>>,
}

impl<'a> AtomIter<'a> {
    pub fn from_atom(atom_opt: Option<&'a Atom>) -> Self {
        Self {
            iter: atom_opt.map(|atom| atom.children.iter()),
        }
    }
}

impl<'a> Iterator for AtomIter<'a> {
    type Item = &'a Atom;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.as_mut().and_then(|iter| iter.next())
    }
}

impl<'a> DoubleEndedIterator for AtomIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.as_mut().and_then(|iter| iter.next_back())
    }
}

impl<'a> ExactSizeIterator for AtomIter<'a> {
    fn len(&self) -> usize {
        self.iter
            .as_ref()
            .map(|iter| iter.len())
            .unwrap_or_default()
    }
}

pub struct AtomIterMut<'a> {
    children: &'a mut [Atom],
    index: usize,
}

impl<'a> AtomIterMut<'a> {
    pub fn from_atom(atom: &'a mut Atom) -> Self {
        Self {
            children: &mut atom.children,
            index: 0,
        }
    }
}

impl<'a> Iterator for AtomIterMut<'a> {
    type Item = &'a mut Atom;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.children.len() {
            return None;
        }

        let children = std::mem::take(&mut self.children);
        let (current, rest) = children.split_at_mut(self.index + 1);
        self.children = rest;
        let old_index = self.index;
        self.index = 0;

        current.get_mut(old_index)
    }
}

#[derive(Debug, Clone, Copy)]
struct AtomRef<'a>(Option<&'a Atom>);

impl<'a> AtomRef<'a> {
    fn inner(&self) -> Option<&'a Atom> {
        self.0
    }

    fn find_child(&self, typ: &[u8; 4]) -> Option<&'a Atom> {
        self.children().find(|atom| atom.header.atom_type == typ)
    }

    fn children(&self) -> AtomIter<'a> {
        AtomIter::from_atom(self.0)
    }

    fn child_position(&self, typ: &[u8; 4]) -> Option<usize> {
        self.children()
            .position(|atom| atom.header.atom_type == typ)
    }

    fn child_rposition(&self, typ: &[u8; 4]) -> Option<usize> {
        self.children()
            .rposition(|atom| atom.header.atom_type == typ)
    }
}

#[derive(Debug)]
struct AtomRefMut<'a>(&'a mut Atom);

impl<'a> AtomRefMut<'a> {
    fn as_ref(&self) -> AtomRef<'_> {
        AtomRef(Some(self.0))
    }

    fn into_ref(self) -> AtomRef<'a> {
        AtomRef(Some(self.0))
    }

    fn atom_mut(&mut self) -> &'_ mut Atom {
        self.0
    }

    fn get_child(&mut self, index: usize) -> AtomRefMut<'_> {
        AtomRefMut(&mut self.0.children[index])
    }

    fn find_child(&mut self, typ: &[u8; 4]) -> Option<&'_ mut Atom> {
        self.children().find(|atom| atom.header.atom_type == typ)
    }

    fn into_child(self, typ: &[u8; 4]) -> Option<&'a mut Atom> {
        self.into_children()
            .find(|atom| atom.header.atom_type == typ)
    }

    fn remove_child(&mut self, typ: &[u8; 4]) {
        self.0.children.retain(|a| a.header.atom_type != typ);
    }

    fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children.iter_mut()
    }

    fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        AtomIterMut::from_atom(self.0)
    }

    fn insert_child(&mut self, index: usize, child: Atom) {
        self.0.children.insert(index, child);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FtypAtomRef<'a>(AtomRef<'a>);

impl<'a> FtypAtomRef<'a> {
    pub fn data(&self) -> Option<&'a FileTypeAtom> {
        self.0
            .inner()
            .and_then(|ftyp| ftyp.data.as_ref())
            .and_then(|data| match data {
                AtomData::FileType(data) => Some(data),
                _ => None,
            })
    }
}

#[derive(Debug)]
pub struct FtypAtomRefMut<'a>(AtomRefMut<'a>);

impl<'a> FtypAtomRefMut<'a> {
    pub fn as_ref(&self) -> FtypAtomRef<'_> {
        FtypAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> FtypAtomRef<'a> {
        FtypAtomRef(self.0.into_ref())
    }

    pub fn replace(&mut self, data: FileTypeAtom) {
        self.0.atom_mut().data = Some(data.into())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MoovAtomRef<'a>(AtomRef<'a>);

impl<'a> MoovAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    pub fn header(&self) -> Option<&'a MovieHeaderAtom> {
        let atom = self.children().find(|a| a.header.atom_type == MVHD)?;
        match atom.data.as_ref()? {
            AtomData::MovieHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Iterate through TRAK atoms
    pub fn into_tracks_iter(self) -> impl Iterator<Item = TrakAtomRef<'a>> {
        self.children()
            .filter(|a| a.header.atom_type == TRAK)
            .map(TrakAtomRef::new)
    }

    /// Iterate through TRAK atoms with handler type Audio
    pub fn into_audio_track_iter(self) -> impl Iterator<Item = TrakAtomRef<'a>> {
        self.into_tracks_iter().filter(|trak| {
            matches!(
                trak.media()
                    .handler_reference()
                    .map(|hdlr| &hdlr.handler_type),
                Some(HandlerType::Audio)
            )
        })
    }
}

#[derive(Debug)]
pub struct MoovAtomRefMut<'a>(AtomRefMut<'a>);

impl<'a> MoovAtomRefMut<'a> {
    pub fn as_ref(&self) -> MoovAtomRef<'_> {
        MoovAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> MoovAtomRef<'a> {
        MoovAtomRef(self.0.into_ref())
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children()
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

    pub fn user_data(&mut self) -> UserDataAtomRefMut<'_> {
        if let Some(index) = self.0.as_ref().child_position(UDTA) {
            return UserDataAtomRefMut(AtomRefMut(&mut self.0 .0.children[index]));
        }

        let last_trak_index = self.0.as_ref().child_rposition(TRAK);
        let mvhd_index = self.0.as_ref().child_position(MVHD);
        let index = last_trak_index
            .or(mvhd_index)
            .map(|i| i + 1)
            .unwrap_or_default();

        self.0.insert_child(
            index,
            Atom::builder().header(AtomHeader::new(*UDTA)).build(),
        );
        UserDataAtomRefMut(AtomRefMut(&mut self.0 .0.children[index]))
    }

    pub fn tracks(&mut self) -> impl Iterator<Item = TrakAtomRefMut<'_>> {
        self.0
            .children()
            .filter(|a| a.header.atom_type == TRAK)
            .map(TrakAtomRefMut::new)
    }

    /// Iterate through TRAK atoms with handler type Audio
    pub fn audio_tracks(&mut self) -> impl Iterator<Item = TrakAtomRefMut<'_>> {
        self.tracks().filter(|trak| {
            matches!(
                trak.as_ref()
                    .media()
                    .handler_reference()
                    .map(|hdlr| &hdlr.handler_type),
                Some(HandlerType::Audio)
            )
        })
    }

    /// Retains only the TRAK atoms specified by the predicate
    pub fn tracks_retain<P>(self, mut pred: P) -> Self
    where
        P: FnMut(TrakAtomRef) -> bool,
    {
        self.0
             .0
            .children
            .retain(|a| a.header.atom_type != TRAK || pred(TrakAtomRef::new(a)));
        self
    }

    /// Adds trak atom to moov
    ///
    /// Insetion position is either after the last TRAK or MVHD, or at the beginning
    pub fn add_track(&mut self, trak: Atom) {
        let last_trak_index = self.0.as_ref().child_rposition(TRAK);
        let mvhd_index = self.0.as_ref().child_position(MVHD);
        let index = last_trak_index
            .or(mvhd_index)
            .map(|i| i + 1)
            .unwrap_or_default();
        self.0.insert_child(index, trak);
    }
}

pub struct UserDataAtomRefMut<'a>(AtomRefMut<'a>);

impl<'a> UserDataAtomRefMut<'a> {
    /// Adds or replaces the CHPL (chapter list) atom
    pub fn add_or_replace_chpl(&mut self, chpl: ChapterListAtom) {
        self.0.remove_child(CHPL);
        let meta_index = self.0.as_ref().child_position(META);
        let index = meta_index.map(|i| i + 1).unwrap_or_default();
        self.0.insert_child(
            index,
            Atom::builder()
                .header(AtomHeader::new(*CHPL))
                .data(chpl)
                .build(),
        );
    }
}

#[derive(Clone, Copy)]
pub struct TrakAtomRef<'a>(AtomRef<'a>);

impl<'a> fmt::Debug for TrakAtomRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrakAtomRef")
            .field("track_id", &self.header().unwrap().track_id)
            .finish()
    }
}

impl<'a> TrakAtomRef<'a> {
    fn new(atom: &'a Atom) -> Self {
        Self(AtomRef(Some(atom)))
    }

    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    /// Finds the TKHD atom
    pub fn header(&self) -> Option<&'a TrackHeaderAtom> {
        let atom = self.0.find_child(TKHD)?;
        match atom.data.as_ref()? {
            AtomData::TrackHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the MDIA atom
    pub fn media(&self) -> MdiaAtomRef<'a> {
        MdiaAtomRef(AtomRef(self.0.find_child(MDIA)))
    }

    pub fn track_id(&self) -> Option<u32> {
        let tkhd = self.header()?;
        Some(tkhd.track_id)
    }

    /// Returns the sum of all sample sizes
    pub fn size(&self) -> usize {
        self.media()
            .media_information()
            .sample_table()
            .sample_size()
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
            .header()
            .map(|mdhd| (mdhd.duration as f64) / (mdhd.timescale as f64))?;

        self.media()
            .media_information()
            .sample_table()
            .sample_size()
            .map(|s| {
                let num_bits = s
                    .entry_sizes
                    .iter()
                    .map(|s| *s as usize)
                    .sum::<usize>()
                    .saturating_mul(8);

                let bitrate = (num_bits as f64) / duration_secds;
                bitrate.round() as u32
            })
    }
}

#[derive(Debug)]
pub struct TrakAtomRefMut<'a>(AtomRefMut<'a>);

impl<'a> TrakAtomRefMut<'a> {
    fn new(atom: &'a mut Atom) -> Self {
        Self(AtomRefMut(atom))
    }

    fn insert_child(&mut self, index: usize, child: Atom) {
        self.0.insert_child(index, child);
    }

    pub fn as_ref(&self) -> TrakAtomRef<'_> {
        TrakAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> TrakAtomRef<'a> {
        TrakAtomRef(self.0.into_ref())
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children()
    }

    pub fn header(&mut self) -> Option<&mut TrackHeaderAtom> {
        let atom = self.0.find_child(TKHD)?;
        debug_assert!(matches!(atom.data, Some(AtomData::TrackHeader(_))));
        match atom.data.as_mut()? {
            AtomData::TrackHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds or creates the MDIA atom
    pub fn media(&mut self) -> MdiaAtomRefMut<'_> {
        if let Some(index) = self.0.as_ref().child_position(MDIA) {
            return MdiaAtomRefMut(self.0.get_child(index));
        }
        let index = vec![TREF, EDTS, TKHD]
            .into_iter()
            .find_map(|typ| self.0.as_ref().child_position(typ))
            .map(|i| i + 1)
            .unwrap_or_default();
        self.0.insert_child(
            index,
            Atom::builder().header(AtomHeader::new(*MDIA)).build(),
        );
        MdiaAtomRefMut(self.0.get_child(index))
    }

    pub fn into_media(self) -> Option<MdiaAtomRefMut<'a>> {
        let atom = self.0.into_child(MDIA)?;
        Some(MdiaAtomRefMut(AtomRefMut(atom)))
    }

    pub fn add_track_reference(&mut self, references: impl Into<Vec<TrackReference>>) {
        // try and insert after track header but before mdia
        let tkhd_index = self.0.as_ref().child_position(TKHD);
        let mdia_index = self.0.as_ref().child_position(MDIA);
        let index = tkhd_index
            .map(|i| i + 1)
            .or_else(|| mdia_index.map(|i| i - 1))
            .unwrap_or_default();
        self.insert_child(
            index,
            Atom::builder()
                .header(AtomHeader::new(*TREF))
                .data(TrackReferenceAtom::new(references))
                .build(),
        );
    }

    pub fn add_edit_list(&mut self, entries: impl Into<Vec<EditEntry>>) {
        // try and insert after track header but before mdia
        let tkhd_index = self.0.as_ref().child_position(TKHD);
        let mdia_index = self.0.as_ref().child_position(MDIA);
        let index = tkhd_index
            .map(|i| i + 1)
            .or_else(|| mdia_index.map(|i| i - 1))
            .unwrap_or_default();
        self.insert_child(
            index,
            Atom::builder()
                .header(AtomHeader::new(*EDTS))
                .children(vec![Atom::builder()
                    .header(AtomHeader::new(*ELST))
                    .data(EditListAtom::new(entries))
                    .build()])
                .build(),
        );
    }

    /// Updates track metadata with the new audio bitrate
    ///
    /// Creates any missing atoms needed to do so
    pub fn update_audio_bitrate(&mut self, bitrate: u32) {
        let mut mdia = self.media();
        let mut minf = mdia.media_information();
        let mut stbl = minf.sample_table();
        let stsd = stbl.sample_description();

        let entry = stsd.find_or_create_entry(
            |entry| matches!(entry.data, SampleEntryData::Audio(_)),
            || SampleEntry {
                entry_type: SampleEntryType::Mp4a,
                data_reference_index: 0,
                data: SampleEntryData::Audio(AudioSampleEntry::default()),
            },
        );

        entry.entry_type = SampleEntryType::Mp4a;

        if let SampleEntryData::Audio(audio) = &mut entry.data {
            let mut sample_frequency = None;
            audio
                .extensions
                .retain(|ext| matches!(ext, StsdExtension::Esds(_)));
            let esds = audio.find_or_create_extension(
                |ext| matches!(ext, StsdExtension::Esds(_)),
                || StsdExtension::Esds(EsdsExtension::default()),
            );
            if let StsdExtension::Esds(esds) = esds {
                let cfg = esds
                    .es_descriptor
                    .decoder_config_descriptor
                    .get_or_insert_default();
                cfg.avg_bitrate = bitrate;
                cfg.max_bitrate = bitrate;
                if let Some(DecoderSpecificInfo::Audio(a, _)) = cfg.decoder_specific_info.as_ref() {
                    sample_frequency = Some(a.sampling_frequency.as_hz());
                }
            }
            audio.extensions.push(StsdExtension::Btrt(BtrtExtension {
                buffer_size_db: 0,
                avg_bitrate: bitrate,
                max_bitrate: bitrate,
            }));

            if let Some(hz) = sample_frequency {
                audio.sample_rate = hz as f32;
            }
        } else {
            // this indicates a programming error since we won't get here with parsed data
            unreachable!("STSD constructed with invalid data")
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MdiaAtomRef<'a>(AtomRef<'a>);

impl<'a> MdiaAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    /// Finds the MDHD atom
    pub fn header(&self) -> Option<&'a MediaHeaderAtom> {
        let atom = self.0.find_child(MDHD)?;
        match atom.data.as_ref()? {
            AtomData::MediaHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the HDLR atom
    pub fn handler_reference(&self) -> Option<&'a HandlerReferenceAtom> {
        let atom = self.0.find_child(HDLR)?;
        match atom.data.as_ref()? {
            AtomData::HandlerReference(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the MINF atom
    pub fn media_information(&self) -> MinfAtomRef<'a> {
        let atom = self.0.find_child(MINF);
        MinfAtomRef(AtomRef(atom))
    }
}

#[derive(Debug)]
pub struct MdiaAtomRefMut<'a>(AtomRefMut<'a>);

impl<'a> MdiaAtomRefMut<'a> {
    pub fn as_ref(&self) -> MdiaAtomRef<'_> {
        MdiaAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> MdiaAtomRef<'a> {
        MdiaAtomRef(self.0.into_ref())
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children()
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.into_children()
    }

    /// Finds the MDHD atom
    pub fn header(&mut self) -> Option<&mut MediaHeaderAtom> {
        let atom = self.0.find_child(MDHD)?;
        match atom.data.as_mut()? {
            AtomData::MediaHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the HDLR atom
    pub fn handler_reference(&mut self) -> Option<&mut HandlerReferenceAtom> {
        let atom = self.0.find_child(HDLR)?;
        match atom.data.as_mut()? {
            AtomData::HandlerReference(data) => Some(data),
            _ => None,
        }
    }

    /// Finds or creates the MINF atom
    pub fn media_information(&mut self) -> MinfAtomRefMut<'_> {
        if let Some(index) = self.0.as_ref().child_position(MINF) {
            return MinfAtomRefMut(self.0.get_child(index));
        }
        let index = vec![HDLR, MDHD]
            .into_iter()
            .find_map(|typ| self.0.as_ref().child_position(typ))
            .map(|i| i + 1)
            .unwrap_or_default();
        self.0.insert_child(
            index,
            Atom::builder().header(AtomHeader::new(*MINF)).build(),
        );
        MinfAtomRefMut(self.0.get_child(index))
    }

    /// Finds the MINF atom
    pub fn into_media_information(self) -> Option<MinfAtomRefMut<'a>> {
        let atom = self.0.into_child(MINF)?;
        Some(MinfAtomRefMut(AtomRefMut(atom)))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MinfAtomRef<'a>(AtomRef<'a>);

impl<'a> MinfAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    /// Finds the STBL atom
    pub fn sample_table(&self) -> StblAtomRef<'a> {
        let atom = self.0.find_child(STBL);
        StblAtomRef(AtomRef(atom))
    }
}

#[derive(Debug)]
pub struct MinfAtomRefMut<'a>(AtomRefMut<'a>);

impl<'a> MinfAtomRefMut<'a> {
    pub fn as_ref(&self) -> MinfAtomRef<'_> {
        MinfAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> MinfAtomRef<'a> {
        MinfAtomRef(self.0.into_ref())
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.into_children()
    }

    /// Finds or creates the STBL atom
    pub fn sample_table(&mut self) -> StblAtomRefMut<'_> {
        if let Some(index) = self.0.as_ref().child_position(STBL) {
            return StblAtomRefMut(self.0.get_child(index));
        }
        let index = vec![DINF, SMHD]
            .into_iter()
            .find_map(|typ| self.0.as_ref().child_position(typ))
            .map(|i| i + 1)
            .unwrap_or_default();
        self.0.insert_child(
            index,
            Atom::builder().header(AtomHeader::new(*STBL)).build(),
        );
        StblAtomRefMut(self.0.get_child(index))
    }

    /// Finds the STBL atom
    pub fn into_sample_table(self) -> Option<StblAtomRefMut<'a>> {
        let atom = self.0.into_child(STBL)?;
        Some(StblAtomRefMut(AtomRefMut(atom)))
    }
}

#[derive(Debug)]
pub struct StblAtomRef<'a>(AtomRef<'a>);

impl<'a> StblAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    /// Finds the STSD atom
    pub fn sample_description(&self) -> Option<&'a SampleDescriptionTableAtom> {
        let atom = self.0.find_child(STSD)?;
        match atom.data.as_ref()? {
            AtomData::SampleDescriptionTable(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STTS atom
    pub fn time_to_sample(&self) -> Option<&'a TimeToSampleAtom> {
        let atom = self.0.find_child(STTS)?;
        match atom.data.as_ref()? {
            AtomData::TimeToSample(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSC atom
    pub fn sample_to_chunk(&self) -> Option<&'a SampleToChunkAtom> {
        let atom = self.0.find_child(STSC)?;
        match atom.data.as_ref()? {
            AtomData::SampleToChunk(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSZ atom
    pub fn sample_size(&self) -> Option<&'a SampleSizeAtom> {
        let atom = self.0.find_child(STSZ)?;
        match atom.data.as_ref()? {
            AtomData::SampleSize(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STCO atom
    pub fn chunk_offset(&self) -> Option<&'a ChunkOffsetAtom> {
        let atom = self.0.find_child(STCO)?;
        match atom.data.as_ref()? {
            AtomData::ChunkOffset(data) => Some(data),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct StblAtomRefMut<'a>(AtomRefMut<'a>);

impl<'a> StblAtomRefMut<'a> {
    pub fn as_ref(&self) -> StblAtomRef<'_> {
        StblAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> StblAtomRef<'a> {
        StblAtomRef(self.0.into_ref())
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.into_children()
    }

    /// Finds or creates the STSD atom
    pub fn sample_description(&mut self) -> &'_ mut SampleDescriptionTableAtom {
        if let Some(index) = self.0.as_ref().child_position(STSD) {
            let stsd = self.0.get_child(index);
            if let Some(AtomData::SampleDescriptionTable(data)) = stsd.0.data.as_mut() {
                return data;
            }
            // this indicates programming error since we'll never end up here with a parsed Atom
            unreachable!("STSD constructed with invalid data")
        }

        let index = 0;
        self.0.insert_child(
            index,
            Atom::builder()
                .header(AtomHeader::new(*STSD))
                .data(SampleDescriptionTableAtom::default())
                .build(),
        );
        let stsd = self.0.get_child(index);
        if let Some(AtomData::SampleDescriptionTable(data)) = stsd.0.data.as_mut() {
            return data;
        }
        unreachable!()
    }

    /// Finds the STTS atom
    pub fn time_to_sample(&mut self) -> Option<&mut TimeToSampleAtom> {
        let atom = self.0.find_child(STTS)?;
        match atom.data.as_mut()? {
            AtomData::TimeToSample(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSC atom
    pub fn sample_to_chunk(&mut self) -> Option<&mut SampleToChunkAtom> {
        let atom = self.0.find_child(STSC)?;
        match atom.data.as_mut()? {
            AtomData::SampleToChunk(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSZ atom
    pub fn sample_size(&mut self) -> Option<&mut SampleSizeAtom> {
        let atom = self.0.find_child(STSZ)?;
        match atom.data.as_mut()? {
            AtomData::SampleSize(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STCO atom
    pub fn chunk_offset(&mut self) -> Option<&mut ChunkOffsetAtom> {
        let atom = self.0.find_child(STCO)?;
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
            trak: self.tracks[track_idx],
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
            .header()
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
