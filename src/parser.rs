use anyhow::anyhow;
use bon::bon;
use derive_more::Display;
use futures_io::{AsyncRead, AsyncSeek};
use futures_util::io::{AsyncReadExt, AsyncSeekExt, Cursor};
use std::collections::VecDeque;
use std::fmt::{self, Debug};
use std::future::Future;
use std::io::SeekFrom;
use std::ops::{Deref, DerefMut, Range, RangeBounds};
use std::time::Duration;
use thiserror::Error;

use crate::atom::containers::{
    DINF, EDTS, MDIA, MFRA, MINF, MOOF, MOOV, SCHI, SINF, STBL, TRAF, TRAK, UDTA,
};
use crate::atom::hdlr::HandlerType;
use crate::atom::stco_co64::ChunkOffsets;
use crate::atom::stsc::SampleToChunkEntry;
use crate::atom::stsd::{
    AudioSampleEntry, BtrtExtension, DecoderSpecificInfo, EsdsExtension, SampleEntry,
    SampleEntryData, SampleEntryType, StsdExtension,
};
use crate::atom::stts::TimeToSampleEntry;
use crate::atom::util::time::{scaled_duration_range, unscaled_duration};
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

/// Async trait for parsing atoms from an `AsyncRead` stream
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

        let size = u64::from(u32::from_be_bytes([
            header[0], header[1], header[2], header[3],
        ]));
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

    /// Transforms into (reader, `current_offset`, atoms)
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
        for atom in &mut self.atoms {
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

    /// Returns the sum of `metadata_size` and `mdat_size`
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
                .and_then(MdiaAtomRefMut::into_media_information)
                .and_then(MinfAtomRefMut::into_sample_table)
                .ok_or(UpdateChunkOffsetError::SampleTableNotFound)?;
            let stco = stbl.chunk_offset();
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
        self.iter.as_mut().and_then(std::iter::Iterator::next)
    }
}

impl DoubleEndedIterator for AtomIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .as_mut()
            .and_then(std::iter::DoubleEndedIterator::next_back)
    }
}

impl ExactSizeIterator for AtomIter<'_> {
    fn len(&self) -> usize {
        self.iter
            .as_ref()
            .map(ExactSizeIterator::len)
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

    fn into_child(self, typ: &[u8; 4]) -> Option<&'a mut Atom> {
        self.into_children()
            .find(|atom| atom.header.atom_type == typ)
    }

    fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children.iter_mut()
    }

    fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        AtomIterMut::from_atom(self.0)
    }

    fn insert_child(&mut self, index: usize, child: Atom) -> AtomRefMut<'_> {
        self.0.children.insert(index, child);
        self.get_child(index)
    }
}

#[bon]
impl<'a> AtomRefMut<'a> {
    #[builder]
    fn find_or_insert_child(
        &mut self,
        #[builder(start_fn)] atom_type: &[u8; 4],
        #[builder(default = Vec::new())] insert_before: Vec<&[u8; 4]>,
        #[builder(default = Vec::new())] insert_after: Vec<&[u8; 4]>,
        insert_index: Option<usize>,
        insert_data: Option<AtomData>,
    ) -> AtomRefMut<'_> {
        if let Some(index) = self.as_ref().child_position(atom_type) {
            self.get_child(index)
        } else {
            let index = insert_index.unwrap_or_else(|| {
                self.get_insert_position()
                    .before(insert_before)
                    .after(insert_after)
                    .call()
            });
            self.insert_child(
                index,
                Atom::builder()
                    .header(AtomHeader::new(*atom_type))
                    .maybe_data(insert_data)
                    .build(),
            )
        }
    }

    #[builder]
    fn get_insert_position(
        &self,
        #[builder(default = Vec::new())] before: Vec<&[u8; 4]>,
        #[builder(default = Vec::new())] after: Vec<&[u8; 4]>,
    ) -> usize {
        before
            .into_iter()
            .find_map(|typ| self.as_ref().child_rposition(typ))
            .or_else(|| {
                after
                    .into_iter()
                    .find_map(|typ| self.as_ref().child_position(typ))
                    .map(|i| i + 1)
            })
            .unwrap_or_default()
    }
}

/// Unwrap atom data enum given variant type.
///
/// # Example
/// ```ignore
/// let mut data = Atom::builder()
///     .header(AtomHeader::new(*TKHD))
///     .data(AtomData::TrackHeader(TrackHeaderAtom::default()))
///     .build();
/// let _: &mut TrackHeaderAtom = unwrap_atom_data!(
///     AtomRefMut(&mut data),
///     AtomData::TrackHeader,
/// );
/// ```
macro_rules! unwrap_atom_data {
    ($ref:expr, $variant:path $(,)?) => {{
        let atom = $ref.0;
        if let Some($variant(data)) = &mut atom.data {
            data
        } else {
            unreachable!(
                "invalid {} atom: data is None or the wrong variant",
                atom.header.atom_type,
            )
        }
    }};
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
        self.0.atom_mut().data = Some(data.into());
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

    /// Finds or inserts MVHD atom
    pub fn header(&mut self) -> &'_ mut MovieHeaderAtom {
        unwrap_atom_data!(
            self.0.find_or_insert_child(MVHD).call(),
            AtomData::MovieHeader,
        )
    }

    /// Finds or inserts UDTA atom
    pub fn user_data(&mut self) -> UserDataAtomRefMut<'_> {
        UserDataAtomRefMut(
            self.0
                .find_or_insert_child(UDTA)
                .insert_after(vec![TRAK, MVHD])
                .call(),
        )
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
    pub fn tracks_retain<P>(&mut self, mut pred: P) -> &mut Self
    where
        P: FnMut(TrakAtomRef) -> bool,
    {
        self.0
             .0
            .children
            .retain(|a| a.header.atom_type != TRAK || pred(TrakAtomRef::new(a)));
        self
    }
}

#[bon]
impl<'a> MoovAtomRefMut<'a> {
    /// Trim duration from tracks.
    ///
    /// See also [`Self::retain_duration`].
    #[builder(finish_fn(name = "trim"), builder_type = TrimDuration)]
    pub fn trim_duration(
        &mut self,
        from_start: Option<Duration>,
        from_end: Option<Duration>,
    ) -> &mut Self {
        use std::ops::Bound;
        let start_duration = from_start.map(|d| (Bound::Unbounded, Bound::Excluded(d)));
        let end_duration = from_end.map(|d| {
            let d = self.header().duration().saturating_sub(d);
            (Bound::Included(d), Bound::Unbounded)
        });
        let trim_ranges = vec![start_duration, end_duration]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        self.trim_duration_ranges(&trim_ranges)
    }

    /// Retains given duration range, trimming everything before and after.
    ///
    /// See also [`Self::trim_duration`].
    pub fn retain_duration(&mut self, range: Range<Duration>) -> &mut Self {
        use std::ops::Bound;
        let trim_ranges = vec![
            (Bound::Unbounded, Bound::Included(range.start)),
            (Bound::Excluded(range.end), Bound::Unbounded),
        ];
        self.trim_duration_ranges(&trim_ranges)
    }

    fn trim_duration_ranges<R>(&mut self, trim_ranges: &[R]) -> &mut Self
    where
        R: RangeBounds<Duration> + Clone + Debug,
    {
        // TODO: after trimming samples,
        // - [ ] Update mdhd duration to match: sample_count × 1024
        // - [ ] Update mvhd duration proportionally: (mdhd_duration × 600) / 44100
        let movie_timescale = u64::from(self.header().timescale);
        let trimmed_duration = self
            .tracks()
            .map(|mut trak| trak.trim_duration(movie_timescale, trim_ranges))
            .min();
        if let Some(trimmed_duration) = trimmed_duration {
            self.header().update_duration(|d| d - trimmed_duration);
        }
        self
    }
}

#[bon]
impl<'a, 'b, S: trim_duration::State> TrimDuration<'a, 'b, S> {
    #[builder(finish_fn(name = "trim"), builder_type = TrimDurationRanges)]
    pub fn ranges<R>(
        self,
        #[builder(start_fn)] ranges: impl IntoIterator<Item = R>,
    ) -> &'b mut MoovAtomRefMut<'a>
    where
        R: RangeBounds<Duration> + Clone + Debug,
        S::FromEnd: trim_duration::IsUnset,
        S::FromStart: trim_duration::IsUnset,
    {
        self.self_receiver
            .trim_duration_ranges(&ranges.into_iter().collect::<Vec<_>>())
    }
}

#[bon]
impl<'a> MoovAtomRefMut<'a> {
    /// Adds trak atom to moov
    #[builder]
    pub fn add_track(
        &mut self,
        #[builder(default = Vec::new())] children: Vec<Atom>,
    ) -> TrakAtomRefMut<'_> {
        let trak = Atom::builder()
            .header(AtomHeader::new(*TRAK))
            .children(children)
            .build();
        let index = self.0.get_insert_position().after(vec![TRAK, MDHD]).call();
        TrakAtomRefMut(self.0.insert_child(index, trak))
    }
}

pub struct UserDataAtomRefMut<'a>(AtomRefMut<'a>);

impl UserDataAtomRefMut<'_> {
    /// Finds or inserts CHPL atom
    pub fn chapter_list(&mut self) -> &'_ mut ChapterListAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(CHPL)
                .insert_after(vec![META])
                .insert_data(AtomData::ChapterList(ChapterListAtom::default()))
                .call(),
            AtomData::ChapterList,
        )
    }
}

#[derive(Clone, Copy)]
pub struct TrakAtomRef<'a>(AtomRef<'a>);

impl fmt::Debug for TrakAtomRef<'_> {
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
            .map_or(0, |s| {
                if s.entry_sizes.is_empty() {
                    s.sample_size * s.sample_count
                } else {
                    s.entry_sizes.iter().sum::<u32>()
                }
            }) as usize
    }

    /// Calculates the track's bitrate
    ///
    /// Returns None if either stsz or mdhd atoms can't be found
    pub fn bitrate(&self) -> Option<u32> {
        let duration_secds = self
            .media()
            .header()
            .map(|mdhd| (mdhd.duration as f64) / f64::from(mdhd.timescale))?;

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

    pub fn as_ref(&self) -> TrakAtomRef<'_> {
        TrakAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> TrakAtomRef<'a> {
        TrakAtomRef(self.0.into_ref())
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children()
    }

    /// Finds or inserts the TKHD atom
    pub fn header(&mut self) -> &mut TrackHeaderAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(TKHD)
                .insert_data(AtomData::TrackHeader(TrackHeaderAtom::default()))
                .call(),
            AtomData::TrackHeader,
        )
    }

    /// Finds or creates the MDIA atom
    pub fn media(&mut self) -> MdiaAtomRefMut<'_> {
        MdiaAtomRefMut(
            self.0
                .find_or_insert_child(MDIA)
                .insert_after(vec![TREF, EDTS, TKHD])
                .call(),
        )
    }

    /// Finds the MDIA atom
    pub fn into_media(self) -> Option<MdiaAtomRefMut<'a>> {
        let atom = self.0.into_child(MDIA)?;
        Some(MdiaAtomRefMut(AtomRefMut(atom)))
    }

    /// Finds or creates the EDTS atom
    pub fn edit_list_container(&mut self) -> EdtsAtomRefMut<'_> {
        EdtsAtomRefMut(
            self.0
                .find_or_insert_child(EDTS)
                .insert_after(vec![TREF, TKHD])
                .call(),
        )
    }

    /// Finds or inserts the TREF atom
    pub fn track_reference(&mut self) -> &mut TrackReferenceAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(TREF)
                .insert_after(vec![TKHD])
                .insert_data(AtomData::TrackReference(TrackReferenceAtom::default()))
                .call(),
            AtomData::TrackReference,
        )
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

    /// trims given duration range, excluding partially matched samples, and returns the actual duration trimmed
    fn trim_duration<R>(&mut self, movie_timescale: u64, trim_ranges: &[R]) -> Duration
    where
        R: RangeBounds<Duration> + Clone + Debug,
    {
        let mut mdia = self.media();
        let media_timescale = u64::from(mdia.header().timescale);
        let mut minf = mdia.media_information();
        let mut stbl = minf.sample_table();

        let scaled_ranges = trim_ranges
            .iter()
            .cloned()
            .map(|range| scaled_duration_range(range, media_timescale))
            .collect::<Vec<_>>();

        // Step 1: Determine which samples to remove based on time
        let (trimmed_duration, sample_indices_to_remove) =
            stbl.time_to_sample().trim_duration(&scaled_ranges);

        let trimmed_duration = unscaled_duration(trimmed_duration, media_timescale);

        // Step 2: Update sample sizes
        stbl.sample_size()
            .remove_sample_indices(&sample_indices_to_remove);

        // Step 3: Calculate and remove chunks based on samples
        let total_chunks = stbl.chunk_offset().chunk_count();
        let chunk_indices_to_remove = stbl
            .sample_to_chunk()
            .remove_sample_indices(&sample_indices_to_remove, total_chunks);

        // Step 4: Remove chunk offsets
        stbl.chunk_offset()
            .remove_chunk_indices(&chunk_indices_to_remove);

        // Step 5: Update headers
        mdia.header().update_duration(|d| d - trimmed_duration);
        self.header()
            .update_duration(movie_timescale, |d| d - trimmed_duration);

        trimmed_duration
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

    /// Finds or inserts the MDHD atom
    pub fn header(&mut self) -> &mut MediaHeaderAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(MDHD)
                .insert_data(AtomData::MediaHeader(MediaHeaderAtom::default()))
                .call(),
            AtomData::MediaHeader,
        )
    }

    /// Finds or inserts the HDLR atom
    pub fn handler_reference(&mut self) -> &mut HandlerReferenceAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(HDLR)
                .insert_data(AtomData::HandlerReference(HandlerReferenceAtom::default()))
                .call(),
            AtomData::HandlerReference,
        )
    }

    /// Finds or inserts the MINF atom
    pub fn media_information(&mut self) -> MinfAtomRefMut<'_> {
        MinfAtomRefMut(
            self.0
                .find_or_insert_child(MINF)
                .insert_after(vec![HDLR, MDHD])
                .call(),
        )
    }

    /// Finds the MINF atom
    pub fn into_media_information(self) -> Option<MinfAtomRefMut<'a>> {
        let atom = self.0.into_child(MINF)?;
        Some(MinfAtomRefMut(AtomRefMut(atom)))
    }
}

#[derive(Debug)]
pub struct EdtsAtomRefMut<'a>(AtomRefMut<'a>);

impl EdtsAtomRefMut<'_> {
    /// Finds or creates the ELST atom
    pub fn edit_list(&mut self) -> &mut EditListAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(ELST)
                .insert_data(AtomData::EditList(EditListAtom::default()))
                .call(),
            AtomData::EditList,
        )
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

    /// Finds or inserts the STBL atom
    pub fn sample_table(&mut self) -> StblAtomRefMut<'_> {
        StblAtomRefMut(
            self.0
                .find_or_insert_child(STBL)
                .insert_after(vec![DINF, SMHD])
                .call(),
        )
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

    /// Finds or inserts the STSD atom
    pub fn sample_description(&mut self) -> &'_ mut SampleDescriptionTableAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STSD)
                .insert_data(AtomData::SampleDescriptionTable(
                    SampleDescriptionTableAtom::default(),
                ))
                .call(),
            AtomData::SampleDescriptionTable,
        )
    }

    /// Finds or inserts the STTS atom
    pub fn time_to_sample(&mut self) -> &mut TimeToSampleAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STTS)
                .insert_after(vec![STTS, STSD])
                .insert_data(AtomData::TimeToSample(TimeToSampleAtom::default()))
                .call(),
            AtomData::TimeToSample,
        )
    }

    /// Finds or inserts the STSC atom
    pub fn sample_to_chunk(&mut self) -> &mut SampleToChunkAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STSC)
                .insert_after(vec![STTS, STSD])
                .insert_data(AtomData::SampleToChunk(SampleToChunkAtom::default()))
                .call(),
            AtomData::SampleToChunk,
        )
    }

    /// Finds or inserts the STSZ atom
    pub fn sample_size(&mut self) -> &mut SampleSizeAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STSZ)
                .insert_after(vec![STSC, STSD])
                .insert_data(AtomData::SampleSize(SampleSizeAtom::default()))
                .call(),
            AtomData::SampleSize,
        )
    }

    /// Finds or inserts the STCO atom
    pub fn chunk_offset(&mut self) -> &mut ChunkOffsetAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STCO)
                .insert_after(vec![STSZ, STSD])
                .insert_data(AtomData::ChunkOffset(ChunkOffsetAtom::default()))
                .call(),
            AtomData::ChunkOffset,
        )
    }
}

pub struct ChunkParser<'a, R> {
    reader: Mp4Reader<R>,
    /// Reference to each track's metadata
    tracks: Vec<TrakAtomRef<'a>>,
    /// Chunk offsets for each track
    chunk_offsets: Vec<&'a [u64]>,
    /// [`SampleToChunkEntry`]s for each track
    sample_to_chunk: Vec<&'a [SampleToChunkEntry]>,
    /// Sample sizes for each track
    sample_sizes: Vec<&'a [u32]>,
    /// [`TimeToSampleEntry`]s for each track
    time_to_sample: Vec<&'a [TimeToSampleEntry]>,
    /// [`ChunkInfo`]s for each track
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

impl fmt::Debug for Chunk<'_> {
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

impl fmt::Debug for Sample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sample")
            .field("size", &self.size)
            .field("duration", &self.duration)
            .field("timescale", &self.timescale)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;

    use super::*;
    use crate::atom::{
        containers::{DINF, MDIA, MINF, MOOV, STBL, TRAK},
        dref::{DataReferenceAtom, DataReferenceEntry, DREF},
        ftyp::{FileTypeAtom, FTYP},
        hdlr::{HandlerReferenceAtom, HandlerType, HDLR},
        mdhd::{MediaHeaderAtom, MDHD},
        mvhd::{MovieHeaderAtom, MVHD},
        smhd::{SoundMediaHeaderAtom, SMHD},
        stco_co64::{ChunkOffsetAtom, STCO},
        stsc::{SampleToChunkAtom, SampleToChunkEntry, STSC},
        stsd::{SampleDescriptionTableAtom, STSD},
        stsz::{SampleSizeAtom, STSZ},
        stts::{TimeToSampleAtom, TimeToSampleEntry, STTS},
        tkhd::{TrackHeaderAtom, TKHD},
        util::time::scaled_duration,
        Atom, AtomHeader,
    };
    use std::time::Duration;

    #[bon::builder(finish_fn(name = "build"))]
    fn create_test_metadata(
        movie_timescale: u32,
        media_timescale: u32,
        duration: Duration,
    ) -> Metadata {
        let atoms = vec![
            // Create ftyp atom
            Atom::builder()
                .header(AtomHeader::new(*FTYP))
                .data(
                    FileTypeAtom::builder()
                        .major_brand(*b"isom")
                        .minor_version(512)
                        .compatible_brands(
                            vec![*b"isom", *b"iso2", *b"mp41"]
                                .into_iter()
                                .map(FourCC::from)
                                .collect::<Vec<_>>(),
                        )
                        .build(),
                )
                .build(),
            // Create moov atom with a single track with complex sample data
            Atom::builder()
                .header(AtomHeader::new(*MOOV))
                .children(vec![
                    // Movie header (mvhd)
                    Atom::builder()
                        .header(AtomHeader::new(*MVHD))
                        .data(
                            MovieHeaderAtom::builder()
                                .timescale(movie_timescale)
                                .duration(scaled_duration(duration, movie_timescale as u64))
                                .next_track_id(2)
                                .build(),
                        )
                        .build(),
                    // Track (trak) with complex sample data
                    create_test_track(movie_timescale, media_timescale, duration),
                ])
                .build(),
        ];

        Metadata::new(atoms.into())
    }

    fn create_test_track(movie_timescale: u32, media_timescale: u32, duration: Duration) -> Atom {
        Atom::builder()
            .header(AtomHeader::new(*TRAK))
            .children(vec![
                // Track header (tkhd)
                Atom::builder()
                    .header(AtomHeader::new(*TKHD))
                    .data(
                        TrackHeaderAtom::builder()
                            .track_id(1)
                            .duration(scaled_duration(duration, movie_timescale as u64))
                            .build(),
                    )
                    .build(),
                // Media (mdia) with complex sample data
                create_test_media(media_timescale, duration),
            ])
            .build()
    }

    fn create_test_media(media_timescale: u32, duration: Duration) -> Atom {
        Atom::builder()
            .header(AtomHeader::new(*MDIA))
            .children(vec![
                // Media header (mdhd)
                Atom::builder()
                    .header(AtomHeader::new(*MDHD))
                    .data(
                        MediaHeaderAtom::builder()
                            .timescale(media_timescale)
                            .duration(scaled_duration(duration, media_timescale as u64))
                            .build(),
                    )
                    .build(),
                // Handler reference (hdlr)
                Atom::builder()
                    .header(AtomHeader::new(*HDLR))
                    .data(
                        HandlerReferenceAtom::builder()
                            .handler_type(HandlerType::Audio)
                            .name("SoundHandler".to_string())
                            .build(),
                    )
                    .build(),
                // Media information (minf)
                create_test_media_info(duration, media_timescale),
            ])
            .build()
    }

    fn create_test_media_info(duration: Duration, media_timescale: u32) -> Atom {
        Atom::builder()
            .header(AtomHeader::new(*MINF))
            .children(vec![
                // Sound media information header (smhd)
                Atom::builder()
                    .header(AtomHeader::new(*SMHD))
                    .data(SoundMediaHeaderAtom::default())
                    .build(),
                // Data information (dinf)
                Atom::builder()
                    .header(AtomHeader::new(*DINF))
                    .children(vec![
                        // Data reference (dref)
                        Atom::builder()
                            .header(AtomHeader::new(*DREF))
                            .data(
                                DataReferenceAtom::builder()
                                    .entry(DataReferenceEntry::builder().url("").build())
                                    .build(),
                            )
                            .build(),
                    ])
                    .build(),
                // Sample table (stbl) with complex data
                create_test_sample_table(duration, media_timescale),
            ])
            .build()
    }

    fn create_test_sample_table(duration: Duration, media_timescale: u32) -> Atom {
        // Create one sample per second with 2 samples per chunk
        let duration_secs = duration.as_secs() as u32;
        let total_samples = duration_secs; // One sample per second
        let samples_per_chunk = 2u32;

        // Calculate number of chunks needed
        let total_chunks = (total_samples + samples_per_chunk - 1) / samples_per_chunk;

        // bytes per sample
        const SAMPLE_SIZE: usize = 256;

        // Create chunk offsets
        let mut chunk_offsets = Vec::new();
        let mut current_offset = 1000u64;
        for _ in 0..total_chunks {
            chunk_offsets.push(current_offset);
            current_offset += samples_per_chunk as u64 * SAMPLE_SIZE as u64;
        }

        // Create sample sizes (256 bytes per sample)
        let sample_sizes: Vec<u32> = vec![SAMPLE_SIZE as u32; total_samples as usize];

        // Create single sample-to-chunk entry (all chunks have 2 samples)
        let stsc_entries = vec![SampleToChunkEntry::builder()
            .first_chunk(1)
            .samples_per_chunk(samples_per_chunk)
            .sample_description_index(1)
            .build()];

        Atom::builder()
            .header(AtomHeader::new(*STBL))
            .children(vec![
                // Sample Description (stsd)
                Atom::builder()
                    .header(AtomHeader::new(*STSD))
                    .data(SampleDescriptionTableAtom::default())
                    .build(),
                // Time to Sample (stts) - each sample represents 1 second
                Atom::builder()
                    .header(AtomHeader::new(*STTS))
                    .data(
                        TimeToSampleAtom::builder()
                            .entry(
                                TimeToSampleEntry::builder()
                                    .sample_count(total_samples)
                                    .sample_duration(media_timescale) // 1 second per sample
                                    .build(),
                            )
                            .build(),
                    )
                    .build(),
                // Sample to Chunk (stsc) - 2 samples per chunk
                Atom::builder()
                    .header(AtomHeader::new(*STSC))
                    .data(SampleToChunkAtom::from(stsc_entries))
                    .build(),
                // Sample Size (stsz)
                Atom::builder()
                    .header(AtomHeader::new(*STSZ))
                    .data(SampleSizeAtom::builder().entry_sizes(sample_sizes).build())
                    .build(),
                // Chunk Offset (stco)
                Atom::builder()
                    .header(AtomHeader::new(*STCO))
                    .data(
                        ChunkOffsetAtom::builder()
                            .chunk_offsets(chunk_offsets)
                            .build(),
                    )
                    .build(),
            ])
            .build()
    }

    fn test_moov_trim_duration<F>(test_case: F)
    where
        F: FnOnce() -> TrimDurationTestCase,
    {
        let test_case = test_case();

        let movie_timescale = 1_000;
        let media_timescale = 10_000;

        // Create fresh metadata for each test case
        let mut metadata = create_test_metadata()
            .movie_timescale(movie_timescale)
            .media_timescale(media_timescale)
            .duration(test_case.original_duration)
            .build();

        // Perform the trim operation with the range bounds
        let range = (test_case.start_bound, test_case.end_bound);
        metadata
            .moov_mut()
            .trim_duration()
            .ranges(vec![range])
            .trim();

        // Verify movie header duration was updated
        let new_movie_duration = metadata.moov().header().map(|h| h.duration).unwrap_or(0);
        let expected_movie_duration = scaled_duration(
            test_case.expected_remaining_duration,
            movie_timescale as u64,
        );
        assert_eq!(
            new_movie_duration, expected_movie_duration,
            "Movie duration should match expected",
        );

        // Verify track header duration was updated
        let new_track_duration = metadata
            .moov()
            .into_tracks_iter()
            .next()
            .and_then(|t| t.header().map(|h| h.duration))
            .unwrap_or(0);
        let expected_track_duration = scaled_duration(
            test_case.expected_remaining_duration,
            movie_timescale as u64,
        );
        assert_eq!(
            new_track_duration, expected_track_duration,
            "Track duration should match expected",
        );

        // Verify media header duration was updated
        let new_media_duration = metadata
            .moov()
            .into_tracks_iter()
            .next()
            .map(|t| t.media().header().map(|h| h.duration).unwrap_or(0))
            .unwrap_or(0);
        let expected_media_duration = scaled_duration(
            test_case.expected_remaining_duration,
            media_timescale as u64,
        );
        assert_eq!(
            new_media_duration, expected_media_duration,
            "Media duration should match expected",
        );

        // Verify sample table structure is still valid
        let track = metadata.moov().into_tracks_iter().next().unwrap();
        let stbl = track.media().media_information().sample_table();

        // Validate that all required sample table atoms exist
        let stts = stbl
            .time_to_sample()
            .expect("Time-to-sample atom should exist");
        let stsc = stbl
            .sample_to_chunk()
            .expect("Sample-to-chunk atom should exist");
        let stsz = stbl.sample_size().expect("Sample-size atom should exist");
        let stco = stbl.chunk_offset().expect("Chunk-offset atom should exist");

        // Calculate total samples from sample sizes
        let total_samples = stsz.sample_count() as u32;
        if test_case.expected_remaining_duration != Duration::ZERO {
            assert!(total_samples > 0, "Sample table should have samples",);
        }

        // Validate time-to-sample consistency
        let stts_total_samples: u32 = stts.entries.iter().map(|entry| entry.sample_count).sum();
        assert_eq!(
            stts_total_samples, total_samples,
            "Time-to-sample total samples should match sample size count",
        );

        // Validate sample-to-chunk references
        let chunk_count = stco.chunk_count() as u32;
        assert!(chunk_count > 0, "Should have at least one chunk",);

        // Verify all chunk references in stsc are valid
        for entry in stsc.entries.iter() {
            assert!(
                entry.first_chunk >= 1 && entry.first_chunk <= chunk_count,
                "Sample-to-chunk first_chunk {} should be between 1 and {}",
                entry.first_chunk,
                chunk_count,
            );
            assert!(
                entry.samples_per_chunk > 0,
                "Sample-to-chunk samples_per_chunk should be > 0",
            );
        }

        // Verify expected duration consistency with time-to-sample
        let total_duration: u64 = stts
            .entries
            .iter()
            .map(|entry| entry.sample_count as u64 * entry.sample_duration as u64)
            .sum();
        let expected_duration_scaled = scaled_duration(
            test_case.expected_remaining_duration,
            media_timescale as u64,
        );

        assert_eq!(
            total_duration, expected_duration_scaled,
            "Sample table total duration should match the expected duration",
        );
    }

    struct TrimDurationTestCase {
        original_duration: Duration,
        start_bound: Bound<Duration>,
        end_bound: Bound<Duration>,
        expected_remaining_duration: Duration,
    }

    macro_rules! test_moov_trim_duration {
        ($($name:ident => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    test_moov_trim_duration($test_case);
                }
            )*
        };
    }

    test_moov_trim_duration!(
        trim_start_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::ZERO),
            end_bound: Bound::Included(Duration::from_secs(2)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_end_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(8)),
            end_bound: Bound::Included(Duration::from_secs(10)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_middle_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(4)),
            end_bound: Bound::Included(Duration::from_secs(6)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_middle_included_start_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(2)),
            end_bound: Bound::Included(Duration::from_secs(4)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_middle_excluded_start_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_millis(10_000),
            start_bound: Bound::Excluded(Duration::from_millis(1_999)),
            end_bound: Bound::Included(Duration::from_millis(4_000)),
            expected_remaining_duration: Duration::from_millis(8_000),
        },
        trim_middle_excluded_end_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(1)),
            end_bound: Bound::Excluded(Duration::from_secs(3)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_start_unbounded_5_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Unbounded,
            end_bound: Bound::Included(Duration::from_secs(5)),
            expected_remaining_duration: Duration::from_secs(5),
        },
        trim_end_unbounded_6_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(100),
            start_bound: Bound::Included(Duration::from_secs(94)),
            end_bound: Bound::Unbounded,
            expected_remaining_duration: Duration::from_secs(94),
        },
        trim_unbounded => || TrimDurationTestCase {
            original_duration: Duration::from_secs(100),
            start_bound: Bound::Unbounded,
            end_bound: Bound::Unbounded,
            expected_remaining_duration: Duration::ZERO,
        },
    );
}
