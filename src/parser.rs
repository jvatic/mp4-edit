/*!
 * This mod is concerned with parsing mp4 files.
 */

use anyhow::anyhow;
use derive_more::Display;
use futures_io::{AsyncRead, AsyncSeek};
use std::collections::VecDeque;
use std::fmt::{self, Debug};
use std::io::SeekFrom;
use std::ops::{Deref, DerefMut};
use thiserror::Error;

use crate::atom::util::parser::stream;
use crate::chunk_offset_builder;
pub use crate::reader::{Mp4Reader, NonSeekable, ReadCapability, Seekable};
use crate::{
    atom::{
        atom_ref::{AtomRef, AtomRefMut},
        container::{is_container_atom, META, META_VERSION_FLAGS_SIZE, MOOV},
        ftyp::{FileTypeAtom, FtypAtomRef, FtypAtomRefMut, FTYP},
        stco_co64::ChunkOffsets,
        stsc::SampleToChunkEntry,
        stts::TimeToSampleEntry,
        util::DebugEllipsis,
        AtomHeader, FourCC, MdiaAtomRefMut, MinfAtomRefMut, MoovAtomRef, MoovAtomRefMut, RawData,
        TrakAtomRef,
    },
    chunk_offset_builder::{ChunkInfo, ChunkOffsetBuilder},
    writer::SerializeAtom,
    Atom, AtomData,
};

pub const MDAT: &[u8; 4] = b"mdat";

/// This trait is implemented on [`AtomData`] and the inner value of each of it's variants.
///
/// Note that the [`AtomHeader`] has already been consumed, this trait is concerned with parsing the data.
pub(crate) trait ParseAtomData: Sized {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError>;
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
    pub(crate) fn new_unexpected_atom_oneof(atom_type: FourCC, expected: Vec<FourCC>) -> Self {
        if expected.len() == 1 {
            return Self::new_unexpected_atom(atom_type, expected[0]);
        }

        let expected = expected
            .into_iter()
            .map(|expected| expected.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Self {
            kind: ParseErrorKind::UnexpectedAtom,
            location: Some((0, 4)),
            source: Some(
                anyhow!("expected one of {expected}, got {atom_type}").into_boxed_dyn_error(),
            ),
        }
    }

    fn new_unexpected_atom(atom_type: FourCC, expected: FourCC) -> Self {
        let expected = FourCC::from(*expected);
        Self {
            kind: ParseErrorKind::UnexpectedAtom,
            location: Some((0, 4)),
            source: Some(anyhow!("expected {expected}, got {atom_type}").into_boxed_dyn_error()),
        }
    }

    pub(crate) fn from_winnow(
        error: winnow::error::ParseError<
            winnow::LocatingSlice<&winnow::Bytes>,
            winnow::error::ContextError,
        >,
    ) -> Self {
        use winnow::error::StrContext;
        let mut ctx_iter = error.inner().context().peekable();
        let mut ctx_tree = Vec::with_capacity(ctx_iter.size_hint().0);
        while let Some(ctx) = ctx_iter.next() {
            eprintln!("ctx: {ctx:?}");
            match ctx {
                StrContext::Expected(exp) => {
                    let mut label = None;
                    if matches!(ctx_iter.peek(), Some(StrContext::Label(_))) {
                        label = Some(ctx_iter.next().unwrap().to_string());
                    }
                    ctx_tree.push(format!(
                        "{}({exp})",
                        label.map(|label| label.to_string()).unwrap_or_default()
                    ));
                }
                StrContext::Label(label) => {
                    ctx_tree.push(label.to_string());
                }
                _ => {}
            }
        }
        ctx_tree.reverse();

        Self {
            kind: crate::parser::ParseErrorKind::AtomParsing,
            location: Some((error.offset(), 0)),
            source: match ctx_tree {
                ctx if ctx.is_empty() => None,
                ctx => Some(anyhow::format_err!("{}", ctx.join(" -> ")).into_boxed_dyn_error()),
            },
        }
    }
}

impl
    From<
        winnow::error::ParseError<
            winnow::LocatingSlice<&winnow::Bytes>,
            winnow::error::ContextError,
        >,
    > for ParseError
{
    fn from(
        value: winnow::error::ParseError<
            winnow::LocatingSlice<&winnow::Bytes>,
            winnow::error::ContextError,
        >,
    ) -> Self {
        ParseError::from_winnow(value)
    }
}

pub struct Parser<R, C: ReadCapability = NonSeekable> {
    reader: Mp4Reader<R, C>,
    mdat: Option<AtomHeader>,
}

impl<R: AsyncRead + AsyncSeek + Unpin + Send> Parser<R, Seekable> {
    pub fn new_seekable(reader: R) -> Self {
        Parser {
            reader: Mp4Reader::<R, Seekable>::new(reader),
            mdat: None,
        }
    }

    /// parses metadata atoms, both before and after mdat if moov isn't found before
    pub async fn parse_metadata(mut self) -> Result<MdatParser<R, Seekable>, ParseError> {
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
}

impl<R: AsyncRead + Unpin + Send> Parser<R, NonSeekable> {
    pub fn new(reader: R) -> Self {
        Parser {
            reader: Mp4Reader::<R, NonSeekable>::new(reader),
            mdat: None,
        }
    }

    /// parses metadata atoms until mdat found
    pub async fn parse_metadata(mut self) -> Result<MdatParser<R, NonSeekable>, ParseError> {
        let atoms = self.parse_metadata_inner(None).await?;
        Ok(MdatParser::new(
            self.reader,
            Metadata::new(atoms),
            self.mdat,
        ))
    }
}

impl<R: AsyncRead + Unpin + Send, C: ReadCapability> Parser<R, C> {
    async fn parse_metadata_inner(
        &mut self,
        length_limit: Option<usize>,
    ) -> Result<Vec<Atom>, ParseError> {
        let start_offset = self.reader.current_offset;

        let mut atoms = Vec::new();

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

            // only parse as far as the mdat atom
            if header.atom_type == MDAT {
                self.mdat = Some(header);
                break;
            }

            if is_container_atom(header.atom_type) {
                // META containers have additional header data
                let (size, data) = if header.atom_type == META {
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

                atoms.push(container_atom);
            } else {
                let atom_data = self.parse_atom_data(&header).await?;
                let atom = Atom {
                    header,
                    data: Some(atom_data),
                    children: Vec::new(),
                };
                atoms.push(atom);
            }
        }

        Ok(atoms)
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
        let input = stream(&content_data);

        AtomData::parse_atom_data(header.atom_type, &input).map_err(|err| {
            let (header_offset, _) = header.location();
            let content_offset = header_offset + header.header_size;
            ParseError {
                kind: ParseErrorKind::AtomParsing,
                location: Some(err.location.map_or_else(
                    || (content_offset, 0),
                    |(offset, size)| (content_offset + offset, size),
                )),
                source: Some(anyhow::Error::from(err).context(header.atom_type).into()),
            }
        })
    }
}

pub struct MdatParser<R, C: ReadCapability> {
    meta: Metadata,
    reader: Option<Mp4Reader<R, C>>,
    mdat: Option<AtomHeader>,
}

impl<R, C: ReadCapability> Clone for MdatParser<R, C> {
    fn clone(&self) -> Self {
        Self {
            meta: self.meta.clone(),
            reader: None,
            mdat: None,
        }
    }
}

impl<R, C: ReadCapability> Deref for MdatParser<R, C> {
    type Target = Metadata;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl<R, C: ReadCapability> DerefMut for MdatParser<R, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meta
    }
}

impl<R, C: ReadCapability> MdatParser<R, C> {
    fn new(reader: Mp4Reader<R, C>, meta: Metadata, mdat: Option<AtomHeader>) -> Self {
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
    pub fn chunks(&mut self) -> Result<ChunkParser<'_, R, C>, ParseError> {
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
    pub(crate) fn new(atoms: Vec<Atom>) -> Self {
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

    fn atom_position(&self, typ: FourCC) -> Option<usize> {
        self.atoms.iter().position(|a| a.header.atom_type == typ)
    }

    fn find_atom(&self, typ: FourCC) -> AtomRef<'_> {
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
    pub fn update_chunk_offsets(
        &mut self,
    ) -> Result<chunk_offset_builder::BuildMetadata, UpdateChunkOffsetError> {
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

        let (mut chunk_offsets, build_meta) = chunk_offsets
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

        Ok(build_meta)
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

pub struct ChunkParser<'a, R, C: ReadCapability> {
    reader: Mp4Reader<R, C>,
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

impl<'a, R: AsyncRead + Unpin + Send, C: ReadCapability> ChunkParser<'a, R, C> {
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
