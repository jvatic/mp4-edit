use derive_more::Display;
use futures_io::AsyncRead;
use futures_util::io::{AsyncReadExt, Cursor};
use futures_util::stream::{Stream, StreamExt, TryStreamExt};
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
        meta::{MetadataAtom, META},
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

pub struct Parser {
    current_offset: usize,
    atoms: Vec<Atom>,
}

struct ParsedAtom {
    atom_type: [u8; 4],
    size: u64,
    offset: u64,
    header_size: usize,
    content_data: Vec<u8>,
    complete_atom_data: Vec<u8>,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            current_offset: 0,
            atoms: Vec::new(),
        }
    }

    pub async fn parse<R: AsyncRead + Unpin>(
        &mut self,
        mut reader: R,
    ) -> Result<&[Atom], ParseError> {
        self.current_offset = 0;
        self.atoms.clear();
        self.atoms = self
            .parse_atoms_from_reader(&mut reader, None)
            .try_collect()
            .await?;

        if self.atoms.is_empty() {
            return Err(ParseError {
                kind: ParseErrorKind::InsufficientData,
                location: Some((self.current_offset, 0)),
                source: None,
            });
        }

        Ok(&self.atoms)
    }

    pub fn parse_stream<'a, R: AsyncRead + Unpin + 'a>(
        &'a mut self,
        reader: R,
    ) -> impl Stream<Item = Result<(FourCC, AtomData), ParseError>> + 'a {
        self.current_offset = 0;
        self.parse_atom_stream(reader, None)
    }

    pub fn to_vec(self) -> Vec<Atom> {
        self.atoms
    }

    async fn read_exact<R: AsyncRead + Unpin>(
        &mut self,
        reader: &mut R,
        buf: &mut [u8],
    ) -> Result<(), ParseError> {
        reader.read_exact(buf).await.map_err(|e| ParseError {
            kind: ParseErrorKind::Io,
            location: Some((self.current_offset, buf.len())),
            source: Some(Box::new(e)),
        })?;
        self.current_offset += buf.len();
        Ok(())
    }

    async fn read_data<R: AsyncRead + Unpin>(
        &mut self,
        reader: &mut R,
        size: usize,
    ) -> Result<Vec<u8>, ParseError> {
        let mut data = vec![0u8; size];
        self.read_exact(reader, &mut data).await?;
        Ok(data)
    }

    async fn parse_next_atom<R: AsyncRead + Unpin>(
        &mut self,
        reader: &mut R,
        length_limit: Option<usize>,
        start_offset: usize,
    ) -> Result<Option<ParsedAtom>, ParseError> {
        if length_limit.is_some_and(|limit| self.current_offset - start_offset >= limit) {
            return Ok(None);
        }

        // Try to read the atom header (size and type)
        let mut header = [0u8; 8];
        match reader.read_exact(&mut header).await {
            Ok(()) => {
                self.current_offset += 8;
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // End of stream, which is normal
                return Ok(None);
            }
            Err(e) => {
                return Err(ParseError {
                    kind: ParseErrorKind::Io,
                    location: Some((self.current_offset, 8)),
                    source: Some(Box::new(e)),
                });
            }
        }

        let size_32 = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        let atom_type = [header[4], header[5], header[6], header[7]];

        // Handle extended size (64-bit) format
        let (size, header_size) = if size_32 == 1 {
            // Extended size format - next 8 bytes contain the actual 64-bit size
            let mut extended_size_bytes = [0u8; 8];
            self.read_exact(reader, &mut extended_size_bytes).await?;
            let extended_size = u64::from_be_bytes(extended_size_bytes);
            (extended_size, 16) // 8 bytes basic header + 8 bytes extended size
        } else if size_32 == 0 {
            // Size 0 means "rest of container" or "rest of file"
            if let Some(limit) = length_limit {
                // We're in a container, use remaining bytes
                let bytes_read = self.current_offset - start_offset;
                let remaining = limit.saturating_sub(bytes_read);
                (remaining as u64 + 8, 8) // Include header size
            } else {
                // At top level with size=0 - treat as error for now
                return Err(ParseError {
                    kind: ParseErrorKind::UnsupportedAtom,
                    location: Some((self.current_offset - 8, 8)),
                    source: None,
                });
            }
        } else {
            (size_32 as u64, 8) // Standard 32-bit size
        };

        let content_size = if size > header_size {
            (size as usize).saturating_sub(header_size as usize)
        } else {
            0
        };

        let atom_offset = (self.current_offset - header_size as usize) as u64;

        // Read the atom content
        let content_data = self.read_data(reader, content_size).await?;

        // Create complete atom data (header + content) for atom parsers
        let mut complete_atom_data = Vec::new();
        if header_size == 16 {
            // Extended size format: size(4) + type(4) + extended_size(8)
            complete_atom_data.extend_from_slice(&1u32.to_be_bytes()); // size = 1 indicates extended
            complete_atom_data.extend_from_slice(&atom_type);
            complete_atom_data.extend_from_slice(&size.to_be_bytes());
        } else {
            // Standard format: size(4) + type(4)
            complete_atom_data.extend_from_slice(&(size as u32).to_be_bytes());
            complete_atom_data.extend_from_slice(&atom_type);
        }
        complete_atom_data.extend_from_slice(&content_data);

        Ok(Some(ParsedAtom {
            atom_type,
            size,
            offset: atom_offset,
            header_size: header_size as usize,
            content_data,
            complete_atom_data,
        }))
    }

    fn parse_atom_data(&self, parsed_atom: &ParsedAtom) -> Result<AtomData, ParseError> {
        let atom_type_fourcc = FourCC::from(parsed_atom.atom_type);
        let complete_atom_data = parsed_atom.complete_atom_data.as_slice();

        let atom_data = match &parsed_atom.atom_type {
            FTYP => Some(FileTypeAtom::try_from(complete_atom_data).map(AtomData::from)),
            MVHD => Some(MovieHeaderAtom::try_from(complete_atom_data).map(AtomData::from)),
            ELST => Some(EditListAtom::try_from(complete_atom_data).map(AtomData::from)),
            MDHD => Some(MediaHeaderAtom::try_from(complete_atom_data).map(AtomData::from)),
            HDLR => Some(HandlerReferenceAtom::try_from(complete_atom_data).map(AtomData::from)),
            GMHD => Some(GenericMediaHeaderAtom::try_from(complete_atom_data).map(AtomData::from)),
            SMHD => Some(SoundMediaHeaderAtom::try_from(complete_atom_data).map(AtomData::from)),
            META => Some(MetadataAtom::try_from(complete_atom_data).map(AtomData::from)),
            ILST => Some(ItemListAtom::try_from(complete_atom_data).map(AtomData::from)),
            TKHD => Some(TrackHeaderAtom::try_from(complete_atom_data).map(AtomData::from)),
            STSD => {
                Some(SampleDescriptionTableAtom::try_from(complete_atom_data).map(AtomData::from))
            }
            TREF => Some(TrackReferenceAtom::try_from(complete_atom_data).map(AtomData::from)),
            DREF => Some(DataReferenceAtom::try_from(complete_atom_data).map(AtomData::from)),
            STSZ => Some(SampleSizeAtom::try_from(complete_atom_data).map(AtomData::from)),
            STCO | CO64 => Some(ChunkOffsetAtom::try_from(complete_atom_data).map(AtomData::from)),
            STTS => Some(TimeToSampleAtom::try_from(complete_atom_data).map(AtomData::from)),
            STSC => Some(SampleToChunkAtom::try_from(complete_atom_data).map(AtomData::from)),
            CHPL => Some(ChapterListAtom::try_from(complete_atom_data).map(AtomData::from)),
            SGPD => {
                Some(SampleGroupDescriptionAtom::try_from(complete_atom_data).map(AtomData::from))
            }
            SBGP => Some(SampleToGroupAtom::try_from(complete_atom_data).map(AtomData::from)),
            FREE | SKIP => Some(FreeAtom::try_from(complete_atom_data).map(AtomData::from)),
            _ => None,
        }
        .transpose()
        .map_err(|e| ParseError {
            kind: ParseErrorKind::AtomParsing,
            location: Some((
                parsed_atom.offset as usize,
                parsed_atom.complete_atom_data.len(),
            )),
            source: Some(e.context(atom_type_fourcc).into()),
        })?
        .unwrap_or_else(|| RawData(parsed_atom.content_data.clone()).into());

        Ok(atom_data)
    }

    fn parse_atom_stream<'a, R: AsyncRead + Unpin + 'a>(
        &'a mut self,
        mut reader: R,
        length_limit: Option<usize>,
    ) -> impl Stream<Item = Result<(FourCC, AtomData), ParseError>> + 'a {
        async_stream::stream! {
            let start_offset = self.current_offset;
            let mut current_offset = start_offset;

            while let Some(parsed_atom) = self.parse_next_atom_with_offset(&mut reader, length_limit, current_offset).await? {
                let atom_type_fourcc = FourCC::from(parsed_atom.atom_type);

                // Update offset for next iteration
                current_offset = parsed_atom.offset as usize + parsed_atom.complete_atom_data.len();

                // Skip container atoms in the stream, but process leaf atoms
                if !is_container_atom(&parsed_atom.atom_type) && &parsed_atom.atom_type != META {
                    // Yield non-container atoms
                    yield Ok((atom_type_fourcc, self.parse_atom_data(&parsed_atom)?));
                } else if is_container_atom(&parsed_atom.atom_type) || &parsed_atom.atom_type == META {
                    // For container atoms, recursively parse children and yield their content as they come
                    let (mut cursor, size) = if &parsed_atom.atom_type == META {
                        // Handle META atoms specially, ignoring the META-specific headers for now
                        match self.parse_atom_data(&parsed_atom)? {
                            AtomData::Metadata(meta_atom) => {
                                let size = meta_atom.child_data.len();
                                (Cursor::new(meta_atom.child_data), size)
                            }
                            _ => {
                                unreachable!("META atoms should always contain [AtomData::Metadata]");
                            }
                        }
                    } else {
                        let size = parsed_atom.content_data.len();
                        (Cursor::new(parsed_atom.content_data), size)
                    };
                    let child_stream = self.parse_atoms_from_reader(&mut cursor, Some(size));
                    futures_util::pin_mut!(child_stream);

                    while let Some(child_result) = child_stream.next().await {
                        match child_result {
                            Ok(child_atom) => {
                                if let Some(data) = child_atom.data {
                                    yield Ok((child_atom.atom_type, data));
                                }
                            }
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                }

                // Size can be 0 (meaning "rest of file") so we need to prevent infinite loop.
                if parsed_atom.size == 0 {
                    break;
                }
            }
        }
    }

    async fn parse_next_atom_with_offset<R: AsyncRead + Unpin>(
        &mut self,
        reader: &mut R,
        length_limit: Option<usize>,
        offset: usize,
    ) -> Result<Option<ParsedAtom>, ParseError> {
        let saved_offset = self.current_offset;
        self.current_offset = offset;
        let result = self.parse_next_atom(reader, length_limit, offset).await;
        self.current_offset = saved_offset;
        result
    }

    /// Recursively parse atoms from a reader with an optional length limit.
    fn parse_atoms_from_reader<'a, R: AsyncRead + Unpin + 'a>(
        &'a mut self,
        reader: &'a mut R,
        length_limit: Option<usize>,
    ) -> impl Stream<Item = Result<Atom, ParseError>> + 'a {
        async_stream::stream! {
            let start_offset = self.current_offset;

            while let Some(parsed_atom) = self
                .parse_next_atom(reader, length_limit, start_offset)
                .await?
            {
                let atom_data = self.parse_atom_data(&parsed_atom)?;

                let mut atom = Atom {
                    atom_type: parsed_atom.atom_type.into(),
                    size: parsed_atom.size,
                    offset: parsed_atom.offset,
                    children: Vec::new(),
                    data: Some(atom_data),
                };

                if parsed_atom.atom_type == *META {
                    // MetadataAtom is a special type of container atom
                    if let Some(AtomData::Metadata(MetadataAtom { child_data, .. })) = &atom.data {
                        let mut cursor = child_data.as_slice();
                        let saved_offset = self.current_offset;
                        self.current_offset -= child_data.len();

                        // Collect children from stream
                        let children: Result<Vec<_>, _> = Box::pin(
                            self.parse_atoms_from_reader(&mut cursor, Some(child_data.len()))
                        ).try_collect().await;
                        atom.children = children?;

                        self.current_offset = saved_offset;
                    } else {
                        yield Err(ParseError {
                            kind: ParseErrorKind::AtomParsing,
                            location: Some((
                                parsed_atom.offset as usize,
                                parsed_atom.content_data.len(),
                            )),
                            source: Some(anyhow::anyhow!("Invalid meta atom").into()),
                        });
                        return;
                    }
                } else if is_container_atom(&parsed_atom.atom_type) {
                    // Parse children for container atoms
                    let mut cursor = parsed_atom.content_data.as_slice();
                    let saved_offset = self.current_offset;
                    self.current_offset = parsed_atom.offset as usize + parsed_atom.header_size;

                    // Collect children from stream
                    let children: Result<Vec<_>, _> = Box::pin(
                        self.parse_atoms_from_reader(&mut cursor, Some(parsed_atom.content_data.len()))
                    ).try_collect().await;
                    atom.children = children?;

                    self.current_offset = saved_offset;
                } else if atom.data.is_none() {
                    // Unhandled atom, store raw data
                    atom.data = Some(RawData(parsed_atom.content_data).into());
                }

                yield Ok(atom);

                // Size can be 0 (meaning "rest of file") so we need to prevent infinite loop.
                if parsed_atom.size == 0 {
                    break;
                }
            }
        }
    }
}

pub async fn parse_mp4<R: AsyncRead + Unpin>(reader: R) -> Result<Vec<Atom>, ParseError> {
    let mut parser = Parser::new();
    parser.parse(reader).await?;
    Ok(parser.to_vec())
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
    )
}

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use super::*;

    #[tokio::test]
    async fn test_32bit_size_parsing() {
        // Create a simple FTYP atom with 32-bit size
        let mut data = Vec::new();
        data.extend_from_slice(&20u32.to_be_bytes()); // Size: 20 bytes total
        data.extend_from_slice(b"ftyp"); // Type: ftyp
        data.extend_from_slice(b"mp41"); // Major brand
        data.extend_from_slice(&0u32.to_be_bytes()); // Minor version
        data.extend_from_slice(b"mp41"); // Compatible brand

        let mut parser = Parser::new();
        let result = parser.parse(data.as_slice()).await;

        if let Err(ref e) = result {
            println!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        let atoms = result.unwrap();
        assert_eq!(atoms.len(), 1);
        let ftyp_fourcc = b"ftyp";
        assert_eq!(atoms[0].atom_type, ftyp_fourcc);
        assert_eq!(atoms[0].size, 20);
        assert_eq!(atoms[0].offset, 0);
    }

    #[tokio::test]
    async fn test_64bit_extended_size_parsing() {
        // Create an atom with extended 64-bit size
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_be_bytes()); // Size: 1 (indicates extended size)
        data.extend_from_slice(b"test"); // Type: test
        data.extend_from_slice(&24u64.to_be_bytes()); // Extended size: 24 bytes total
        data.extend_from_slice(b"hello!"); // 6 bytes of content (24 - 16 - 2 padding)
        data.extend_from_slice(&[0u8; 2]); // padding to make exact size

        let mut parser = Parser::new();
        let result = parser.parse(data.as_slice()).await;

        assert!(result.is_ok());
        let atoms = result.unwrap();
        assert_eq!(atoms.len(), 1);
        let test_fourcc = b"test";
        assert_eq!(atoms[0].atom_type, test_fourcc);
        assert_eq!(atoms[0].size, 24);
        assert_eq!(atoms[0].offset, 0);
    }

    #[tokio::test]
    async fn test_multiple_atoms() {
        // Create two atoms back-to-back
        let mut data = Vec::new();

        // First atom
        data.extend_from_slice(&16u32.to_be_bytes()); // Size: 16 bytes
        data.extend_from_slice(b"tes1"); // Type: tes1
        data.extend_from_slice(b"data1234"); // 8 bytes content

        // Second atom
        data.extend_from_slice(&16u32.to_be_bytes()); // Size: 16 bytes
        data.extend_from_slice(b"tes2"); // Type: tes2
        data.extend_from_slice(b"data5678"); // 8 bytes content

        let mut parser = Parser::new();
        let result = parser.parse(data.as_slice()).await;

        assert!(result.is_ok());
        let atoms = result.unwrap();
        assert_eq!(atoms.len(), 2);

        let tes1_fourcc = b"tes1";
        let tes2_fourcc = b"tes2";
        assert_eq!(atoms[0].atom_type, tes1_fourcc);
        assert_eq!(atoms[0].size, 16);
        assert_eq!(atoms[0].offset, 0);

        assert_eq!(atoms[1].atom_type, tes2_fourcc);
        assert_eq!(atoms[1].size, 16);
        assert_eq!(atoms[1].offset, 16);
    }

    #[tokio::test]
    async fn test_container_atom_parsing() {
        // Create a simple container atom (moov) with one child
        let mut data = Vec::new();
        data.extend_from_slice(&24u32.to_be_bytes()); // Size: 24 bytes total
        data.extend_from_slice(b"moov"); // Type: moov (container)

        // Child atom inside moov
        data.extend_from_slice(&16u32.to_be_bytes()); // Size: 16 bytes
        data.extend_from_slice(b"chld"); // Type: chld (4 bytes)
        data.extend_from_slice(b"content!"); // 8 bytes content

        let mut parser = Parser::new();
        let result = parser.parse(data.as_slice()).await;

        assert!(result.is_ok());
        let atoms = result.unwrap();
        assert_eq!(atoms.len(), 1);
        let moov_fourcc = b"moov";
        let chld_fourcc = b"chld";
        assert_eq!(atoms[0].atom_type, moov_fourcc);
        assert_eq!(atoms[0].size, 24);
        assert_eq!(atoms[0].children.len(), 1);
        assert_eq!(atoms[0].children[0].atom_type, chld_fourcc);
        assert_eq!(atoms[0].children[0].size, 16);
    }

    #[tokio::test]
    async fn test_insufficient_data_error() {
        // Create data that's too short for a complete atom header
        let data = vec![0u8; 4]; // Only 4 bytes, need at least 8

        let mut parser = Parser::new();
        let result = parser.parse(data.as_slice()).await;

        assert!(result.is_err());
        // Should not panic or crash
    }

    #[tokio::test]
    async fn test_invalid_size_error() {
        // Create an atom with size smaller than header
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_be_bytes()); // Size: 4 bytes (smaller than 8-byte header)
        data.extend_from_slice(b"test"); // Type: test

        let mut parser = Parser::new();
        let result = parser.parse(data.as_slice()).await;

        // Should handle gracefully - size of 4 means content_size = 0
        assert!(result.is_ok());
        let atoms = result.unwrap();
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].size, 4);
    }

    #[tokio::test]
    async fn test_stream_parsing() {
        use futures_util::pin_mut;
        use futures_util::stream::StreamExt;

        // Create test data with multiple atoms
        let mut data = Vec::new();

        // First atom
        data.extend_from_slice(&16u32.to_be_bytes()); // Size: 16 bytes
        data.extend_from_slice(b"tes1"); // Type: tes1
        data.extend_from_slice(b"content1"); // 8 bytes content

        // Second atom
        data.extend_from_slice(&16u32.to_be_bytes()); // Size: 16 bytes
        data.extend_from_slice(b"tes2"); // Type: tes2
        data.extend_from_slice(b"content2"); // 8 bytes content

        let mut parser = Parser::new();
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        // Collect all stream items
        let mut atoms = Vec::new();
        while let Some(result) = stream.next().await {
            atoms.push(result.unwrap());
        }

        // Should have 2 atoms
        assert_eq!(atoms.len(), 2);

        // Check first atom
        let (fourcc1, _data1) = &atoms[0];
        assert_eq!(fourcc1.deref(), b"tes1");

        // Check second atom
        let (fourcc2, _data2) = &atoms[1];
        assert_eq!(fourcc2.deref(), b"tes2");
    }
}
