use derive_more::Display;
use futures_io::AsyncRead;
use futures_util::io::AsyncReadExt;
use std::pin::Pin;
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
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    pub fn new() -> Self {
        Parser { current_offset: 0 }
    }

    pub async fn parse<R: AsyncRead + Unpin>(
        &mut self,
        mut reader: R,
    ) -> Result<Vec<Atom>, ParseError> {
        self.current_offset = 0;
        let atoms = self.parse_atoms_from_reader(&mut reader, None).await?;

        if atoms.is_empty() {
            return Err(ParseError {
                kind: ParseErrorKind::InsufficientData,
                location: Some((self.current_offset, 0)),
                source: None,
            });
        }

        Ok(atoms)
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

    /// Recursively parse atoms from a reader with an optional length limit.
    fn parse_atoms_from_reader<'a, R: AsyncRead + Unpin>(
        &'a mut self,
        reader: &'a mut R,
        length_limit: Option<usize>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<Atom>, ParseError>> + 'a>> {
        Box::pin(async move {
            let mut atoms = Vec::new();
            let start_offset = self.current_offset;

            while length_limit.is_none_or(|limit| self.current_offset - start_offset < limit) {
                // Try to read the atom header (size and type)
                let mut header = [0u8; 8];
                match reader.read_exact(&mut header).await {
                    Ok(()) => {
                        self.current_offset += 8;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        // End of stream, which is normal
                        break;
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

                let atom_offset = (self.current_offset - header_size as usize) as u64; // Store the offset where this atom started

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

                let atom_data = {
                    // For leaf atoms, parse the data using complete atom data
                    let atom_type_fourcc = FourCC::from(atom_type);
                    let complete_atom_data = complete_atom_data.as_slice();
                    match &atom_type {
                        FTYP => {
                            Some(FileTypeAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        MVHD => {
                            Some(MovieHeaderAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        ELST => {
                            Some(EditListAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        MDHD => {
                            Some(MediaHeaderAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        HDLR => Some(
                            HandlerReferenceAtom::try_from(complete_atom_data).map(AtomData::from),
                        ),
                        GMHD => Some(
                            GenericMediaHeaderAtom::try_from(complete_atom_data)
                                .map(AtomData::from),
                        ),
                        SMHD => Some(
                            SoundMediaHeaderAtom::try_from(complete_atom_data).map(AtomData::from),
                        ),
                        META => {
                            Some(MetadataAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        ILST => {
                            Some(ItemListAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        TKHD => {
                            Some(TrackHeaderAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        STSD => Some(
                            SampleDescriptionTableAtom::try_from(complete_atom_data)
                                .map(AtomData::from),
                        ),
                        TREF => Some(
                            TrackReferenceAtom::try_from(complete_atom_data).map(AtomData::from),
                        ),
                        DREF => Some(
                            DataReferenceAtom::try_from(complete_atom_data).map(AtomData::from),
                        ),
                        STSZ => {
                            Some(SampleSizeAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        STCO | CO64 => {
                            Some(ChunkOffsetAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        STTS => {
                            Some(TimeToSampleAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        STSC => Some(
                            SampleToChunkAtom::try_from(complete_atom_data).map(AtomData::from),
                        ),
                        CHPL => {
                            Some(ChapterListAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        SGPD => Some(
                            SampleGroupDescriptionAtom::try_from(complete_atom_data)
                                .map(AtomData::from),
                        ),
                        SBGP => Some(
                            SampleToGroupAtom::try_from(complete_atom_data).map(AtomData::from),
                        ),
                        FREE | SKIP => {
                            Some(FreeAtom::try_from(complete_atom_data).map(AtomData::from))
                        }
                        _ => None,
                    }
                    .map(|opt_result| {
                        opt_result.map_err(|e| ParseError {
                            kind: ParseErrorKind::AtomParsing,
                            location: Some((atom_offset as usize, complete_atom_data.len())),
                            source: Some(e.context(atom_type_fourcc).into()),
                        })
                    })
                    .transpose()?
                };

                let mut atom = Atom {
                    atom_type: atom_type.into(),
                    size,
                    offset: atom_offset,
                    children: Vec::new(),
                    data: atom_data,
                };

                if &atom_type == META {
                    // MetadataAtom is a special type of container atom
                    if let Some(AtomData::Metadata(MetadataAtom { child_data, .. })) = &atom.data {
                        let mut cursor = child_data.as_slice();
                        let saved_offset = self.current_offset;
                        self.current_offset -= child_data.len();
                        atom.children = self
                            .parse_atoms_from_reader(&mut cursor, Some(child_data.len()))
                            .await?;
                        self.current_offset = saved_offset;
                    } else {
                        return Err(ParseError {
                            kind: ParseErrorKind::AtomParsing,
                            location: Some((atom_offset as usize, content_size)),
                            source: Some(anyhow::anyhow!("Invalid meta atom").into()),
                        });
                    }
                } else if is_container_atom(&atom_type) {
                    // Parse children for container atoms
                    let mut cursor = content_data.as_slice();
                    let saved_offset = self.current_offset;
                    self.current_offset = atom_offset as usize + header_size as usize;
                    atom.children = self
                        .parse_atoms_from_reader(&mut cursor, Some(content_size))
                        .await?;
                    self.current_offset = saved_offset;
                } else if atom.data.is_none() {
                    // Unhandled atom, store raw data
                    atom.data = Some(RawData(content_data).into());
                }

                atoms.push(atom);

                // Size can be 0 (meaning "rest of file") so we need to prevent infinite loop.
                if size == 0 {
                    break;
                }
            }
            Ok(atoms)
        })
    }
}

pub async fn parse_mp4<R: AsyncRead + Unpin>(reader: R) -> Result<Vec<Atom>, ParseError> {
    let mut parser = Parser::new();
    parser.parse(reader).await
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
}
