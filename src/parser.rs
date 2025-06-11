use derive_more::Display;
use futures_io::AsyncRead;
use futures_util::io::{AsyncReadExt, Cursor};
use futures_util::stream::{Stream, StreamExt};
use std::future::Future;
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

/// Async trait for parsing atoms from an AsyncRead stream
pub trait Parse: Sized {
    fn parse<R: AsyncRead + Unpin + Send>(
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

pub enum ParseEvent {
    EnterContainer(Atom),
    Leaf(Atom),
    ExitContainer,
}

pub struct Parser {
    current_offset: usize,
}

struct ParsedAtom {
    atom_type: [u8; 4],
    size: u64,
    offset: u64,
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
        Parser { current_offset: 0 }
    }

    pub fn parse_stream<'a, R: AsyncRead + Unpin + 'a>(
        &'a mut self,
        reader: R,
    ) -> impl Stream<Item = Result<ParseEvent, ParseError>> + 'a {
        self.current_offset = 0;
        self.parse_atom_stream(reader, None)
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

        let atom_offset = self.current_offset as u64;

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
            (size - header_size) as usize
        } else {
            0
        };

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
            content_data,
            complete_atom_data,
        }))
    }

    async fn parse_atom_data(&self, parsed_atom: &ParsedAtom) -> Result<AtomData, ParseError> {
        let atom_type_fourcc = FourCC::from(parsed_atom.atom_type);
        let complete_atom_data = parsed_atom.complete_atom_data.as_slice();

        let cursor = Cursor::new(complete_atom_data.to_vec());
        let atom_data = match &parsed_atom.atom_type {
            FTYP => Some(FileTypeAtom::parse(cursor).await.map(AtomData::from)),
            MVHD => Some(MovieHeaderAtom::parse(cursor).await.map(AtomData::from)),
            MDHD => Some(MediaHeaderAtom::parse(cursor).await.map(AtomData::from)),
            ELST => Some(EditListAtom::parse(cursor).await.map(AtomData::from)),
            HDLR => Some(
                HandlerReferenceAtom::parse(cursor)
                    .await
                    .map(AtomData::from),
            ),
            GMHD => Some(
                GenericMediaHeaderAtom::parse(cursor)
                    .await
                    .map(AtomData::from),
            ),
            SMHD => Some(
                SoundMediaHeaderAtom::parse(cursor)
                    .await
                    .map(AtomData::from),
            ),
            META => Some(MetadataAtom::parse(cursor).await.map(AtomData::from)),
            ILST => Some(ItemListAtom::parse(cursor).await.map(AtomData::from)),
            TKHD => Some(TrackHeaderAtom::parse(cursor).await.map(AtomData::from)),
            STSD => Some(
                SampleDescriptionTableAtom::parse(cursor)
                    .await
                    .map(AtomData::from),
            ),
            TREF => Some(TrackReferenceAtom::parse(cursor).await.map(AtomData::from)),
            DREF => Some(DataReferenceAtom::parse(cursor).await.map(AtomData::from)),
            STSZ => Some(SampleSizeAtom::parse(cursor).await.map(AtomData::from)),
            STCO | CO64 => Some(ChunkOffsetAtom::parse(cursor).await.map(AtomData::from)),
            STTS => Some(TimeToSampleAtom::parse(cursor).await.map(AtomData::from)),
            STSC => Some(SampleToChunkAtom::parse(cursor).await.map(AtomData::from)),
            CHPL => Some(ChapterListAtom::parse(cursor).await.map(AtomData::from)),
            SGPD => Some(
                SampleGroupDescriptionAtom::parse(cursor)
                    .await
                    .map(AtomData::from),
            ),
            SBGP => Some(SampleToGroupAtom::parse(cursor).await.map(AtomData::from)),
            FREE | SKIP => Some(FreeAtom::parse(cursor).await.map(AtomData::from)),
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
    ) -> impl Stream<Item = Result<ParseEvent, ParseError>> + 'a {
        async_stream::stream! {
            let start_offset = self.current_offset;
            let mut current_offset = start_offset;

            while let Some(parsed_atom) = self.parse_next_atom_with_offset(&mut reader, length_limit, current_offset).await? {
                let atom_type_fourcc = FourCC::from(parsed_atom.atom_type);

                // Update offset for next iteration
                current_offset = parsed_atom.offset as usize + parsed_atom.complete_atom_data.len();

                if !is_container_atom(&parsed_atom.atom_type) && &parsed_atom.atom_type != META {
                    // Yield leaf atoms
                    let atom_data = self.parse_atom_data(&parsed_atom).await?;
                    let atom = Atom {
                        atom_type: atom_type_fourcc,
                        size: parsed_atom.size,
                        offset: parsed_atom.offset,
                        data: Some(atom_data),
                    };
                    yield Ok(ParseEvent::Leaf(atom));
                } else if is_container_atom(&parsed_atom.atom_type) || &parsed_atom.atom_type == META {
                    // For container atoms, emit EnterContainer, then recursively emit children, then ExitContainer
                    let container_atom = Atom {
                        atom_type: atom_type_fourcc,
                        size: parsed_atom.size,
                        offset: parsed_atom.offset,
                        data: None,
                    };
                    yield Ok(ParseEvent::EnterContainer(container_atom));

                    let (cursor, size) = if &parsed_atom.atom_type == META {
                        // Handle META atoms specially, ignoring the META-specific headers for now
                        match self.parse_atom_data(&parsed_atom).await? {
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

                    // Recursively parse children and emit their events
                    let mut child_stream = Box::pin(self.parse_atom_stream(cursor, Some(size)));

                    while let Some(child_event) = child_stream.as_mut().next().await {
                        match child_event {
                            Ok(event) => yield Ok(event),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }

                    yield Ok(ParseEvent::ExitContainer);
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
    use futures_util::pin_mut;
    use futures_util::stream::StreamExt;
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
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        let mut atoms = Vec::new();
        while let Some(result) = stream.next().await {
            match result.unwrap() {
                ParseEvent::Leaf(atom) => atoms.push(atom),
                _ => {}
            }
        }

        assert_eq!(atoms.len(), 1);
        let ftyp_fourcc = b"ftyp";
        assert_eq!(atoms[0].atom_type.deref(), ftyp_fourcc);
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
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        let mut atoms = Vec::new();
        while let Some(result) = stream.next().await {
            match result.unwrap() {
                ParseEvent::Leaf(atom) => atoms.push(atom),
                _ => {}
            }
        }

        assert_eq!(atoms.len(), 1);
        let test_fourcc = b"test";
        assert_eq!(atoms[0].atom_type.deref(), test_fourcc);
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
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        let mut atoms = Vec::new();
        while let Some(result) = stream.next().await {
            match result.unwrap() {
                ParseEvent::Leaf(atom) => atoms.push(atom),
                _ => {}
            }
        }

        assert_eq!(atoms.len(), 2);

        let tes1_fourcc = b"tes1";
        let tes2_fourcc = b"tes2";
        assert_eq!(atoms[0].atom_type.deref(), tes1_fourcc);
        assert_eq!(atoms[0].size, 16);
        assert_eq!(atoms[0].offset, 0);

        assert_eq!(atoms[1].atom_type.deref(), tes2_fourcc);
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
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        let mut events = Vec::new();
        while let Some(result) = stream.next().await {
            events.push(result.unwrap());
        }

        // Should have: EnterContainer, Leaf, ExitContainer
        assert_eq!(events.len(), 3);

        let moov_fourcc = b"moov";
        let chld_fourcc = b"chld";

        match &events[0] {
            ParseEvent::EnterContainer(atom) => {
                assert_eq!(atom.atom_type.deref(), moov_fourcc);
                assert_eq!(atom.size, 24);
            }
            _ => panic!("Expected EnterContainer event"),
        }

        match &events[1] {
            ParseEvent::Leaf(atom) => {
                assert_eq!(atom.atom_type.deref(), chld_fourcc);
                assert_eq!(atom.size, 16);
            }
            _ => panic!("Expected Leaf event"),
        }

        match &events[2] {
            ParseEvent::ExitContainer => {}
            _ => panic!("Expected ExitContainer event"),
        }
    }

    #[tokio::test]
    async fn test_insufficient_data_error() {
        // Create data that's too short for a complete atom header
        let data = vec![0u8; 4]; // Only 4 bytes, need at least 8

        let mut parser = Parser::new();
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        let mut event_count = 0;
        while let Some(result) = stream.next().await {
            result.unwrap(); // Should not error
            event_count += 1;
        }

        // Should produce no events when there's insufficient data
        assert_eq!(event_count, 0);
        // Should not panic or crash
    }

    #[tokio::test]
    async fn test_invalid_size_error() {
        // Create an atom with size smaller than header
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_be_bytes()); // Size: 4 bytes (smaller than 8-byte header)
        data.extend_from_slice(b"test"); // Type: test

        let mut parser = Parser::new();
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        let mut atoms = Vec::new();
        while let Some(result) = stream.next().await {
            match result.unwrap() {
                ParseEvent::Leaf(atom) => atoms.push(atom),
                _ => {}
            }
        }

        // Should handle gracefully - size of 4 means content_size = 0
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].size, 4);
    }

    #[tokio::test]
    async fn test_stream_parsing() {
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
        let mut events = Vec::new();
        while let Some(result) = stream.next().await {
            events.push(result.unwrap());
        }

        // Should have 2 leaf events
        assert_eq!(events.len(), 2);

        // Check first atom
        if let ParseEvent::Leaf(atom1) = &events[0] {
            assert_eq!(atom1.atom_type.deref(), b"tes1");
        } else {
            panic!("Expected Leaf event");
        }

        // Check second atom
        if let ParseEvent::Leaf(atom2) = &events[1] {
            assert_eq!(atom2.atom_type.deref(), b"tes2");
        } else {
            panic!("Expected Leaf event");
        }
    }

    #[tokio::test]
    async fn test_container_event_parsing() {
        // Create test data with a container atom containing child atoms
        let mut data = Vec::new();

        // Container atom "moov" with two child atoms
        let child1_data = b"child1data";
        let child2_data = b"child2data";
        let child1_size = 4 + 4 + child1_data.len(); // header + type + data
        let child2_size = 4 + 4 + child2_data.len(); // header + type + data
        let container_content_size = child1_size + child2_size;
        let container_size = 4 + 4 + container_content_size; // header + type + content

        // Container atom header
        data.extend_from_slice(&(container_size as u32).to_be_bytes()); // Size
        data.extend_from_slice(b"moov"); // Type: moov (container)

        // First child atom
        data.extend_from_slice(&(child1_size as u32).to_be_bytes()); // Size
        data.extend_from_slice(b"chd1"); // Type: chd1
        data.extend_from_slice(child1_data); // Content

        // Second child atom
        data.extend_from_slice(&(child2_size as u32).to_be_bytes()); // Size
        data.extend_from_slice(b"chd2"); // Type: chd2
        data.extend_from_slice(child2_data); // Content

        let mut parser = Parser::new();
        let stream = parser.parse_stream(data.as_slice());
        pin_mut!(stream);

        // Collect all stream events
        let mut events = Vec::new();
        while let Some(result) = stream.next().await {
            events.push(result.unwrap());
        }

        // Should have: EnterContainer, Leaf, Leaf, ExitContainer
        assert_eq!(events.len(), 4);

        // Check event sequence
        match &events[0] {
            ParseEvent::EnterContainer(atom) => {
                assert_eq!(atom.atom_type.deref(), b"moov");
            }
            _ => panic!("Expected EnterContainer event"),
        }

        match &events[1] {
            ParseEvent::Leaf(atom) => {
                assert_eq!(atom.atom_type.deref(), b"chd1");
            }
            _ => panic!("Expected first Leaf event"),
        }

        match &events[2] {
            ParseEvent::Leaf(atom) => {
                assert_eq!(atom.atom_type.deref(), b"chd2");
            }
            _ => panic!("Expected second Leaf event"),
        }

        match &events[3] {
            ParseEvent::ExitContainer => {}
            _ => panic!("Expected ExitContainer event"),
        }
    }
}
