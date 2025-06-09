use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::{
    atom::util::{parser::parse_fixed_size_atom, FourCC},
    parser::Parse,
};

pub const TREF: &[u8; 4] = b"tref";

/// Track Reference Types - Common reference types found in MP4 files
pub mod reference_types {
    /// Hint track reference - points to the media track that this hint track describes
    pub const HINT: &[u8; 4] = b"hint";
    /// Chapter track reference - points to the track that contains chapter information
    pub const CHAP: &[u8; 4] = b"chap";
    /// Subtitle track reference - points to the subtitle track
    pub const SUBT: &[u8; 4] = b"subt";
    /// Audio description track reference
    pub const ADSC: &[u8; 4] = b"adsc";
    /// Forced subtitle track reference
    pub const FORC: &[u8; 4] = b"forc";
    /// Karaoke track reference
    pub const KARO: &[u8; 4] = b"karo";
    /// Metadata track reference
    pub const META: &[u8; 4] = b"meta";
    /// Auxiliary video track reference
    pub const AUXV: &[u8; 4] = b"auxv";
    /// Closed caption track reference
    pub const CLCP: &[u8; 4] = b"clcp";
}

/// A single track reference entry containing the reference type and target track IDs
#[derive(Debug, Clone)]
pub struct TrackReference {
    /// The type of reference (e.g., "hint", "chap", "subt")
    pub reference_type: FourCC,
    /// List of track IDs that this reference points to
    pub track_ids: Vec<u32>,
}

impl TrackReference {
    /// Check if this reference is of a specific type
    pub fn is_type(&self, ref_type: &[u8; 4]) -> bool {
        self.reference_type == ref_type
    }
}

impl fmt::Display for TrackReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> [{}]",
            self.reference_type,
            self.track_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Track Reference Atom (tref) - ISO/IEC 14496-12
/// Contains references from this track to other tracks
#[derive(Debug, Clone)]
pub struct TrackReferenceAtom {
    /// List of track references
    pub references: Vec<TrackReference>,
}

impl Parse for TrackReferenceAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(reader: R) -> Result<Self, anyhow::Error> {
        let (atom_type, data) = parse_fixed_size_atom(reader).await?;
        if atom_type != TREF {
            return Err(anyhow!(
                "Invalid atom type: expected tref, got {}",
                atom_type
            ));
        }

        // Parse the data using existing sync function
        let cursor = Cursor::new(data);
        parse_tref_data(cursor)
    }
}

impl fmt::Display for TrackReferenceAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.references.is_empty() {
            write!(f, "TrackReferenceAtom {{ no references }}")
        } else {
            write!(f, "TrackReferenceAtom {{")?;
            for (i, reference) in self.references.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, " {}", reference)?;
            }
            write!(f, " }}")
        }
    }
}

fn parse_tref_data<R: Read>(mut reader: R) -> Result<TrackReferenceAtom, anyhow::Error> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data).context("reading tref data")?;

    let mut references = Vec::new();
    let mut offset = 0;

    // Parse child atoms (reference type atoms)
    while offset < data.len() {
        if offset + 8 > data.len() {
            return Err(anyhow!(
                "Incomplete reference atom header at offset {}",
                offset
            ));
        }

        // Read size and type of reference atom
        let size = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;

        let reference_type = FourCC([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);

        if size < 8 {
            return Err(anyhow!("Invalid reference atom size: {}", size));
        }

        if offset + size > data.len() {
            return Err(anyhow!(
                "Reference atom extends beyond tref data: offset={}, size={}, data_len={}",
                offset,
                size,
                data.len()
            ));
        }

        // Parse track IDs (remaining data after 8-byte header, each ID is 4 bytes)
        let ids_data = &data[offset + 8..offset + size];
        if ids_data.len() % 4 != 0 {
            return Err(anyhow!(
                "Invalid track IDs data length: {} bytes (must be multiple of 4)",
                ids_data.len()
            ));
        }

        let mut track_ids = Vec::new();
        for chunk in ids_data.chunks_exact(4) {
            let track_id = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            track_ids.push(track_id);
        }

        references.push(TrackReference {
            reference_type,
            track_ids,
        });

        offset += size;
    }

    Ok(TrackReferenceAtom { references })
}
