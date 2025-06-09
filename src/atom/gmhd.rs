use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::io::{Cursor, Read};

use crate::{atom::util::parse_fixed_size_atom, parser::Parse};

pub const GMHD: &[u8; 4] = b"gmhd";

#[derive(Debug, Clone, PartialEq)]
pub struct GenericMediaHeaderAtom {
    /// Version of the gmhd atom format (0)
    pub version: u8,
    /// Flags for the gmhd atom (usually all zeros)
    pub flags: [u8; 3],
    /// Graphics mode for compositing (usually 0 = copy)
    pub graphics_mode: u16,
    /// RGB color values for graphics mode (each component is 16-bit)
    pub opcolor: [u16; 3], // [red, green, blue]
}

impl Parse for GenericMediaHeaderAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(reader: R) -> Result<Self, anyhow::Error> {
        let (atom_type, data) = parse_fixed_size_atom(reader).await?;
        if atom_type != GMHD {
            return Err(anyhow!("Invalid atom type: {}", atom_type));
        }

        // Parse the data using existing sync function
        let cursor = Cursor::new(data);
        parse_gmhd_data(cursor)
    }
}

fn parse_gmhd_data<R: Read>(mut reader: R) -> Result<GenericMediaHeaderAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version
    if version != 0 {
        return Err(anyhow!("unsupported version {}", version));
    }

    // Read graphics mode (2 bytes)
    let mut graphics_mode_buf = [0u8; 2];
    reader
        .read_exact(&mut graphics_mode_buf)
        .context("read graphics mode")?;
    let graphics_mode = u16::from_be_bytes(graphics_mode_buf);

    // Read opcolor RGB values (6 bytes total, 2 bytes each)
    let mut opcolor = [0u16; 3];
    for (i, color) in opcolor.iter_mut().enumerate() {
        let mut color_buf = [0u8; 2];
        reader
            .read_exact(&mut color_buf)
            .context(format!("read opcolor component {}", i))?;
        *color = u16::from_be_bytes(color_buf);
    }

    Ok(GenericMediaHeaderAtom {
        version,
        flags,
        graphics_mode,
        opcolor,
    })
}
