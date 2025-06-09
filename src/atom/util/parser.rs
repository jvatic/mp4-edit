use crate::atom::util::fourcc::FourCC;
use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use futures_util::AsyncReadExt;

pub async fn parse_atom_header<R: AsyncRead + Unpin>(
    mut reader: R,
) -> Result<(FourCC, u64, u64), anyhow::Error> {
    // Read atom header (8 bytes minimum)
    let mut header = [0u8; 8];
    reader
        .read_exact(&mut header)
        .await
        .context("read header")?;

    let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
    let atom_type: [u8; 4] = header[4..8].try_into()?;

    // Handle extended size (64-bit) if needed
    let (header_size, data_size) = if size == 1 {
        // Extended size format
        let mut extended_size = [0u8; 8];
        reader
            .read_exact(&mut extended_size)
            .await
            .context("read extended size")?;
        let full_size = u64::from_be_bytes(extended_size);
        if full_size < 16 {
            return Err(anyhow!("Invalid atom size"));
        }
        (16u64, full_size - 16)
    } else if size == 0 {
        // Size extends to end of file
        return Err(anyhow!("Invalid atom size"));
    } else {
        if size < 8 {
            return Err(anyhow!("Invalid atom size"));
        }
        (8u64, size - 8)
    };

    Ok((FourCC(atom_type), header_size, data_size))
}

pub async fn parse_fixed_size_atom<R: AsyncRead + Unpin>(
    mut reader: R,
) -> Result<(FourCC, Vec<u8>), anyhow::Error> {
    let (atom_type, _header_size, data_size) = parse_atom_header(&mut reader).await?;

    // Read the remaining atom data
    let mut data = vec![0u8; data_size as usize];
    reader
        .read_exact(&mut data)
        .await
        .context(format!("read atom data (size={data_size})"))?;

    Ok((atom_type, data))
}
