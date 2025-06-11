use futures_io::AsyncRead;
use futures_util::AsyncReadExt;

pub async fn async_to_sync_read<R: AsyncRead + Unpin>(
    mut reader: R,
) -> std::io::Result<std::io::Cursor<Vec<u8>>> {
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).await?;
    Ok(std::io::Cursor::new(buffer))
}
