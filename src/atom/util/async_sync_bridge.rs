use futures_io::AsyncRead;
use futures_util::AsyncReadExt;

use crate::{parser::ParseErrorKind, ParseError};

pub async fn read_to_end<R: AsyncRead + Unpin>(mut reader: R) -> Result<Vec<u8>, ParseError> {
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .await
        .map_err(|err| ParseError {
            kind: ParseErrorKind::Io,
            location: None,
            source: Some(Box::new(err)),
        })?;
    Ok(buffer)
}

pub async fn async_to_sync_read<R: AsyncRead + Unpin>(
    mut reader: R,
) -> Result<std::io::Cursor<Vec<u8>>, ParseError> {
    let mut buffer = Vec::new();
    reader
        .read_to_end(&mut buffer)
        .await
        .map_err(|err| ParseError {
            kind: ParseErrorKind::Io,
            location: None,
            source: Some(Box::new(err)),
        })?;
    Ok(std::io::Cursor::new(buffer))
}
