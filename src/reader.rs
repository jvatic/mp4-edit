use futures_io::{AsyncRead, AsyncSeek};
use futures_util::io::{AsyncReadExt, AsyncSeekExt};
use std::io::SeekFrom;
use std::marker::PhantomData;

use crate::parser::ParseErrorKind;
use crate::ParseError;

mod sealed {
    pub trait Sealed {}
}

pub struct Seekable;
pub struct NonSeekable;

impl sealed::Sealed for Seekable {}
impl sealed::Sealed for NonSeekable {}

pub trait ReadCapability: sealed::Sealed {}

impl ReadCapability for NonSeekable {}

impl ReadCapability for Seekable {}

pub struct Mp4Reader<R, C: ReadCapability> {
    reader: R,
    pub(crate) current_offset: usize,
    peek_buffer: Vec<u8>,
    _capability: PhantomData<C>,
}

impl<R: AsyncRead + Unpin + Send> Mp4Reader<R, NonSeekable> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            current_offset: 0,
            peek_buffer: Vec::new(),
            _capability: PhantomData,
        }
    }
}

impl<R: AsyncRead + Unpin + Send, C: ReadCapability> Mp4Reader<R, C> {
    pub(crate) async fn peek_exact(&mut self, buf: &mut [u8]) -> Result<(), ParseError> {
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

    pub(crate) async fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), ParseError> {
        self.peek_exact(buf).await?;
        self.peek_buffer.drain(..buf.len());
        self.current_offset += buf.len();
        Ok(())
    }

    pub(crate) async fn read_data(&mut self, size: usize) -> Result<Vec<u8>, ParseError> {
        let mut data = vec![0u8; size];
        self.read_exact(&mut data).await?;
        Ok(data)
    }
}

impl<R: AsyncRead + AsyncSeek + Unpin + Send> Mp4Reader<R, Seekable> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            current_offset: 0,
            peek_buffer: Vec::new(),
            _capability: PhantomData,
        }
    }

    pub(crate) async fn seek(&mut self, pos: SeekFrom) -> Result<(), ParseError> {
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
}
