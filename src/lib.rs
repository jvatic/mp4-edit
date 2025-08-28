/*!
 * This crate provides tools for losslessly editing mp4 files.
 */

pub mod atom;
pub mod chapter_track_builder;
pub mod chunk_offset_builder;
pub mod parser;
pub mod writer;

pub use atom::{Atom, AtomData, FourCC};
pub use parser::{ParseError, Parser};
pub use writer::Mp4Writer;
