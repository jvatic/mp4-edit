pub mod atom;
pub mod chunk_offset_builder;
pub mod parser;
pub mod writer;

pub use atom::{util::FourCC, Atom, AtomData};
pub use parser::{ParseError, Parser};
pub use writer::Mp4Writer;
