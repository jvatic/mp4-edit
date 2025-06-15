pub mod atom;
pub mod parser;

pub use atom::{Atom, AtomData};
pub use parser::{ParseError, ParseMetadataEvent, Parser};
