pub mod atom;
pub mod parser;

pub use atom::{Atom, AtomData};
pub use parser::{parse_mp4, ParseError};
