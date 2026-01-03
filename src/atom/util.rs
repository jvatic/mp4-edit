mod debug;
pub mod parser;
pub mod serializer;
mod time;

use std::fmt;

pub use debug::*;
pub use time::*;

#[derive(Clone, Default)]
pub struct ColorRgb {
    pub red: u16,
    pub green: u16,
    pub blue: u16,
}

impl fmt::Debug for ColorRgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColorRgb")
            .field("red", &DebugUpperHex(self.red))
            .field("green", &DebugUpperHex(self.green))
            .field("blue", &DebugUpperHex(self.blue))
            .finish()
    }
}
