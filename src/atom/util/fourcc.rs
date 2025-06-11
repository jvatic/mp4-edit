use derive_more::Deref;
use std::fmt;

#[derive(Clone, Copy, Deref, PartialEq, Eq)]
pub struct FourCC(pub(crate) [u8; 4]);

impl From<[u8; 4]> for FourCC {
    fn from(value: [u8; 4]) -> Self {
        FourCC(value)
    }
}

impl PartialEq<&[u8; 4]> for FourCC {
    fn eq(&self, other: &&[u8; 4]) -> bool {
        &self.0 == *other
    }
}

impl fmt::Display for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", std::str::from_utf8(&self.0).unwrap_or("????"))
    }
}

impl fmt::Debug for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FourCC({})", self)
    }
}
