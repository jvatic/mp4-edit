use derive_more::Deref;
use std::fmt;

#[derive(Clone, Copy, Deref, PartialEq, Eq)]
pub struct FourCC(pub(crate) [u8; 4]);

impl FourCC {
    pub fn into_bytes(self) -> [u8; 4] {
        self.0
    }
}

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
        write!(
            f,
            "{}",
            std::str::from_utf8(&self.0)
                .map(|s| s.to_owned())
                .unwrap_or_else(|_| convert_mac_roman_to_utf8(&self.0))
        )
    }
}

impl fmt::Debug for FourCC {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FourCC({})", self)
    }
}

fn convert_mac_roman_to_utf8(bytes: &[u8]) -> String {
    let mut result = String::new();
    for &byte in bytes {
        match byte {
            0xA9 => result.push('©'), // Copyright symbol
            0xAE => result.push('®'), // Registered trademark symbol
            0x99 => result.push('™'), // Trademark symbol
            // For other bytes, treat as ASCII if valid, otherwise use replacement char
            b if b.is_ascii() => result.push(b as char),
            _ => result.push('�'),
        }
    }
    result
}
