use std::fmt::{self, UpperHex};

pub struct DebugEllipsis(pub Option<usize>);

impl fmt::Debug for DebugEllipsis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("...")?;
        if let Some(size) = self.0 {
            write!(f, "({size})")?;
        }
        Ok(())
    }
}

pub struct DebugUpperHex<T: UpperHex>(pub T);

impl<T: UpperHex> fmt::Debug for DebugUpperHex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:08X}", self.0)
    }
}
