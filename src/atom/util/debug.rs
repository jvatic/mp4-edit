use std::fmt;

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
