use std::{
    fmt::{self, UpperHex},
    marker::PhantomData,
};

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

pub struct DebugList<Item, List>(List, usize, PhantomData<Item>);

impl<Item, List> DebugList<Item, List> {
    pub fn new(list: List, max_len: usize) -> Self {
        Self(list, max_len, PhantomData)
    }
}

impl<Item, List> fmt::Debug for DebugList<Item, List>
where
    List: Iterator<Item = Item> + ExactSizeIterator + Clone,
    Item: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (list, max_len) = (self.0.clone(), self.1);
        let (lower_bound, upper_bound) = list.size_hint();
        let size = upper_bound.unwrap_or(lower_bound);
        if size <= max_len {
            f.debug_list().entries(list).finish()
        } else {
            f.debug_list()
                .entries(list.take(max_len))
                .entry(&DebugEllipsis(Some(size - max_len)))
                .finish()
        }
    }
}

pub struct DebugUpperHex<T: UpperHex>(pub T);

impl<T: UpperHex> fmt::Debug for DebugUpperHex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}
