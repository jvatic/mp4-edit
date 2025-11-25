use std::{ops::Range, vec};

#[derive(Debug)]
pub struct RangeCollection<T>(Vec<Option<Range<T>>>);

impl<T> RangeCollection<T>
where
    T: Eq + PartialOrd,
{
    pub fn new() -> Self {
        Self(vec![None])
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        vec.push(None);
        Self(vec)
    }

    /// Inserts the given range, either exending an existing entry if it's contiguous with it, or adding a new entry
    /// otherwise.
    ///
    /// # Panics
    ///
    /// This method will panic if item overlaps with an existing range.
    pub fn insert(&mut self, item: Range<T>) {
        let mut new_entry = None;
        match self.0.last_mut() {
            Some(range) => match range {
                Some(range) => {
                    if range.end == item.start {
                        range.end = item.end;
                    } else {
                        debug_assert!(
                            !range.contains(&item.start),
                            "invariant: item overlaps with an existing range"
                        );
                        new_entry = Some(item)
                    }
                }
                None => *range = Some(item),
            },
            None => unreachable!("invariant: internal Vec is never empty"),
        }
        if let Some(new_entry) = new_entry {
            self.0.push(Some(new_entry));
        }
    }

    pub fn into_iter(self) -> impl Iterator<Item = Range<T>> {
        self.0.into_iter().flatten()
    }
}
