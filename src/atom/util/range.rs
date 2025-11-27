use std::{fmt, ops::Range, vec};

#[derive(Debug)]
pub struct RangeCollection<T>(Vec<Option<Range<T>>>);

impl<T> RangeCollection<T>
where
    T: Eq + PartialOrd + fmt::Debug,
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
    /// Attempting to add the same range more than once is a no-op.
    ///
    /// # Panics
    ///
    /// This method will panic if item overlaps with an existing range.
    pub fn insert(&mut self, item: Range<T>) {
        let mut new_entry = None;
        let len = self.0.len();
        match self.0.last_mut() {
            Some(range) => match range {
                Some(range) => {
                    if range.start == item.start && range.end == item.end {
                        return;
                    }

                    if range.end == item.start {
                        range.end = item.end;
                    } else if range.start == item.end {
                        range.start = item.start;
                    } else if item.start < range.start && len > 1 {
                        // item belongs before this position
                        self.insert_before(len - 1, item);
                    } else {
                        debug_assert!(
                            !range.contains(&item.start),
                            "invariant: item overlaps with an existing range: item={item:?} range={range:?}"
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

    /// Inserts the given range before the given index, backpeddling until a suitable position is found
    fn insert_before(&mut self, before_index: usize, item: Range<T>) {
        if before_index == 0 {
            // we've gone as far back as we can, so make this the first range in the collection
            self.0.insert(before_index, Some(item));
            return;
        }

        match self.0.get_mut(before_index - 1) {
            Some(range) => match range {
                Some(range) => {
                    if range.start == item.start && range.end == item.end {
                        return;
                    }

                    if range.end == item.start {
                        range.end = item.end;
                    } else if range.start == item.end {
                        range.start = item.start;
                    } else {
                        self.insert_before(before_index.saturating_sub(2), item);
                        return;
                    }
                }
                None => unreachable!("invariant: only the last item in the Vec will ever be None"),
            },
            None => unreachable!("invariant: index must be in Vec"),
        }

        unreachable!("invariant: item must fit somewhere in the collection");
    }

    pub fn into_iter(self) -> impl Iterator<Item = Range<T>> {
        self.0.into_iter().flatten()
    }
}
