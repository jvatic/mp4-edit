use crate::atom::Atom;

pub struct AtomIter<'a> {
    pub(crate) iter: Option<std::slice::Iter<'a, Atom>>,
}

impl<'a> AtomIter<'a> {
    pub fn from_atom(atom_opt: Option<&'a Atom>) -> Self {
        Self {
            iter: atom_opt.map(|atom| atom.children.iter()),
        }
    }
}

impl<'a> Iterator for AtomIter<'a> {
    type Item = &'a Atom;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.as_mut().and_then(std::iter::Iterator::next)
    }
}

impl DoubleEndedIterator for AtomIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .as_mut()
            .and_then(std::iter::DoubleEndedIterator::next_back)
    }
}

impl ExactSizeIterator for AtomIter<'_> {
    fn len(&self) -> usize {
        self.iter
            .as_ref()
            .map(ExactSizeIterator::len)
            .unwrap_or_default()
    }
}

pub struct AtomIterMut<'a> {
    pub(crate) children: &'a mut [Atom],
    pub(crate) index: usize,
}

impl<'a> AtomIterMut<'a> {
    pub fn from_atom(atom: &'a mut Atom) -> Self {
        Self {
            children: &mut atom.children,
            index: 0,
        }
    }
}

impl<'a> Iterator for AtomIterMut<'a> {
    type Item = &'a mut Atom;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.children.len() {
            return None;
        }

        let children = std::mem::take(&mut self.children);
        let (current, rest) = children.split_at_mut(self.index + 1);
        self.children = rest;
        let old_index = self.index;
        self.index = 0;

        current.get_mut(old_index)
    }
}
