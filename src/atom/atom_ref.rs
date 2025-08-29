/*!
* [`AtomRef`] and [`AtomRefMut`] provide utilities for working with shared and mutable references to [`Atom`]s that have children.
*
* See [`crate::atom::container`] for more useful types that wrap these.
*/

use bon::bon;

use crate::atom::AtomHeader;

use crate::AtomData;

use crate::atom::Atom;

/// Unwrap atom data enum given variant type.
///
/// # Example
/// ```ignore
/// let mut data = Atom::builder()
///     .header(AtomHeader::new(*TKHD))
///     .data(AtomData::TrackHeader(TrackHeaderAtom::default()))
///     .build();
/// let _: &mut TrackHeaderAtom = unwrap_atom_data!(
///     AtomRefMut(&mut data),
///     AtomData::TrackHeader,
/// );
/// ```
#[macro_export]
macro_rules! unwrap_atom_data {
    ($ref:expr, $variant:path $(,)?) => {{
        let atom = $ref.0;
        if let Some($variant(data)) = &mut atom.data {
            data
        } else {
            unreachable!(
                "invalid {} atom: data is None or the wrong variant",
                atom.header.atom_type,
            )
        }
    }};
}

#[derive(Debug, Clone, Copy)]
/// Wraps a shared [`Atom`] reference in an [`Option`] and provides methods for traversing it.
pub struct AtomRef<'a>(pub Option<&'a Atom>);

impl<'a> AtomRef<'a> {
    pub fn inner(&self) -> Option<&'a Atom> {
        self.0
    }

    pub fn find_child(&self, typ: &[u8; 4]) -> Option<&'a Atom> {
        self.children().find(|atom| atom.header.atom_type == typ)
    }

    pub fn children(&self) -> crate::atom::iter::AtomIter<'a> {
        crate::atom::iter::AtomIter::from_atom(self.0)
    }

    pub fn child_position(&self, typ: &[u8; 4]) -> Option<usize> {
        self.children()
            .position(|atom| atom.header.atom_type == typ)
    }

    pub fn child_rposition(&self, typ: &[u8; 4]) -> Option<usize> {
        self.children()
            .rposition(|atom| atom.header.atom_type == typ)
    }
}

#[derive(Debug)]
/// Wraps a mutable reference to an [`Atom`] and provides methods for manipulating and traversing it.
pub struct AtomRefMut<'a>(pub &'a mut Atom);

impl<'a> AtomRefMut<'a> {
    pub fn as_ref(&self) -> AtomRef<'_> {
        AtomRef(Some(self.0))
    }

    pub fn into_ref(self) -> AtomRef<'a> {
        AtomRef(Some(self.0))
    }

    pub fn atom_mut(&mut self) -> &'_ mut Atom {
        self.0
    }

    pub fn get_child(&mut self, index: usize) -> AtomRefMut<'_> {
        AtomRefMut(&mut self.0.children[index])
    }

    pub fn into_child(self, typ: &[u8; 4]) -> Option<&'a mut Atom> {
        self.into_children()
            .find(|atom| atom.header.atom_type == typ)
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children.iter_mut()
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        crate::atom::iter::AtomIterMut::from_atom(self.0)
    }

    pub fn insert_child(&mut self, index: usize, child: Atom) -> AtomRefMut<'_> {
        self.0.children.insert(index, child);
        self.get_child(index)
    }
}

#[bon]
impl<'a> AtomRefMut<'a> {
    #[builder]
    pub fn find_or_insert_child(
        &mut self,
        #[builder(start_fn)] atom_type: &[u8; 4],
        #[builder(default = Vec::new())] insert_before: Vec<&[u8; 4]>,
        #[builder(default = Vec::new())] insert_after: Vec<&[u8; 4]>,
        insert_index: Option<usize>,
        insert_data: Option<AtomData>,
    ) -> AtomRefMut<'_> {
        if let Some(index) = self.as_ref().child_position(atom_type) {
            self.get_child(index)
        } else {
            let index = insert_index.unwrap_or_else(|| {
                self.get_insert_position()
                    .before(insert_before)
                    .after(insert_after)
                    .call()
            });
            self.insert_child(
                index,
                Atom::builder()
                    .header(AtomHeader::new(*atom_type))
                    .maybe_data(insert_data)
                    .build(),
            )
        }
    }

    #[builder]
    pub fn get_insert_position(
        &self,
        #[builder(default = Vec::new())] before: Vec<&[u8; 4]>,
        #[builder(default = Vec::new())] after: Vec<&[u8; 4]>,
    ) -> usize {
        before
            .into_iter()
            .find_map(|typ| self.as_ref().child_rposition(typ))
            .or_else(|| {
                after
                    .into_iter()
                    .find_map(|typ| self.as_ref().child_position(typ))
                    .map(|i| i + 1)
            })
            .unwrap_or_default()
    }
}
