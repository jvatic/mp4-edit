/*!
 * This mod is concerned with mp4 atoms and how to {de}serialize them.
*/

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
pub(crate) mod util;

pub(crate) mod atom_ref;
pub mod container;
pub mod fourcc;
pub mod iter;
pub mod leaf;

use bon::bon;

use crate::{
    parser::{ParseAtom, ParseErrorKind},
    writer::SerializeAtom,
    ParseError,
};

pub use self::{container::*, fourcc::*, leaf::*};

/// Represents raw atom bytes.
#[derive(Clone)]
pub struct RawData {
    atom_type: FourCC,
    data: Vec<u8>,
}

impl RawData {
    pub fn new(atom_type: FourCC, data: Vec<u8>) -> Self {
        Self { atom_type, data }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn to_vec(self) -> Vec<u8> {
        self.data
    }
}

impl ParseAtom for RawData {
    async fn parse<R: futures_io::AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        mut reader: R,
    ) -> Result<Self, ParseError> {
        use futures_util::AsyncReadExt;
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .await
            .map_err(|err| ParseError {
                kind: ParseErrorKind::Io,
                location: None,
                source: Some(Box::new(err)),
            })?;
        Ok(RawData::new(atom_type, buffer).into())
    }
}

impl SerializeAtom for RawData {
    fn atom_type(&self) -> FourCC {
        self.atom_type
    }

    fn into_body_bytes(self) -> Vec<u8> {
        self.data
    }
}

impl std::fmt::Debug for RawData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[u8; {}]", self.data.len())
    }
}

#[derive(Debug, Clone)]
pub struct AtomHeader {
    pub atom_type: FourCC,
    pub offset: usize,
    pub header_size: usize,
    pub data_size: usize,
}

impl AtomHeader {
    pub fn new(atom_type: impl Into<FourCC>) -> Self {
        Self {
            atom_type: atom_type.into(),
            offset: 0,
            header_size: 0,
            data_size: 0,
        }
    }

    pub fn location(&self) -> (usize, usize) {
        (self.offset, self.header_size + self.data_size)
    }

    pub fn atom_size(&self) -> usize {
        self.header_size + self.data_size
    }
}

/// Represents a tree of mp4 atoms. Container atoms usually don't have [`Self::data`] (except for e.g. [meta]),
/// and leaf atoms don't have any [`Self::children`].
///
/// This structure allows us to represent any tree of mp4 atoms (boxes), even ones we don't (yet) support (via [RawData]).
#[derive(Debug, Clone)]
pub struct Atom {
    pub header: AtomHeader,
    pub data: Option<AtomData>,
    pub children: Vec<Atom>,
}

#[bon]
impl Atom {
    #[builder]
    pub fn new(
        header: AtomHeader,
        #[builder(into)] data: Option<AtomData>,
        #[builder(default = Vec::new())] children: Vec<Atom>,
    ) -> Self {
        Self {
            header,
            data,
            children,
        }
    }

    /// Recursively retains only the atoms that satisfy the predicate,
    /// one level of depth at a time (least to most nested).
    pub fn children_flat_retain_mut<P>(&mut self, mut pred: P)
    where
        P: FnMut(&mut Atom) -> bool,
    {
        let mut current_level = vec![self];

        while !current_level.is_empty() {
            let mut next_level = Vec::new();

            for atom in current_level {
                // Apply retain to this atom's children
                atom.children.retain_mut(|child| pred(child));

                // Collect remaining children for next level processing
                for child in &mut atom.children {
                    next_level.push(child);
                }
            }

            current_level = next_level;
        }
    }
}

impl SerializeAtom for Atom {
    fn atom_type(&self) -> FourCC {
        self.header.atom_type
    }

    /// Serialize [Atom]'s body and all children
    fn into_body_bytes(self) -> Vec<u8> {
        // Serialize all children
        let mut children_bytes = Vec::new();
        for child in self.children {
            let mut child_bytes = child.into_bytes();
            children_bytes.append(&mut child_bytes);
        }

        let mut body = self
            .data
            .map(SerializeAtom::into_body_bytes)
            .unwrap_or_default();

        body.append(&mut children_bytes);
        body
    }
}

macro_rules! define_atom_data {
    ( $(#[$meta:meta])* $enum:ident { $( $pattern:pat => $variant:ident($struct:ident) ),+ $(,)? } $(,)? ) => {
        $(#[$meta])*
        pub enum $enum {
            $( $variant($struct), )+
        }

        $(
            impl From<$struct> for $enum {
                fn from(atom: $struct) -> Self {
                    $enum::$variant(atom)
                }
            }
        )+

        impl ParseAtom for $enum {
            async fn parse<R: futures_io::AsyncRead + Unpin + Send>(
                atom_type: FourCC,
                reader: R,
            ) -> Result<Self, ParseError> {
                use std::ops::Deref;
                match atom_type.deref() {
                    $($pattern => $struct::parse(atom_type, reader).await.map($enum::from), )+
                }
            }
        }

        impl SerializeAtom for $enum {
            fn atom_type(&self) -> FourCC {
                match self {
                    $( $enum::$variant(atom) => atom.atom_type(), )+
                }
            }

            fn into_body_bytes(self) -> Vec<u8> {
                match self {
                    $( $enum::$variant(atom) => atom.into_body_bytes(), )+
                }
            }
        }
    };
}

define_atom_data!(
    /// Represents data contained in an atom (other than children).
    ///
    /// Usually only leaf atoms contain data, but some container types such as [meta] have some extra headers.
    #[derive(Debug, Clone)]
    AtomData {
        ftyp::FTYP => FileType(FileTypeAtom),
        mvhd::MVHD => MovieHeader(MovieHeaderAtom),
        mdhd::MDHD => MediaHeader(MediaHeaderAtom),
        elst::ELST => EditList(EditListAtom),
        hdlr::HDLR => HandlerReference(HandlerReferenceAtom),
        smhd::SMHD => SoundMediaHeader(SoundMediaHeaderAtom),
        gmin::GMIN => BaseMediaInfo(BaseMediaInfoAtom),
        text::TEXT => TextMediaInfo(TextMediaInfoAtom),
        ilst::ILST => ItemList(ItemListAtom),
        tkhd::TKHD => TrackHeader(TrackHeaderAtom),
        stsd::STSD => SampleDescriptionTable(SampleDescriptionTableAtom),
        tref::TREF => TrackReference(TrackReferenceAtom),
        dref::DREF => DataReference(DataReferenceAtom),
        stsz::STSZ => SampleSize(SampleSizeAtom),
        stco_co64::STCO | stco_co64::CO64 => ChunkOffset(ChunkOffsetAtom),
        stts::STTS => TimeToSample(TimeToSampleAtom),
        stsc::STSC => SampleToChunk(SampleToChunkAtom),
        chpl::CHPL => ChapterList(ChapterListAtom),
        free::FREE | free::SKIP => Free(FreeAtom),
        _ => RawData(RawData),
    },
);
