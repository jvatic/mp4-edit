use futures_io::AsyncRead;
use std::fmt;

use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const TREF: &[u8; 4] = b"tref";

/// A single track reference entry containing the reference type and target track IDs
#[derive(Debug, Clone)]
pub struct TrackReference {
    /// The type of reference (e.g., "hint", "chap", "subt")
    pub reference_type: FourCC,
    /// List of track IDs that this reference points to
    pub track_ids: Vec<u32>,
}

impl TrackReference {
    /// Check if this reference is of a specific type
    pub fn is_type(&self, ref_type: &[u8; 4]) -> bool {
        self.reference_type == ref_type
    }
}

impl fmt::Display for TrackReference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> [{}]",
            self.reference_type,
            self.track_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Track Reference Atom (tref) - ISO/IEC 14496-12
/// Contains references from this track to other tracks
#[derive(Default, Debug, Clone)]
pub struct TrackReferenceAtom {
    /// List of track references
    pub references: Vec<TrackReference>,
}

impl TrackReferenceAtom {
    pub fn new(references: impl Into<Vec<TrackReference>>) -> Self {
        Self {
            references: references.into(),
        }
    }

    pub fn replace_references(&mut self, references: impl Into<Vec<TrackReference>>) -> &mut Self {
        self.references = references.into();
        self
    }
}

impl fmt::Display for TrackReferenceAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.references.is_empty() {
            write!(f, "TrackReferenceAtom {{ no references }}")
        } else {
            write!(f, "TrackReferenceAtom {{")?;
            for (i, reference) in self.references.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, " {reference}")?;
            }
            write!(f, " }}")
        }
    }
}

impl ParseAtom for TrackReferenceAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != TREF {
            return Err(ParseError::new_unexpected_atom(atom_type, TREF));
        }
        let data = read_to_end(reader).await?;
        parser::parse_tref_data(&data)
    }
}

impl SerializeAtom for TrackReferenceAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*TREF)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_tref_atom(self)
    }
}

mod serializer {
    use crate::atom::util::serializer::{prepend_size_inclusive, SizeU32};

    use super::TrackReferenceAtom;

    pub fn serialize_tref_atom(tref: TrackReferenceAtom) -> Vec<u8> {
        tref.references
            .into_iter()
            .flat_map(|reference| {
                prepend_size_inclusive::<SizeU32, _>(move || {
                    let mut data = Vec::new();

                    data.extend(reference.reference_type.into_bytes());
                    for track_id in reference.track_ids {
                        data.extend(track_id.to_be_bytes());
                    }

                    data
                })
            })
            .collect()
    }
}

mod parser {
    use winnow::{
        binary::be_u32,
        combinator::{repeat, seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::{TrackReference, TrackReferenceAtom};
    use crate::atom::util::parser::{
        combinators::inclusive_length_and_then, fourcc, stream, Stream,
    };

    pub fn parse_tref_data(input: &[u8]) -> Result<TrackReferenceAtom, crate::ParseError> {
        parse_tref_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_tref_data_inner(input: &mut Stream<'_>) -> ModalResult<TrackReferenceAtom> {
        trace(
            "tref",
            seq!(TrackReferenceAtom {
                references: repeat(0.., inclusive_length_and_then(be_u32, reference))
                    .context(StrContext::Label("references")),
            })
            .context(StrContext::Label("tref")),
        )
        .parse_next(input)
    }

    fn reference(input: &mut Stream<'_>) -> ModalResult<TrackReference> {
        trace(
            "reference",
            seq!(TrackReference {
                reference_type: fourcc.context(StrContext::Label("reference_type")),
                track_ids: repeat(0.., be_u32.context(StrContext::Label("track_id")))
                    .context(StrContext::Label("track_ids")),
            })
            .context(StrContext::Label("reference")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available tref test data files
    #[test]
    fn test_tref_roundtrip() {
        test_atom_roundtrip_sync::<TrackReferenceAtom>(TREF);
    }
}
