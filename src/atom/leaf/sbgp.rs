use futures_io::AsyncRead;

use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const SBGP: &[u8; 4] = b"sbgp";

/// Sample-to-Group Atom (sbgp) - ISO/IEC 14496-12
/// This atom maps samples to sample groups defined in the corresponding sgpd atom.
#[derive(Debug, Clone)]
pub struct SampleToGroupAtom {
    /// Version of the sbgp atom format (0 or 1+)
    pub version: u8,
    /// Flags for the atom
    pub flags: [u8; 3],
    /// Grouping type - identifies the type of grouping (must match corresponding sgpd)
    pub grouping_type: FourCC,
    /// Grouping type parameter (version >= 1 only)
    pub grouping_type_parameter: Option<u32>,
    /// Sample-to-group mapping entries
    pub entries: Vec<SampleToGroupEntry>,
}

/// A single sample-to-group mapping entry
#[derive(Debug, Clone)]
pub struct SampleToGroupEntry {
    /// Number of consecutive samples that belong to the same group
    pub sample_count: u32,
    /// Index into the sample group description table (1-based, 0 means no group assignment)
    pub group_description_index: u32,
}

impl ParseAtom for SampleToGroupAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != SBGP {
            return Err(ParseError::new_unexpected_atom(atom_type, SBGP));
        }
        let data = read_to_end(reader).await?;
        parser::parse_sbgp_data(&data)
    }
}

impl SerializeAtom for SampleToGroupAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*SBGP)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_sbgp_atom(self)
    }
}

mod serializer {
    use super::SampleToGroupAtom;

    pub fn serialize_sbgp_atom(sbgp: SampleToGroupAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(
            // Ensure version >= 1 if grouping_type_parameter is set
            if sbgp.version == 0 && sbgp.grouping_type_parameter.is_some() {
                1
            } else {
                sbgp.version
            },
        );
        data.extend(sbgp.flags);
        data.extend(sbgp.grouping_type.into_bytes());

        // Grouping type parameter is version >= 1 only
        if let Some(param) = sbgp.grouping_type_parameter {
            data.extend(param.to_be_bytes());
        }

        data.extend(
            u32::try_from(sbgp.entries.len())
                .expect("entries len should fit in u32")
                .to_be_bytes(),
        );

        for entry in sbgp.entries {
            data.extend(entry.sample_count.to_be_bytes());
            data.extend(entry.group_description_index.to_be_bytes());
        }

        data
    }
}

mod parser {
    use winnow::{
        binary::be_u32,
        combinator::{empty, repeat, seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::{SampleToGroupAtom, SampleToGroupEntry};
    use crate::atom::util::parser::{flags3, fourcc, stream, usize_be_u32, version, Stream};

    pub fn parse_sbgp_data(input: &[u8]) -> Result<SampleToGroupAtom, crate::ParseError> {
        parse_sbgp_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_sbgp_data_inner(input: &mut Stream<'_>) -> ModalResult<SampleToGroupAtom> {
        let maybe_group_type_parameter = |version: u8| {
            let with_parameter = |input: &mut Stream<'_>| -> ModalResult<Option<u32>> {
                be_u32.map(|v| Some(v)).parse_next(input)
            };
            let without_parameter = |input: &mut Stream<'_>| -> ModalResult<Option<u32>> {
                empty.value(None).parse_next(input)
            };

            if version >= 1 {
                with_parameter
            } else {
                without_parameter
            }
        };

        trace(
            "sbgp",
            seq!(SampleToGroupAtom {
                version: version,
                flags: flags3,
                grouping_type: fourcc.context(StrContext::Label("grouping_type")),
                grouping_type_parameter: maybe_group_type_parameter(version)
                    .context(StrContext::Label("grouping_type_parameter")),
                entries: entries.context(StrContext::Label("entries")),
            }),
        )
        .parse_next(input)
    }

    fn entries(input: &mut Stream<'_>) -> ModalResult<Vec<SampleToGroupEntry>> {
        trace("entries", move |input: &mut Stream<'_>| {
            let count = usize_be_u32
                .context(StrContext::Label("entry_count"))
                .parse_next(input)?;
            repeat(
                count,
                seq!(SampleToGroupEntry {
                    sample_count: be_u32.context(StrContext::Label("sample_count")),
                    group_description_index: be_u32
                        .context(StrContext::Label("group_description_index")),
                }),
            )
            .parse_next(input)
        })
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available sbgp test data files
    #[test]
    fn test_sbgp_roundtrip() {
        test_atom_roundtrip_sync::<SampleToGroupAtom>(SBGP);
    }
}
