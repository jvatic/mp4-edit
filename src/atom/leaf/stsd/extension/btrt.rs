pub const BTRT: &'static [u8; 4] = b"btrt";

#[derive(Debug, Clone, PartialEq)]
pub struct BtrtExtension {
    pub buffer_size_db: u32,
    pub max_bitrate: u32,
    pub avg_bitrate: u32,
}

pub(crate) mod serializer {
    use super::BtrtExtension;

    pub fn serialize_btrt_extension(btrt: BtrtExtension) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(btrt.buffer_size_db.to_be_bytes());
        data.extend(btrt.max_bitrate.to_be_bytes());
        data.extend(btrt.avg_bitrate.to_be_bytes());
        data
    }
}

pub(crate) mod parser {
    use winnow::{
        binary::be_u32,
        combinator::{seq, trace},
        ModalResult, Parser,
    };

    use super::BtrtExtension;

    use crate::atom::{stsd::StsdExtension, util::parser::Stream};

    pub fn parse_btrt_extension(input: &mut Stream<'_>) -> ModalResult<StsdExtension> {
        trace(
            "btrt",
            seq!(BtrtExtension {
                buffer_size_db: be_u32,
                max_bitrate: be_u32,
                avg_bitrate: be_u32,
            }),
        )
        .map(StsdExtension::Btrt)
        .parse_next(input)
    }

    #[cfg(test)]
    mod tests {
        use crate::atom::{stsd::extension::btrt::BTRT, test_utils::test_stsd_extension_roundtrip};

        /// Test round-trip for all available stsd/btrt test data files
        #[test]
        fn test_btrt_roundtrip() {
            test_stsd_extension_roundtrip(BTRT);
        }
    }
}
