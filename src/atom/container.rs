/*!
 * Atoms with children.
 */

pub mod edts;
pub mod gmhd;
pub mod mdia;
pub mod meta;
pub mod minf;
pub mod moov;
pub mod stbl;
pub mod trak;
pub mod udta;

pub use edts::*;
pub use gmhd::*;
pub use mdia::*;
pub use meta::*;
pub use minf::*;
pub use moov::*;
pub use stbl::*;
pub use trak::*;
pub use udta::*;

use crate::FourCC;

pub const MFRA: FourCC = FourCC::new(b"mfra");
pub const DINF: FourCC = FourCC::new(b"dinf");
pub const MOOF: FourCC = FourCC::new(b"moof");
pub const TRAF: FourCC = FourCC::new(b"traf");
pub const SINF: FourCC = FourCC::new(b"sinf");
pub const SCHI: FourCC = FourCC::new(b"schi");

/// Determines whether a given atom type (fourcc) should be treated as a container for other atoms.
pub fn is_container_atom(atom_type: FourCC) -> bool {
    // Common container types in MP4
    matches!(
        atom_type,
        MOOV | MFRA
            | UDTA
            | TRAK
            | EDTS
            | MDIA
            | MINF
            | GMHD
            | DINF
            | STBL
            | MOOF
            | TRAF
            | SINF
            | SCHI
            | META
    )
}
