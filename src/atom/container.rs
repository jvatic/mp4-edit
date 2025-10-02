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

pub const MFRA: &[u8; 4] = b"mfra";
pub const DINF: &[u8; 4] = b"dinf";
pub const MOOF: &[u8; 4] = b"moof";
pub const TRAF: &[u8; 4] = b"traf";
pub const SINF: &[u8; 4] = b"sinf";
pub const SCHI: &[u8; 4] = b"schi";

/// Determines whether a given atom type (fourcc) should be treated as a container for other atoms.
pub fn is_container_atom(atom_type: &[u8; 4]) -> bool {
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
