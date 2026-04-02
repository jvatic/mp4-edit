/*!
 * Slice metadata leaf atoms into bin files for round-trip testing.
 */

use anyhow::Result;
use clap::Parser as ClapParser;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};
use tokio::fs;
use tokio_util::compat::TokioAsyncReadCompatExt;

use mp4_edit::{
    atom::{
        is_container_atom,
        stsd::{self, StsdExtension},
        SampleDescriptionAtom,
    },
    writer::SerializeAtom,
    Atom, AtomData, FourCC, Parser,
};

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input mp4
    input_mp4: PathBuf,

    /// Path to the output dir
    output_dir: PathBuf,

    /// FourCC to extract (default is everything)
    atom: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Validate supplied atom types to filter
    let atom_types = args.atom.map(|atoms| {
        atoms
            .into_iter()
            .map(|typ| {
                FourCC::new(
                    typ.as_bytes()
                        .try_into()
                        .expect(format!("invalid atom type: {typ}").as_str()),
                )
            })
            .collect::<Vec<_>>()
    });

    // Open input file and parse it's metadata
    let file = fs::File::open(&args.input_mp4).await?;
    let parser = Parser::new_seekable(file.compat());
    let metadata = parser.parse_metadata().await?.into_metadata();

    // Extract leaf atoms into output dir
    let file = std::fs::File::open(args.input_mp4)?;
    LeafAtomExtractor::new(file, atom_types, args.output_dir)
        .extract_atoms(metadata.into_atoms().into_iter());

    Ok(())
}

struct LeafAtomExtractor {
    mp4_file: std::fs::File,
    atom_types: Option<Vec<FourCC>>,
    output_dir: PathBuf,
}

impl LeafAtomExtractor {
    pub fn new(file: std::fs::File, atom_types: Option<Vec<FourCC>>, output_dir: PathBuf) -> Self {
        Self {
            mp4_file: file,
            atom_types,
            output_dir,
        }
    }

    pub fn extract_atoms(&mut self, atoms: impl Iterator<Item = Atom>) {
        atoms.for_each(|atom| {
            if is_container_atom(atom.atom_type()) {
                self.extract_atoms(atom.children.into_iter());
            } else {
                self.maybe_extract_atom(atom);
            }
        });
    }

    fn maybe_extract_atom(&mut self, atom: Atom) {
        if match &self.atom_types {
            Some(atom_types) => atom_types.contains(&atom.header.atom_type),
            None => true,
        } {
            self.extract_atom(atom);
        }
    }

    fn extract_atom(&mut self, atom: Atom) {
        self.mp4_file
            .seek(SeekFrom::Start(atom.header.offset as u64))
            .expect("error seeking input file");

        let mut buf = vec![0u8; atom.header.atom_size()];
        self.mp4_file
            .read_exact(&mut buf)
            .expect("error reading atom bytes");

        match atom.data {
            Some(AtomData::SampleDescription(stsd)) => {
                self.extract_stsd_extensions(stsd)
                    .expect("error extracting stsd extensions");
            }
            _ => {}
        }

        let hash = seahash::hash(&buf);
        let mut output_file = match open_output_file(&self.output_dir, atom.header.atom_type, hash)
        {
            Ok(file) => file,
            Err(err) if matches!(err.kind(), std::io::ErrorKind::AlreadyExists) => {
                return;
            }
            Err(err) => panic!("error opening output file: {err}"),
        };

        output_file
            .write_all(&mut buf)
            .expect("error writing atom bytes");
    }

    fn extract_stsd_extensions(&self, stsd: SampleDescriptionAtom) -> Result<()> {
        let empty_list: Vec<StsdExtension> = Vec::with_capacity(0);
        let extensions = stsd
            .entries
            .into_iter()
            .map(|entry| match entry.data {
                stsd::SampleEntryData::Audio(entry) => entry.extensions.into_iter(),
                _ => empty_list.clone().into_iter(),
            })
            .flatten();

        for extension in extensions {
            match extension {
                // only extract unknown extensions since we don't have the offsets to get the original data
                StsdExtension::Unknown { fourcc, data } => {
                    self.extract_stsd_extension(fourcc, data)?
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn extract_stsd_extension(&self, fourcc: FourCC, mut data: Vec<u8>) -> Result<()> {
        let output_dir = self.output_dir.join("stsd");

        let hash = seahash::hash(&data);
        let mut output_file = match open_output_file(&output_dir, fourcc, hash) {
            Ok(file) => file,
            Err(err) if matches!(err.kind(), std::io::ErrorKind::AlreadyExists) => {
                return Ok(());
            }
            Err(err) => return Err(err.into()),
        };
        output_file.write_all(&mut data)?;

        Ok(())
    }
}

fn open_output_file(dir: &PathBuf, atom_type: FourCC, hash: u64) -> std::io::Result<std::fs::File> {
    std::fs::create_dir_all(dir)?;

    let mut counter = 0;
    loop {
        let filename = format!("{atom_type}{counter:02}.bin");
        let filepath = dir.join(filename);
        match std::fs::File::create_new(&filepath) {
            Ok(file) => return Ok(file),
            Err(err) if matches!(err.kind(), std::io::ErrorKind::AlreadyExists) => {
                let mut file = std::fs::File::open(filepath)?;
                let mut data = Vec::new();
                file.read_to_end(&mut data)?;
                if seahash::hash(&data) == hash {
                    return Err(err);
                }

                counter += 1;
            }
            Err(err) => return Err(err),
        }
    }
}
