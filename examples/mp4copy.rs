use anyhow::Context;
use std::{env, ops::Deref};
use tokio::{
    fs,
    io::{self, AsyncRead},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use mp4_parser::{
    atom::{
        containers::{DINF, EDTS, META},
        hdlr::{HandlerType, HDLR},
        ilst::ILST,
        tref::TREF,
        FourCC,
    },
    Mp4Writer, Parser,
};

async fn open_input(input_name: &str) -> anyhow::Result<Box<dyn AsyncRead + Unpin + Send>> {
    if input_name == "-" {
        Ok(Box::new(io::stdin()))
    } else {
        let file = fs::File::open(input_name).await?;
        Ok(Box::new(file))
    }
}

async fn create_output_file(output_name: &str) -> anyhow::Result<fs::File> {
    if output_name == "-" {
        anyhow::bail!("Stdout output not supported in this version");
    } else {
        let file = fs::File::create(output_name).await?;
        Ok(file)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_mp4> <output_mp4>", args[0]);
        eprintln!("  Use '-' for stdin/stdout");
        std::process::exit(1);
    }

    let input_name = &args[1];
    let output_name = &args[2];

    println!(
        "🎬 Copying MP4 metadata from {} to {}",
        input_name, output_name
    );

    // Open input file and create parser
    let input_file = open_input(input_name).await?;
    let parser = Parser::new(input_file.compat());

    // Parse metadata atoms from input
    println!("📖 Reading metadata atoms...");
    let (_, metadata) = parser
        .parse_metadata()
        .await
        .context("Failed to parse metadata from input file")?;
    let atoms = metadata
        .file_type_mut(|ftyp| {
            ftyp.major_brand = FourCC::from(*b"M4B ");
            ftyp.minor_version = 0x00000200;
            ftyp.compatible_brands = vec![
                FourCC::from(*b"isom"),
                FourCC::from(*b"M4B "),
                FourCC::from(*b"M4A "),
                FourCC::from(*b"mp42"),
            ]
        })
        .tracks_retain(|trak| {
            trak.media()
                .and_then(|m| {
                    m.handler_reference()
                        .map(|hr| matches!(hr.handler_type, HandlerType::Audio))
                })
                .unwrap_or(false)
        })
        .atoms_flat_retain_mut(|atom| match atom.atom_type.deref() {
            DINF | TREF | EDTS => false,
            META => {
                atom.children_flat_retain_mut(|atom| match atom.atom_type.deref() {
                    HDLR => false,
                    ILST => {
                        // TODO: edit tags
                        true
                    }
                    _ => true,
                });
                true
            }
            _ => true,
        })
        .into_atoms();

    println!("✅ Found {} metadata atoms", atoms.len());

    // Open output file for writing
    let output_file = create_output_file(output_name).await?;
    let mut output_writer = output_file.compat_write();

    // Write each metadata atom to output
    println!("✍️  Writing metadata atoms...");
    for (i, atom) in atoms.iter().enumerate() {
        Mp4Writer::write_atom(&mut output_writer, atom.clone())
            .await
            .with_context(|| format!("Failed to write atom {} ({})", i + 1, atom.atom_type))?;

        println!(
            "  ✓ Wrote atom: {} (size: {} bytes)",
            atom.atom_type, atom.size
        );
    }

    println!("🎉 Successfully copied {} metadata atoms!", atoms.len());

    Ok(())
}
