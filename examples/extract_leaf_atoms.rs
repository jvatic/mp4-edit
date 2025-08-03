use anyhow::{anyhow, Context, Result};
use std::{env, path::Path};
use tokio::fs;
use tokio_util::compat::TokioAsyncReadCompatExt;

use mp4_parser::{atom::containers, Atom, Parser};

/// Check if an atom type is a container atom
fn is_container_atom(atom_type: &[u8; 4]) -> bool {
    matches!(
        atom_type,
        containers::MOOV
            | containers::MFRA
            | containers::UDTA
            | containers::TRAK
            | containers::EDTS
            | containers::MDIA
            | containers::MINF
            | containers::DINF
            | containers::STBL
            | containers::MOOF
            | containers::TRAF
            | containers::SINF
            | containers::SCHI
            | containers::META
    )
}

/// Extract leaf atoms from an MP4 file and write them to individual files
async fn extract_leaf_atoms(input_path: &str, output_dir: &str) -> Result<()> {
    // Open the input file
    let file = fs::File::open(input_path)
        .await
        .with_context(|| format!("Failed to open input file: {}", input_path))?;

    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)
        .await
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;

    // Parse the MP4 file
    println!("📂 Parsing MP4 file: {}", input_path);
    let parser = Parser::new(file.compat());
    let metadata = parser
        .parse_metadata()
        .await
        .context("Failed to parse MP4 metadata")?;

    // Read the original file again to extract raw atom data
    let original_data = fs::read(input_path)
        .await
        .with_context(|| format!("Failed to read input file: {}", input_path))?;

    let mut leaf_count = 0;
    let mut total_size = 0;

    // Process all atoms iteratively using a stack
    let mut atoms_to_process: Vec<&Atom> = metadata.atoms_iter().collect();

    while let Some(atom) = atoms_to_process.pop() {
        // Check if this is a leaf atom (no children)
        if atom.children.is_empty() {
            // Skip container atoms even if they have no children
            if !is_container_atom(&atom.header.atom_type) {
                extract_single_atom(atom, &original_data, output_dir).await?;
                leaf_count += 1;
                total_size += atom.header.atom_size();
            }
        } else {
            // Add children to the stack for processing
            for child in &atom.children {
                atoms_to_process.push(child);
            }
        }
    }

    println!("\n✅ Extraction complete!");
    println!("   📄 Leaf atoms extracted: {}", leaf_count);
    println!("   📊 Total size: {} bytes", total_size);

    Ok(())
}

/// Extract a single leaf atom and write it to a file
async fn extract_single_atom(atom: &Atom, original_data: &[u8], output_dir: &str) -> Result<()> {
    let atom_type_str = atom.header.atom_type.to_string();

    let mut output_path = None;
    for filename in std::iter::repeat_n(0, 1_00)
        .enumerate()
        .map(|(i, _)| format!("{atom_type_str}{i:02}.bin"))
    {
        let output_path_candidate = Path::new(output_dir).join(&filename);
        if fs::metadata(&output_path_candidate).await.is_err() {
            output_path = Some((filename, output_path_candidate));
            break;
        }
    }
    let (filename, output_path) = output_path
        .ok_or_else(|| anyhow!("failed to find suitable filename for {atom_type_str}"))?;

    // Extract the complete atom data (header + body) from the original file
    let atom_start = atom.header.offset;
    let atom_end = atom_start + atom.header.atom_size();

    if atom_end > original_data.len() {
        return Err(anyhow::anyhow!(
            "Atom {} extends beyond file boundaries (offset: {}, size: {}, file size: {})",
            atom_type_str,
            atom_start,
            atom.header.atom_size(),
            original_data.len()
        ));
    }

    let atom_data = &original_data[atom_start..atom_end];

    // Write the atom data to file
    fs::write(&output_path, atom_data)
        .await
        .with_context(|| format!("Failed to write atom file: {}", output_path.display()))?;

    println!(
        "   💾 {} → {} ({} bytes)",
        atom_type_str,
        filename,
        atom_data.len()
    );

    Ok(())
}

/// Print usage information
fn print_usage(program_name: &str) {
    eprintln!("Usage: {} <input_mp4> [output_dir]", program_name);
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  input_mp4   Path to the MP4 file to extract atoms from");
    eprintln!("  output_dir  Directory to write extracted atoms to (default: current directory)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} video.mp4", program_name);
    eprintln!("  {} video.mp4 ./atoms", program_name);
    eprintln!();
    eprintln!("This tool extracts leaf atoms from MP4 metadata and saves each atom");
    eprintln!("as a separate binary file (e.g., ilst.bin, chpl.bin) containing the");
    eprintln!("complete atom data including header and size information.");
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_dir = if args.len() > 2 { &args[2] } else { "." };

    // Validate input file exists
    if !Path::new(input_path).exists() {
        eprintln!("❌ Error: Input file does not exist: {}", input_path);
        std::process::exit(1);
    }

    println!("🎬 MP4 Leaf Atom Extractor");
    println!("   Input:  {}", input_path);
    println!("   Output: {}", output_dir);
    println!();

    if let Err(e) = extract_leaf_atoms(input_path, output_dir).await {
        eprintln!("❌ Error: {}", e);

        // Print the error chain for more context
        let mut source = e.source();
        while let Some(err) = source {
            eprintln!("   Caused by: {}", err);
            source = err.source();
        }

        std::process::exit(1);
    }

    Ok(())
}
