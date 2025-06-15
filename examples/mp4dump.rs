use anyhow::Context;
use futures_util::pin_mut;
use futures_util::stream::StreamExt;
use std::env;
use tokio::{
    fs,
    io::{self, AsyncRead},
};
use tokio_util::compat::TokioAsyncReadCompatExt;

use mp4_parser::{Atom, AtomData, ParseMetadataEvent, Parser};

/// Format file size in human-readable format
fn format_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Get a summary of atom data
fn get_atom_summary(data: &Option<AtomData>) -> String {
    match data {
        Some(atom) => format!("{atom:?}"),
        None => "".to_string(),
    }
}

/// Print atom with proper formatting and indentation
fn print_atom(atom: &Atom, indent: usize) {
    let indent_str = if indent > 0 {
        format!("{:<width$}", "", width = indent)
    } else {
        String::new()
    };

    let atom_display = format!("{}{}", indent_str, atom.atom_type);
    let size_display = format_size(atom.size);
    let offset_display = format!(
        "0x{:08x}..=0x{:08x}",
        atom.offset,
        atom.offset + atom.size - 1
    );
    let summary = get_atom_summary(&atom.data);

    // Color coding based on atom type
    let atom_color = match atom.atom_type.to_string().as_str() {
        "ftyp" | "styp" => "\x1b[1;32m",          // Green for file type
        "moov" | "trak" | "mdia" => "\x1b[1;34m", // Blue for containers
        "mvhd" | "tkhd" | "mdhd" => "\x1b[1;35m", // Magenta for headers
        "stbl" | "stts" | "stsc" | "stsz" | "stco" | "co64" => "\x1b[1;31m", // Red for sample tables
        _ => "\x1b[0m",                                                      // Default
    };

    println!("\x1b[1;36m│\x1b[0m {}{:<20}\x1b[0m │ \x1b[2m{:<23}\x1b[0m │ \x1b[1m{:<10}\x1b[0m │ \x1b[2m{:<30}\x1b[0m",
        atom_color, atom_display, offset_display, size_display, summary);
}

/// Print table header
fn print_table_header() {
    println!("\n\x1b[1;36m╭─ MP4 Atom Structure ──────────────────────────────────────────────────────────────\x1b[0m");
    println!("\x1b[1;36m│\x1b[0m");
    println!(
        "\x1b[1;36m│\x1b[0m \x1b[1;33m{:<20} │ {:<23} │ {:<10} │ {:<30}\x1b[0m",
        "Atom Type", "Offset Range", "Size", "Summary"
    );
    println!(
        "\x1b[1;36m│\x1b[0m \x1b[2m{:─<20}─┼─{:─<23}─┼─{:─<10}─┼─{:─<30}\x1b[0m",
        "", "", "", ""
    );
}

/// Print table footer
fn print_table_footer() {
    println!("\x1b[1;36m│\x1b[0m");
    println!("\x1b[1;36m╰────────────────────────────────────────────────────────────────────────────────────\x1b[0m\n");
}

/// Print atoms directly from stream events
async fn print_atoms_from_stream<R: futures_io::AsyncRead + Unpin + Send>(
    mut parser: Parser<R>,
) -> anyhow::Result<usize> {
    let stream = parser.stream_metadata();
    pin_mut!(stream);

    let mut indent_level = 0;
    let mut atom_count = 0;
    let mut first_atom = true;

    while let Some(event) = stream.next().await {
        let event = event.context("Failed to parse stream event")?;

        match event {
            ParseMetadataEvent::EnterContainer(atom) => {
                if first_atom {
                    print_table_header();
                    first_atom = false;
                }

                print_atom(&atom, indent_level);
                indent_level += 1;
                atom_count += 1;
            }
            ParseMetadataEvent::Leaf(atom) => {
                if first_atom {
                    print_table_header();
                    first_atom = false;
                }

                print_atom(&atom, indent_level);
                atom_count += 1;
            }
            ParseMetadataEvent::ExitContainer => {
                if indent_level > 0 {
                    indent_level -= 1;
                }
            }
        }
    }

    if !first_atom {
        print_table_footer();
    }

    Ok(atom_count)
}

async fn open_input(input_name: &str) -> anyhow::Result<Box<dyn AsyncRead + Unpin + Send>> {
    if input_name == "-" {
        Ok(Box::new(io::stdin()))
    } else {
        let file = fs::File::open(input_name).await?;
        Ok(Box::new(file))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_filename>", args[0]);
        std::process::exit(1);
    }
    let file = open_input(args[1].as_str()).await?;

    println!("\x1b[1;32m🎬 Analyzing MP4 file: {}\x1b[0m", &args[1]);

    let parser = Parser::new(file.compat());
    let atom_count = print_atoms_from_stream(parser)
        .await
        .context("Failed to parse MP4 file")?;

    // Print summary statistics
    println!("\x1b[1;33m📊 Summary:\x1b[0m");
    println!("   Total atoms: \x1b[1m{}\x1b[0m", atom_count);
    if args[1].as_str() != "-" {
        let file_size = fs::metadata(&args[1]).await?.len();
        println!("   File size: \x1b[1m{}\x1b[0m", format_size(file_size));
    }

    Ok(())
}
