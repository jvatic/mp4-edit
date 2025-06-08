use anyhow::Context;
use std::{env, fs::File};

use mp4_parser::{parse_mp4, Atom, AtomData};

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

/// Recursively print [Atom]s with indentation to show nesting along with offset and size details.
fn print_atoms(atoms: &[Atom], indent: usize) {
    if indent == 0 {
        println!("\n\x1b[1;36m╭─ MP4 Atom Structure ──────────────────────────────────────────────────────────────\x1b[0m");
        println!("\x1b[1;36m│\x1b[0m");
        println!(
            "\x1b[1;36m│\x1b[0m \x1b[1;33m{:<20} │ {:<12} │ {:<10} │ {:<30}\x1b[0m",
            "Atom Type", "Offset", "Size", "Summary"
        );
        println!(
            "\x1b[1;36m│\x1b[0m \x1b[2m{:─<20}─┼─{:─<12}─┼─{:─<10}─┼─{:─<30}\x1b[0m",
            "", "", "", ""
        );
    }

    for atom in atoms {
        let indent_str = if indent > 0 {
            format!("{:<width$}", "", width = indent)
        } else {
            String::new()
        };

        let atom_display = format!("{}{}", indent_str, atom.atom_type);
        let size_display = format_size(atom.size);
        let offset_display = format!("0x{:08x}", atom.offset);
        let summary = get_atom_summary(&atom.data);

        // Color coding based on atom type
        let atom_color = match atom.atom_type.to_string().as_str() {
            "ftyp" | "styp" => "\x1b[1;32m",          // Green for file type
            "moov" | "trak" | "mdia" => "\x1b[1;34m", // Blue for containers
            "mvhd" | "tkhd" | "mdhd" => "\x1b[1;35m", // Magenta for headers
            "stbl" | "stts" | "stsc" | "stsz" | "stco" | "co64" => "\x1b[1;31m", // Red for sample tables
            _ => "\x1b[0m",                                                      // Default
        };

        if indent == 0 {
            println!("\x1b[1;36m│\x1b[0m {}{:<20}\x1b[0m │ \x1b[2m{:<12}\x1b[0m │ \x1b[1m{:<10}\x1b[0m │ \x1b[2m{:<30}\x1b[0m",
                atom_color, atom_display, offset_display, size_display, summary);
        } else {
            println!("\x1b[1;36m│\x1b[0m {}{:<20}\x1b[0m │ \x1b[2m{:<12}\x1b[0m │ \x1b[1m{:<10}\x1b[0m │ \x1b[2m{:<30}\x1b[0m",
                atom_color, atom_display, offset_display, size_display, summary);
        }

        if !atom.children.is_empty() {
            print_atoms(&atom.children, indent + 1);
        }
    }

    if indent == 0 {
        println!("\x1b[1;36m│\x1b[0m");
        println!("\x1b[1;36m╰────────────────────────────────────────────────────────────────────────────────────\x1b[0m\n");
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_filename>", args[0]);
        std::process::exit(1);
    }
    let input = File::open(&args[1])?;

    let atoms = parse_mp4(input).context("Failed to parse MP4 file")?;

    println!("\x1b[1;32m🎬 Analyzing MP4 file: {}\x1b[0m", &args[1]);
    print_atoms(&atoms, 0);

    // Print summary statistics
    let total_atoms = count_atoms(&atoms);
    let file_size = std::fs::metadata(&args[1])?.len();
    println!("\x1b[1;33m📊 Summary:\x1b[0m");
    println!("   Total atoms: \x1b[1m{}\x1b[0m", total_atoms);
    println!("   File size: \x1b[1m{}\x1b[0m", format_size(file_size));

    Ok(())
}

/// Count total number of atoms recursively
fn count_atoms(atoms: &[Atom]) -> usize {
    atoms
        .iter()
        .map(|atom| 1 + count_atoms(&atom.children))
        .sum()
}
