use anyhow::{anyhow, Context};
use futures_util::{
    io::{BufReader, BufWriter},
    AsyncSeekExt, AsyncWriteExt,
};
use progress_bar::pb::ProgressBar;
use std::{collections::HashMap, env, io::SeekFrom, ops::Deref};
use tokio::{
    fs,
    io::{self, AsyncRead},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use mp4_parser::{
    atom::{
        chpl::CHPL,
        containers::{DINF, EDTS, META, MOOV},
        free::FREE,
        ftyp::FTYP,
        gmhd::GMHD,
        hdlr::{HandlerType, HDLR},
        ilst::ILST,
        sbgp::SBGP,
        sgpd::SGPD,
        stco_co64::{ChunkOffsets, STCO},
        stsd::{
            BtrtExtension, DecoderSpecificInfo, SampleEntryData, SampleEntryType, StsdExtension,
        },
        stsz::SampleEntrySizes,
        stts::{TimeToSampleEntries, TimeToSampleEntry},
        tref::TREF,
        FileTypeAtom, FourCC, FreeAtom,
    },
    Atom, AtomData, Mp4Writer, Parser,
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

    println!("🎬 Copying {} into {}", input_name, output_name);

    // Open input file and create parser
    let input_file = open_input(input_name).await?;
    let input_reader = BufReader::new(input_file.compat());
    let parser = Parser::new(input_reader);

    // Parse metadata atoms from input
    let (mut reader, metadata) = parser
        .parse_metadata()
        .await
        .context("Failed to parse metadata from input file")?;

    // Filter out non-audio tracks
    let metadata = metadata.tracks_retain(|trak| {
        trak.media()
            .and_then(|mdia| mdia.handler_reference())
            .and_then(|hdlr| Some(matches!(hdlr.handler_type, HandlerType::Audio)))
            .unwrap_or_default()
    });

    let num_samples = metadata.tracks_iter().fold(0, |n, trak| {
        n + trak
            .media()
            .and_then(|m| m.media_information())
            .and_then(|m| m.sample_table())
            .and_then(|st| st.sample_size())
            .and_then(|s| Some(s.sample_count))
            .unwrap_or_default()
    });

    let mut progress_bar = ProgressBar::new_with_eta(num_samples as usize);

    let mut moov_size = 0;
    let mut metadata = metadata.atoms_flat_retain_mut(|atom| match atom.atom_type.deref() {
        FTYP | FREE | TREF | EDTS | SGPD | SBGP | GMHD | CHPL => false,
        MOOV => {
            moov_size = atom.size;
            true
        }
        META => {
            atom.children_flat_retain_mut(|atom| match atom.atom_type.deref() {
                HDLR => false,
                ILST => {
                    // TODO: edit tags
                    true
                }
                _ => true,
            });
            false
        }
        _ => true,
    });

    // Open output file for writing
    let output_file = create_output_file(output_name).await?;
    let output_writer = output_file.compat_write();
    let mut output_writer = BufWriter::new(output_writer);

    let mut mp4_writer = Mp4Writer::new();

    // Write the ftyp atom
    mp4_writer
        .write_atom(
            &mut output_writer,
            Atom {
                atom_type: FourCC::from(*FTYP),
                offset: 0,
                size: 0,
                data: Some(AtomData::FileType(FileTypeAtom {
                    major_brand: FourCC::from(*b"isom"),
                    minor_version: 0x00000200,
                    compatible_brands: vec![
                        FourCC::from(*b"isom"),
                        FourCC::from(*b"iso2"),
                        FourCC::from(*b"mp41"),
                    ],
                })),
                children: Vec::new(),
            },
        )
        .await
        .context("error writing ftyp atom")?;

    // Write FREE atom to reserve enough space for MOOV
    // (we shouldn't need more space than in the input file, but add padding just in case)
    let free_offset = mp4_writer.current_offset();
    let free_content_size = (moov_size as usize); // + (400 << 10);
    let free_atom_bytes = Mp4Writer::serialize_atom(&Atom {
        atom_type: FourCC::from(*b"free"),
        offset: 0,
        size: 0,
        data: Some(AtomData::Free(FreeAtom {
            atom_type: FourCC::from(*b"free"),
            data_size: free_content_size,
            data: vec![0u8; free_content_size],
        })),
        children: Vec::new(),
    });
    let free_size = free_atom_bytes.len();
    mp4_writer
        .write_raw(&mut output_writer, &free_atom_bytes)
        .await
        .context("error writing free atom")?;

    // Write MDAT header (it will have a size=0 which we'll update later)
    let mdat_offset = mp4_writer.current_offset();
    mp4_writer
        .write_atom(
            &mut output_writer,
            Atom {
                atom_type: FourCC::from(*b"mdat"),
                data: None,
                children: Vec::new(),
                offset: 0,
                size: 0,
            },
        )
        .await
        .context("error writing mdat placeholder header")?;

    // Copy and write sample data
    let mut chunk_offsets = metadata
        .tracks_iter()
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let mut sample_sizes = metadata
        .tracks_iter()
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    let mut chunk_idx = 0;
    let mut sample_idx = 0;
    let mut chunk_parser = metadata.chunks()?;
    while let Some(chunk) = chunk_parser.read_next_chunk(&mut reader).await? {
        chunk_offsets[chunk.trak_idx].push(mp4_writer.current_offset() as u64);

        for (i, sample) in chunk.samples().enumerate() {
            sample_sizes[chunk.trak_idx].push(sample.size);

            mp4_writer
                .write_raw(&mut output_writer, &sample.data)
                .await
                .context(format!(
                    "error writing sample {i:02} data in chunk {chunk_idx:02}"
                ))?;

            sample_idx += 1;
            progress_bar.set_progress(sample_idx);
        }

        chunk_idx += 1;
    }
    output_writer
        .flush()
        .await
        .context("error writing mdat content (flush)")?;

    // Calculate bitrate (AAXC file is wrong)
    let mut track_bitrate = Vec::with_capacity(metadata.tracks_iter().count());
    for (trak_idx, trak) in metadata.tracks_iter().enumerate() {
        let num_bits = sample_sizes[trak_idx].iter().sum::<u32>() * 8;

        let duration_secds = trak
            .media()
            .and_then(|m| m.header())
            .and_then(|mdhd| Some((mdhd.duration as f64) / (mdhd.timescale as f64)))
            .unwrap_or_default();

        let bitrate = (num_bits as f64) / duration_secds;
        track_bitrate.push(bitrate.round() as u32);
    }

    // Update metadata
    metadata
        .tracks_iter_mut()
        .enumerate()
        .for_each(|(trak_idx, trak)| {
            let stbl = trak
                .media()
                .and_then(|mdia| mdia.media_information())
                .and_then(|minf| minf.sample_table());
            if let Some(mut stbl) = stbl {
                if let Some(stsz) = stbl.sample_size() {
                    stsz.sample_count = sample_sizes[trak_idx].len() as u32;
                    stsz.entry_sizes = SampleEntrySizes::from(sample_sizes[trak_idx].clone());
                }

                if let Some(stco) = stbl.chunk_offset() {
                    stco.chunk_offsets = ChunkOffsets::from(chunk_offsets[trak_idx].clone());
                }

                if let Some(stsd) = stbl.sample_description() {
                    stsd.entries.retain_mut(|entry| {
                        if !matches!(entry.data, SampleEntryData::Audio(_)) {
                            return true;
                        }

                        let bitrate = track_bitrate[trak_idx];

                        entry.entry_type = SampleEntryType::Mp4a;

                        if let SampleEntryData::Audio(audio) = &mut entry.data {
                            let mut sample_frequency = None;
                            audio.extensions.retain_mut(|ext| match ext {
                                StsdExtension::Esds(esds) => {
                                    esds.es_descriptor.decoder_config_descriptor.as_mut().map(
                                        |c| {
                                            c.avg_bitrate = bitrate;
                                            c.max_bitrate = bitrate;
                                            if let Some(DecoderSpecificInfo::Audio(a)) =
                                                c.decoder_specific_info.as_ref()
                                            {
                                                sample_frequency =
                                                    Some(a.sampling_frequency.as_hz());
                                            }
                                        },
                                    );
                                    true
                                }
                                StsdExtension::Btrt(_) => false,
                                StsdExtension::Unknown { .. } => false,
                            });
                            audio.extensions.push(StsdExtension::Btrt(BtrtExtension {
                                buffer_size_db: 0,
                                avg_bitrate: bitrate,
                                max_bitrate: bitrate,
                            }));

                            if let Some(hz) = sample_frequency {
                                audio.sample_rate = hz as f32;
                            }
                        }

                        true
                    });
                }
            }
        });

    // Write correct mdat header
    let mdat_header_size = 8;
    let mdat_size = mp4_writer.current_offset() - mdat_offset - mdat_header_size;
    output_writer
        .seek(SeekFrom::Start(mdat_offset as u64))
        .await
        .context("error seeding to mdat start")?;
    output_writer
        .write_all(&Mp4Writer::serialize_atom_header(
            FourCC::from(*b"mdat"),
            mdat_size as u64,
        ))
        .await
        .context("error writing mdat header")?;
    output_writer
        .flush()
        .await
        .context("error writing mdat header (flush)")?;

    println!("free_offset={free_offset}, free_size={free_size}");

    // Write metadata atoms where we currently have a FREE atom
    output_writer
        .seek(SeekFrom::Start(free_offset as u64))
        .await
        .context("error seeding to mdat start")?;

    let start_offset = mp4_writer.current_offset();
    for (i, atom) in metadata.atoms_iter().enumerate() {
        mp4_writer
            .write_atom(&mut output_writer, atom.clone())
            .await
            .with_context(|| format!("Failed to write atom {} ({})", i + 1, atom.atom_type))?;
    }
    let metadata_size = mp4_writer.current_offset() - start_offset;

    // Write FREE atom
    if metadata_size > free_size {
        return Err(anyhow!(
            "metadata larger than reserved size, overwrote {} bytes of mdat",
            metadata_size - free_size
        ));
    }
    if metadata_size != free_size {
        let free_atom_bytes = Mp4Writer::serialize_atom(&Atom {
            atom_type: FourCC::from(*b"free"),
            offset: 0,
            size: 0,
            data: Some(AtomData::Free(FreeAtom {
                atom_type: FourCC::from(*b"free"),
                data_size: free_content_size - metadata_size,
                data: vec![0u8; free_content_size - metadata_size],
            })),
            children: Vec::new(),
        });
        if free_size - metadata_size != free_atom_bytes.len() {
            return Err(anyhow!(
                "error writing new free atom: wrong size (expected {}, got {})",
                free_size - metadata_size,
                free_atom_bytes.len()
            ));
        }
        mp4_writer
            .write_raw(&mut output_writer, &free_atom_bytes)
            .await
            .context("error writing new free atom")?;
    }
    output_writer.flush().await.context("final flush")?;

    progress_bar.finalize();

    Ok(())
}
