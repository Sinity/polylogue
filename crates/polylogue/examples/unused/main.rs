//! Reparse exact revisions selected by a live daemon, reporting only field paths
//! and counts. The archive is opened through Store's read-only interfaces.
use polylogue_model::Json;
use polylogue_sources::{Parse, Step, origin};
use polylogue_store::{Archive, ReadPool};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader, Read},
    path::Path,
    process::Command,
    sync::Arc,
};

#[derive(Deserialize)]
struct Input {
    id: String,
    format: String,
    revision: i64,
}
#[derive(Default, Serialize)]
struct Counts {
    inputs: u64,
    bytes: u64,
    records: u64,
    unreadable: u64,
    unavailable: u64,
    fields: BTreeMap<String, u64>,
}

fn live<T: serde::de::DeserializeOwned>(
    binary: &str,
    sql: &str,
) -> Result<T, Box<dyn std::error::Error>> {
    let output = Command::new(binary).args(["--json", "sql", sql]).output()?;
    if !output.status.success() {
        return Err("live read-only SQL failed".into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn add(counts: &mut Counts, step: Step) {
    counts.records += step.counts.records;
    counts.unreadable += step.counts.unreadable.values().sum::<u64>();
    for (field, count) in step.counts.unread_fields {
        *counts.fields.entry(field).or_default() += count;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let binary = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "polylogue-live".into());
    let output = std::env::args()
        .nth(2)
        .ok_or("expected live CLI and output path")?;
    let databases: Vec<Json> = live(&binary, "PRAGMA database_list")?;
    let archive = databases
        .iter()
        .find(|row| row["name"] == "a")
        .and_then(|row| row["file"].as_str())
        .and_then(|file| Path::new(file).parent())
        .ok_or("live archive unavailable")?;
    let inputs: Vec<Input> = live(
        &binary,
        "SELECT i.id,i.format,max(r.number) AS revision FROM a.input i JOIN a.revision r ON r.input=i.id GROUP BY i.id,i.format ORDER BY i.format,i.id",
    )?;
    let pool = ReadPool::open(&Archive::at(archive))?;
    let mut results: BTreeMap<String, Counts> = BTreeMap::new();
    for input in inputs {
        let counts = results.entry(input.format.clone()).or_default();
        counts.inputs += 1;
        let Some(spec) = origin::by_name(&input.format) else {
            counts.unavailable += 1;
            continue;
        };
        let reader = pool.reader()?;
        let Some(stored) = reader.schema_input(&input.id, Some(input.revision))? else {
            counts.unavailable += 1;
            continue;
        };
        let chain = match reader.stored_chain(&stored.input, stored.chain, stored.length) {
            Ok(chain) => chain,
            Err(_) => {
                counts.unavailable += 1;
                continue;
            }
        };
        let bytes = chain.reader(Arc::new(reader.blob_reader()));
        let Some(mut parser) = (spec.open)(Path::new(&stored.path)) else {
            counts.unavailable += 1;
            continue;
        };
        counts.bytes += stored.length;
        match &mut parser {
            Parse::Lines(_) => {
                let mut bytes = BufReader::new(bytes);
                let mut state = None;
                let mut offset = 0;
                let mut chunk = Vec::new();
                loop {
                    chunk.clear();
                    // The batch size only chooses when to hand complete records
                    // to the parser. A single record can exceed it.
                    while chunk.len() < 4 * 1024 * 1024 {
                        if bytes.read_until(b'\n', &mut chunk)? == 0 {
                            break;
                        }
                    }
                    if chunk.is_empty() {
                        break;
                    }
                    let (step, next) = parser.step(state.as_ref(), &chunk, offset)?;
                    offset += step.consumed;
                    add(counts, step);
                    state = Some(next);
                }
            }
            Parse::Export(export) => {
                let mut sink = Items {
                    export: &mut **export,
                    counts,
                    owners: Default::default(),
                };
                // Scan and parse use separate parser instances because scan borrows
                // its format while each item is passed to the other instance.
                let Some(Parse::Export(mut scan)) = (spec.open)(Path::new(&stored.path)) else {
                    return Err("export parser changed".into());
                };
                let mut data = Vec::new();
                BufReader::new(bytes).read_to_end(&mut data)?;
                scan.scan(&input.id, &mut std::io::Cursor::new(data), &mut sink)?;
            }
            _ => {
                let mut data = Vec::new();
                BufReader::new(bytes).read_to_end(&mut data)?;
                let (steps, _) = parser.steps(None, &data, 0)?;
                for step in steps {
                    add(counts, step);
                }
            }
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&results)?)?;
    println!("field-path audit completed for {} formats", results.len());
    Ok(())
}

struct Items<'a> {
    export: &'a mut dyn polylogue_sources::export::ExportParser,
    counts: &'a mut Counts,
    owners: polylogue_sources::export::MemoryState,
}
impl polylogue_sources::export::ItemSink for Items<'_> {
    fn push(
        &mut self,
        range: std::ops::Range<u64>,
        _: Vec<u8>,
        _: Option<&str>,
        raw: &[u8],
    ) -> Result<(), String> {
        add(
            self.counts,
            polylogue_sources::export::parse_item(self.export, raw, range, &mut self.owners)?,
        );
        Ok(())
    }
}
