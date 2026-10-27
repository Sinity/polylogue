//! Offline commands: Store reads and the shared Sources observer; no engine or daemon.
use crate::commands::{Command, Values};
use polylogue_sources::{
    Parse,
    schema::{self, Framing, Observer, Schema, Selection},
};
use polylogue_store::{Archive, ReadPool, Reader};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

pub fn infer(_: &Command, values: &Values, json: bool) -> ExitCode {
    run(values, json, || {
        let origin = values.text("origin").unwrap_or_default();
        let spec = polylogue_sources::origin::by_name(&origin).ok_or("unknown origin")?;
        let inputs: BTreeSet<_> = values.list("input").into_iter().collect();
        if inputs.is_empty() {
            return Err("select at least one --input".into());
        }
        let revision = values
            .integer("revision")
            .map(i64::try_from)
            .transpose()
            .map_err(|_| "revision exceeds signed range")?;
        if revision.is_some_and(|r| r < 1) {
            return Err("revision must be positive".into());
        }
        let root = values
            .text("archive")
            .map(PathBuf::from)
            .unwrap_or_else(Archive::default_root);
        let archive = match values.text("sinex_store") {
            Some(store) => Archive::on_sinex(&root, Path::new(&store)),
            None => Archive::at(&root),
        };
        let pool = ReadPool::open(&archive).map_err(|_| "cannot open stored inputs")?;
        let reader = pool
            .reader()
            .map_err(|_| "cannot read the archive or primary index")?;
        let names = values.list("field").into_iter().collect();
        let inferred = observe(&reader, spec, inputs, revision, names)?;
        serde_json::to_value(inferred).map_err(|e| e.to_string())
    })
}

fn observe(
    reader: &Reader<'_>,
    spec: &polylogue_sources::OriginSpec,
    inputs: BTreeSet<String>,
    revision: Option<i64>,
    names: BTreeSet<String>,
) -> Result<Schema, String> {
    let mut observer = Observer::new(names.clone());
    let mut selection = Vec::new();
    for input in inputs {
        let mut selected = Selection {
            input: input.clone(),
            revision,
            records: 0,
            unreadable_records: 0,
            unreadable: None,
        };
        let result = (|| {
            let stored = reader
                .schema_input(&input, revision)
                .map_err(|_| "input_read")?
                .ok_or("input_or_revision_missing")?;
            selected.revision = Some(stored.revision);
            if stored.format != spec.name {
                return Err("different_origin");
            }
            let chain = reader
                .stored_chain(&stored.input, stored.chain, stored.length)
                .map_err(|_| "stored_bytes_unavailable")?;
            let blobs = Arc::new(reader.blob_reader());
            let mut bytes = BufReader::new(chain.reader(blobs));
            let framing = if matches!(spec.capture, polylogue_sources::Capture::RowDump) {
                Framing::Lines
            } else {
                match (spec.open)(Path::new(&stored.path)).ok_or("unsupported_framing")? {
                    Parse::Lines(_) => Framing::Lines,
                    Parse::CaptureLog(_) => Framing::CaptureLog,
                    Parse::Document(_) | Parse::Export(_) => {
                        skip_space(&mut bytes).map_err(|_| "input_read")?;
                        if bytes.fill_buf().map_err(|_| "input_read")?.first() == Some(&b'[') {
                            Framing::Array
                        } else {
                            Framing::Json
                        }
                    }
                }
            };
            let mut one = Observer::new(names.clone());
            let result = one.read(bytes, framing);
            (selected.records, selected.unreadable_records) = one.counts();
            observer.merge(one)?;
            result.map_err(|_| "unreadable_input")
        })();
        if let Err(reason) = result {
            selected.unreadable = Some(reason.into());
        }
        selection.push(selected);
    }
    Ok(observer.finish(spec.name.into(), selection))
}

fn skip_space(reader: &mut impl BufRead) -> io::Result<()> {
    loop {
        let buffer = reader.fill_buf()?;
        let spaces = buffer
            .iter()
            .take_while(|b| b.is_ascii_whitespace())
            .count();
        if spaces == 0 {
            return Ok(());
        }
        reader.consume(spaces);
    }
}

pub fn diff(_: &Command, values: &Values, json: bool) -> ExitCode {
    run(values, json, || {
        let read = |name| -> Result<Schema, String> {
            let file = File::open(values.text(name).unwrap_or_default())
                .map_err(|_| format!("cannot read {name} schema"))?;
            serde_json::from_reader(BufReader::new(file))
                .map_err(|_| format!("invalid {name} schema artifact"))
        };
        let report = schema::compare(&read("before")?, &read("after")?)?;
        serde_json::to_value(report).map_err(|e| e.to_string())
    })
}

fn run(
    values: &Values,
    json: bool,
    action: impl FnOnce() -> Result<serde_json::Value, String>,
) -> ExitCode {
    let result = action().and_then(|value| {
        let mut bytes = Vec::new();
        let format = if json {
            polylogue_render::Format::Json
        } else {
            polylogue_render::Format::Terminal
        };
        polylogue_render::schema::render(&value, format, &mut bytes).map_err(|e| e.to_string())?;
        match values.text("output") {
            Some(path) => std::fs::write(path, bytes).map_err(|e| e.to_string()),
            None => io::stdout()
                .lock()
                .write_all(&bytes)
                .map_err(|e| e.to_string()),
        }
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => super::fail(json, "schema", &message, 1),
    }
}

#[cfg(test)]
mod tests;
