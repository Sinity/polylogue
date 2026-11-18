//! Explicit conversion of preserved execution bodies. Normal readers never
//! call this; the finite tool is retired after the format transition.
use super::*;
use polylogue_model::BlobHash;
use serde_json::Value;
use std::path::Path;

fn sql(connection: &Connection, text: &str) -> Result<String> {
    Ok(crate::source::Source::of(connection)?.sql(text))
}

pub struct ConvertedRun {
    pub run: QueryRun,
    pub old_binding: BlobHash,
    pub old_run: BlobHash,
}

/// Convert only the field owned by the previous execution-options format.
/// Other variants and all identity, authorship and outcome inputs stay intact.
pub fn convert_run(mut body: Value) -> Result<ConvertedRun> {
    let mut changed = false;
    if body["binding"]["kind"] == "original" {
        let options = body["binding"]["execution"]["query_options"]
            .as_object_mut()
            .ok_or_else(|| StoreError::invalid("execution has no query options"))?;
        if let Some(reference) = options.remove("direct_reference") {
            if options.contains_key("references") {
                return Err(StoreError::invalid(
                    "execution contains both population formats",
                ));
            }
            if !reference.is_null() && !reference.is_string() {
                return Err(StoreError::invalid(
                    "old direct reference is not a reference or null",
                ));
            }
            options.insert(
                "references".into(),
                if reference.is_null() {
                    Value::Null
                } else {
                    Value::Array(vec![reference])
                },
            );
            changed = true;
        }
    }
    let run: QueryRun = serde_json::from_value(body)?;
    super::validate_run(&run)?;
    let mut previous = serde_json::to_value(&run)?;
    if changed {
        let options = previous["binding"]["execution"]["query_options"]
            .as_object_mut()
            .unwrap();
        let references = options.remove("references").unwrap();
        options.insert(
            "direct_reference".into(),
            match references {
                Value::Null => Value::Null,
                Value::Array(mut references) if references.len() == 1 => references.remove(0),
                _ => {
                    return Err(StoreError::invalid(
                        "old population did not contain zero or one reference",
                    ));
                }
            },
        );
    }
    let tag = if previous["binding"]["kind"] == "original_object" {
        "polylogue-object-execution-binding-v1"
    } else {
        "polylogue-execution-binding-v1"
    };
    let hash = |tag, value: &Value| {
        canonical_json::canonical_hash(tag, value)
            .map(|hash| BlobHash(*hash.as_bytes()))
            .map_err(crate::encoding::identity_encoding)
    };
    Ok(ConvertedRun {
        old_binding: hash(tag, &previous["binding"]["execution"])?,
        old_run: hash("polylogue-query-run-begin-v1", &previous)?,
        run,
    })
}
impl ConvertedRun {
    pub fn completion(&self, mut completion: QueryRunOutcome) -> Result<QueryRunOutcome> {
        if completion.run != self.run.id
            || completion.binding_hash != self.old_binding
            || completion.run_hash != self.old_run
        {
            return Err(StoreError::Conflict(completion.run.to_string()));
        }
        completion.binding_hash = self.run.binding_hash()?;
        completion.run_hash = self.run.record_hash()?;
        super::validate_run_outcome(&completion)?;
        Ok(completion)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    Register,
    Products,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Conversion {
    pub changed_runs: u64,
    pub output: OutputKind,
}

/// Convert a real register to the current register, or extract current products
/// from a preserved cache. Its derived tables never become a current cache.
/// The original complete DB/WAL set stays unchanged under its existing lock.
pub fn convert_preserved(
    source: &Path,
    destination: &Path,
    pending: impl FnOnce(&rusqlite::Transaction<'_>) -> Result<()>,
) -> Result<Conversion> {
    use rusqlite::{
        OpenFlags,
        backup::{Backup, StepResult},
    };
    use std::io::Read;
    fn digest(path: &Path) -> Result<blake3::Hash> {
        let mut file = std::fs::File::open(path)?;
        let mut hash = blake3::Hasher::new();
        let mut bytes = [0; 65536];
        loop {
            let n = file.read(&mut bytes)?;
            if n == 0 {
                break;
            }
            hash.update(&bytes[..n]);
        }
        Ok(hash.finalize())
    }
    let source_parent = source
        .parent()
        .ok_or_else(|| StoreError::invalid("source needs its preserved archive directory"))?;
    let source_root = source_parent.canonicalize()?;
    let destination_parent = destination
        .parent()
        .ok_or_else(|| StoreError::invalid("destination needs a parent"))?
        .canonicalize()?;
    if destination_parent.starts_with(&source_root) {
        return Err(StoreError::invalid(
            "the converted register must be outside the preserved input directory",
        ));
    }
    let custody = std::fs::File::open(source_parent.join("lock"))?;
    custody.try_lock_shared().map_err(|error| {
        StoreError::invalid(format!("preserved archive is still open: {error}"))
    })?;
    // The existing archive lock fixes the complete DB/WAL set. SQLite reads
    // a private copy so acquiring its WAL reader cannot alter preserved files.
    let input = tempfile::tempdir_in(
        destination
            .parent()
            .ok_or_else(|| StoreError::invalid("destination needs a parent"))?,
    )?;
    let snapshot = input.path().join("source.db");
    std::fs::copy(source, &snapshot)?;
    for suffix in ["-wal", "-shm"] {
        let member = std::path::PathBuf::from(format!("{}{suffix}", source.display()));
        if member.exists() {
            std::fs::copy(&member, input.path().join(format!("source.db{suffix}")))?;
        }
    }
    let original = Connection::open_with_flags(snapshot, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let identity: i32 = original.pragma_query_value(None, "application_id", |row| row.get(0))?;
    if identity != crate::schema::FORMAT_ID {
        return Err(StoreError::invalid(
            "preserved input is not a Polylogue register or cache",
        ));
    }
    let version: u32 = original.pragma_query_value(None, "user_version", |row| row.get(0))?;
    let primary = matches!(
        crate::source::Source::of(&original)?,
        crate::source::Source::Cache
    );
    let target = match (version, primary) {
        (3 | 4, false) => OutputKind::Register,
        (8, true) => OutputKind::Products,
        _ => {
            return Err(StoreError::invalid(
                "input is outside the execution-body transition",
            ));
        }
    };
    let parent = destination
        .parent()
        .ok_or_else(|| StoreError::invalid("destination needs a parent"))?;
    let staged = tempfile::NamedTempFile::new_in(parent)?;
    let mut output = Connection::open(staged.path())?;
    {
        let backup = Backup::new(&original, &mut output)?;
        loop {
            match backup.step(128)? {
                StepResult::Done => break,
                StepResult::More => {}
                _ => {
                    return Err(StoreError::invalid(
                        "preserved input is busy; freeze its snapshot first",
                    ));
                }
            }
        }
    }
    let tx = output.transaction()?;
    pending(&tx)?;
    let mut changed = 0;
    {
        let mut statement = tx.prepare(&sql(
            &tx,
            "SELECT id,binding_hash,run_hash,run,completion FROM @source.query_run ORDER BY id",
        )?)?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let binding: String = row.get(1)?;
            let intent: String = row.get(2)?;
            let body: Option<Vec<u8>> = row.get(3)?;
            let completion: Option<Vec<u8>> = row.get(4)?;
            let body=body.ok_or_else(||StoreError::invalid(format!("execution {id} has no original begin; locate its preserved register or spool body before conversion")))?;
            let converted = convert_run(crate::encoding::unpack_json(&body)?)?;
            if converted.run.id.as_str() != id
                || converted.old_binding.to_string() != binding
                || converted.old_run.to_string() != intent
            {
                return Err(StoreError::Conflict(id));
            }
            let completion = completion
                .map(|body| converted.completion(crate::encoding::unpack_json(&body)?))
                .transpose()?
                .map(|completion| super::pack(&completion))
                .transpose()?;
            let new_binding = converted.run.binding_hash()?.to_string();
            let new_intent = converted.run.record_hash()?.to_string();
            changed += u64::from(new_binding != binding || new_intent != intent);
            tx.execute(
                &sql(&tx, "UPDATE @source.query_run SET binding_hash=?2,run_hash=?3,run=?4,completion=?5 WHERE id=?1")?,
                params![
                    id,
                    new_binding,
                    new_intent,
                    super::pack(&converted.run)?,
                    completion
                ],
            )?;
        }
    }
    if target == OutputKind::Register {
        tx.execute_batch(crate::schema::ARCHIVE_RESULT_INDEX)?;
        tx.pragma_update(None, "user_version", crate::schema::ARCHIVE_VERSION)?;
    }
    tx.commit()?;
    let staged = match target {
        OutputKind::Register => {
            crate::schema::ensure_archive(&output)?;
            output.close().map_err(|(_, error)| error)?;
            staged
        }
        OutputKind::Products => {
            use std::io::Write;
            let mut products = tempfile::NamedTempFile::new_in(parent)?;
            visit_products(&output, |product| {
                serde_json::to_writer(&mut products, &product)?;
                products.write_all(b"\n")?;
                Ok(())
            })?;
            output.close().map_err(|(_, error)| error)?;
            products
        }
    };
    staged.as_file().sync_all()?;
    if destination.exists() {
        if digest(staged.path())? != digest(destination)? {
            return Err(StoreError::Conflict(destination.display().to_string()));
        }
        return Ok(Conversion {
            changed_runs: 0,
            output: target,
        });
    }
    staged
        .persist_noclobber(destination)
        .map_err(|error| error.error)?;
    std::fs::File::open(parent)?.sync_all()?;
    Ok(Conversion {
        changed_runs: changed,
        output: target,
    })
}

/// Add preserved query products to the staged register before conversion.
/// Their original bodies and hashes remain intact until the one transaction
/// converts every begin and its completion together.
pub fn stage_products(connection: &rusqlite::Transaction<'_>, writes: &[Value]) -> Result<()> {
    for write in writes {
        let product = &write["product"];
        match product["product"].as_str() {
            Some("query") => {
                let object: QueryObject = serde_json::from_value(product["object"].clone())?;
                if !super::known_object(connection, &object)? {
                    let counter = crate::archive::bump_counter(connection)?;
                    super::apply_object(connection, &object, counter)?;
                }
            }
            Some("query_run") => {
                let body = &product["run"];
                let converted = convert_run(body.clone())?;
                let id = converted.run.id.as_str();
                let hashes: Option<(String, String)> = connection
                    .query_row(
                        &sql(
                            connection,
                            "SELECT binding_hash,run_hash FROM @source.query_run WHERE id=?1",
                        )?,
                        [id],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()?;
                let binding = converted.old_binding.to_string();
                let intent = converted.old_run.to_string();
                if hashes.is_some_and(|hashes| hashes != (binding.clone(), intent.clone())) {
                    return Err(StoreError::Conflict(id.into()));
                }
                let object = match &converted.run.binding {
                    RunBinding::OriginalObject { execution } => {
                        Some(execution.reference.stored_form())
                    }
                    _ => None,
                };
                connection.execute(&sql(connection, "INSERT INTO @source.query_run(id,binding_hash,run_hash,query,object_ref,run) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(id) DO UPDATE SET run=coalesce(run,excluded.run),query=coalesce(query,excluded.query),object_ref=coalesce(object_ref,excluded.object_ref)")?,
                    params![id,binding,intent,converted.run.question.map(|question|question.to_string()),object,super::pack(body)?])?;
            }
            Some("query_run_outcome") => {
                let completion: QueryRunOutcome =
                    serde_json::from_value(product["outcome"].clone())?;
                super::validate_run_outcome(&completion)?;
                let id = completion.run.as_str();
                let known: Option<(String, String, Option<Vec<u8>>)> = connection.query_row(
                        &sql(connection, "SELECT binding_hash,run_hash,completion FROM @source.query_run WHERE id=?1")?,
                        [id],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()?;
                if let Some((binding, intent, body)) = known
                    && (binding != completion.binding_hash.to_string()
                        || intent != completion.run_hash.to_string()
                        || body
                            .map(|body| crate::encoding::unpack_json::<QueryRunOutcome>(&body))
                            .transpose()?
                            .is_some_and(|prior| prior != completion))
                {
                    return Err(StoreError::Conflict(id.into()));
                }
                connection.execute(&sql(connection, "INSERT INTO @source.query_run(id,binding_hash,run_hash,completion) VALUES(?1,?2,?3,?4) ON CONFLICT(id) DO UPDATE SET completion=coalesce(completion,excluded.completion)")?,
                    params![id,completion.binding_hash.to_string(),completion.run_hash.to_string(),super::pack(&completion)?])?;
            }
            Some(_) => {
                return Err(StoreError::invalid(
                    "preserved execution input contains a non-query product",
                ));
            }
            None => {
                return Err(StoreError::invalid(
                    "preserved product has no declared variant",
                ));
            }
        }
    }
    Ok(())
}

/// Visit converted question and execution records in dependency order. The
/// destination is a private conversion result, never the preserved register.
pub fn visit_current_products(
    path: &Path,
    visit: impl FnMut(crate::primary::Product) -> Result<()>,
) -> Result<u64> {
    let connection = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let version = crate::schema::version(&connection)?;
    let identity: i32 = connection.pragma_query_value(None, "application_id", |row| row.get(0))?;
    if identity != crate::schema::FORMAT_ID
        || version != crate::schema::ARCHIVE_VERSION
        || matches!(
            crate::source::Source::of(&connection)?,
            crate::source::Source::Cache
        )
    {
        return Err(StoreError::invalid(
            "only a current converted register can be reconstructed",
        ));
    }
    visit_products(&connection, visit)
}

fn visit_products(
    connection: &Connection,
    mut visit: impl FnMut(crate::primary::Product) -> Result<()>,
) -> Result<u64> {
    let mut count = 0;
    let mut objects = connection.prepare(&sql(
        connection,
        "SELECT hash,object FROM @source.query ORDER BY hash",
    )?)?;
    let mut rows = objects.query([])?;
    while let Some(row) = rows.next()? {
        let identity: String = row.get(0)?;
        let object: QueryObject = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(1)?)?;
        super::validate_integrity(&object)?;
        if object.id.to_string() != identity {
            return Err(StoreError::Conflict(identity));
        }
        visit(crate::primary::Product::Query {
            object: Box::new(object),
        })?;
        count += 1;
    }
    let mut runs = connection.prepare(&sql(
        connection,
        "SELECT id,binding_hash,run_hash,run,completion FROM @source.query_run ORDER BY id",
    )?)?;
    let mut rows = runs.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let body: Option<Vec<u8>> = row.get(3)?;
        let run: QueryRun = crate::encoding::unpack_json(
            &body.ok_or_else(|| StoreError::invalid("converted execution has no begin"))?,
        )?;
        super::validate_run(&run)?;
        if run.id.as_str() != id
            || run.binding_hash()?.to_string() != row.get::<_, String>(1)?
            || run.record_hash()?.to_string() != row.get::<_, String>(2)?
        {
            return Err(StoreError::Conflict(id));
        }
        let binding = run.binding_hash()?;
        let intent = run.record_hash()?;
        let completion: Option<Vec<u8>> = row.get(4)?;
        visit(crate::primary::Product::QueryRun { run: Box::new(run) })?;
        count += 1;
        if let Some(body) = completion {
            let outcome: QueryRunOutcome = crate::encoding::unpack_json(&body)?;
            super::validate_run_outcome(&outcome)?;
            if outcome.run.as_str() != id
                || outcome.binding_hash != binding
                || outcome.run_hash != intent
            {
                return Err(StoreError::Conflict(id));
            }
            visit(crate::primary::Product::QueryRunOutcome {
                outcome: Box::new(outcome),
            })?;
            count += 1;
        }
    }
    Ok(count)
}
