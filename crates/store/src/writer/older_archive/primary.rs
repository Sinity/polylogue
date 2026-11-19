//! Older inputs enter the ordinary capture spool and confirmed source stream.
use super::*;
use crate::primary::{
    CaptureMode, CapturePlan, CaptureResult, CapturedInput, InputName, InputProvenance, RawRange,
};

impl Inner {
    pub(super) fn restore_primary_older(&mut self, group: OlderGroup) -> Result<GroupOutcome> {
        self.primary_cache_ready()?;
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary writer"));
        let mut outcomes = Vec::new();
        let mut inputs = Vec::new();
        for input in &group.inputs {
            let outcome = if let Some(reason) = &input.refusal {
                Restored::Refused(reason.clone())
            } else {
                match held(&self.index, input)? {
                    Some(None) => {
                        inputs.push(input);
                        Restored::Present
                    }
                    Some(Some(reason)) => Restored::Refused(reason),
                    None => match taken_observation(&self.index, input)? {
                        Some(reason) => Restored::Refused(reason),
                        None => {
                            inputs.push(input);
                            Restored::Pending
                        }
                    },
                }
            };
            outcomes.push(outcome);
        }
        if !outcomes.contains(&Restored::Pending) {
            inputs.clear();
        }
        let pending = sinex.pending_captures()?;
        let mut members = Vec::new();
        for input in inputs {
            let rows = |revision, extents, observation| InputRows {
                acquisition: None,
                input: input.id.clone(),
                format: input.format.clone(),
                revision,
                extents,
                observation,
            };
            let name = input
                .observations
                .last()
                .map(|o| InputName {
                    origin: input.format.clone(),
                    path: o.path.clone(),
                    host: o.host.clone(),
                })
                .unwrap_or_else(|| InputName {
                    origin: input.format.clone(),
                    path: input.id.clone(),
                    host: String::new(),
                });
            let mut add = |rows: InputRows, file: Option<&PreparedBytes>| -> Result<()> {
                let metadata;
                let file = match file {
                    Some(file) => file,
                    None => {
                        metadata = self
                            .location
                            .prepare_bytes(serde_json::to_vec(&rows)?.as_slice())?;
                        &metadata
                    }
                };
                if file.owner != self.location.root {
                    return Err(StoreError::Invalid(
                        "prepared bytes belong to another archive".into(),
                    ));
                }
                let (index, previous) = match rows.extents.first() {
                    Some(extent) => self.capture_previous(&rows.input, extent.chain, &pending)?,
                    None => (0, None),
                };
                let raw_range = (!rows.extents.is_empty()).then_some(RawRange {
                    start: 0,
                    end: file.len(),
                });
                members.push((
                    CapturedInput {
                        provenance: InputProvenance::Restored,
                        input: name.clone(),
                        rows,
                        index,
                        previous,
                        hash: file.hash(),
                        length: file.len(),
                        group: None,
                        route: polylogue_model::Route::Copy,
                        raw_range,
                        generation: None,
                        raw_total: None,
                        raw_predecessor: None,
                        basis: None,
                    },
                    file.clone(),
                ));
                Ok(())
            };
            for revision in &input.revisions {
                add(rows(Some(revision.clone()), Vec::new(), None), None)?;
            }
            for extent in &input.extents {
                let file = group
                    .files
                    .iter()
                    .find(|file| file.hash() == extent.blob)
                    .or_invalid("an older input's original body was not prepared")?;
                let source_rows = rows(None, vec![extent.clone()], None);
                add(source_rows, Some(file))?;
            }
            for observation in &input.observations {
                add(rows(None, Vec::new(), Some(observation.clone())), None)?;
            }
            // Empty inputs still have an identity and format.
            if input.revisions.is_empty()
                && input.extents.is_empty()
                && input.observations.is_empty()
            {
                add(rows(None, Vec::new(), None), None)?;
            }
        }
        if !members.is_empty() {
            let headers: Vec<_> = members
                .iter()
                .map(|(header, _)| (&header.rows, header.hash, header.length))
                .collect();
            let plan = CapturePlan {
                intent: blake3::hash(&serde_json::to_vec(&headers)?)
                    .to_hex()
                    .to_string(),
                bases: Vec::new(),
                count: members.len() as u64,
                mode: CaptureMode::Immutable,
                required_outcome: None,
            };
            let mut visit = |emit: &mut crate::primary::CaptureEmitter<'_>| {
                for (header, file) in &members {
                    emit(header.clone(), vec![file.body()?])?;
                }
                Ok(())
            };
            match sinex.prepare_input_group(&plan, &mut visit)? {
                CaptureResult::Accepted { .. } => {}
                CaptureResult::UnresolvedBasisConflict { spooled } => {
                    return Err(StoreError::UnresolvedBasisConflict { spooled });
                }
            }
        }
        // Unreferenced originals also retain their ordinary material authority.
        let cited: BTreeSet<_> = members
            .iter()
            .filter(|(header, _)| header.raw_range.is_some())
            .map(|(header, _)| header.hash)
            .collect();
        for file in &group.files {
            if !cited.contains(&file.hash()) {
                sinex.prepare_blob(file.hash(), file.len(), file.body()?)?;
            }
        }
        match self.drain_primary_captures() {
            Ok(()) | Err(StoreError::Sinex(_)) => {}
            Err(error) => return Err(error),
        }
        for (input, outcome) in group.inputs.iter().zip(&mut outcomes) {
            if *outcome == Restored::Pending && held(&self.index, input)? == Some(None) {
                *outcome = Restored::Stored;
            }
        }
        // Input observations restore their machine facts. Independent machine
        // rows have no current capture representation; the rows input retains them.
        let machines = group
            .machines
            .iter()
            .map(|_| {
                Restored::Refused(
                    "independent historical machine rows remain in the rows input".into(),
                )
            })
            .collect();
        Ok(GroupOutcome {
            inputs: outcomes,
            machines,
        })
    }
}

pub(in crate::writer) fn restored_rows(conn: &Connection, rows: &InputRows) -> Result<()> {
    if rows.acquisition.is_some() {
        return Err(StoreError::Invalid(
            "restored history cannot claim a new acquisition".into(),
        ));
    }
    let input = Input {
        id: rows.input.clone(),
        format: rows.format.clone(),
        revisions: rows.revision.iter().cloned().collect(),
        extents: rows.extents.clone(),
        observations: rows.observation.iter().cloned().collect(),
        refusal: None,
        inexact_locations: 0,
    };
    if let Some(Some(reason)) = held(conn, &input)? {
        return Err(StoreError::Invalid(reason));
    }
    if let Some(reason) = taken_observation(conn, &input)? {
        return Err(StoreError::Invalid(reason));
    }
    Ok(())
}
