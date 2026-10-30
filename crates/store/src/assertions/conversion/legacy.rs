use super::*;
use polylogue_model::{
    authorship::Authorship,
    historical::{
        AcceptedHashRule, ConservedEffects, HistoricalAssertionEvidence, HistoricalVariantIntent,
        Intent, NoteIntent, NoteKind, RecordedKind, RecordedRelation, Relation,
    },
    user::UserWrite,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub id: String,
    pub machine: String,
    pub at: Option<Timestamp>,
    pub authorship: Option<Authorship>,
    pub write: Write,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum Write {
    Retired(Action),
    Current(UserWrite),
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "write", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Action {
    AddNote {
        note: NewNote,
    },
    SaveQuery {
        note: NewNote,
    },
    AnnotateQuery {
        query: String,
        note: NewNote,
    },
    Withdraw {
        note: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    SetSetting {
        key: String,
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        author: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        machine: Option<String>,
    },
    Variant {
        variant: Variant,
    },
}

// Field order and omission are part of the historical semantic hash rule.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NewNote {
    kind: NoteKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    targets: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    relations: Vec<(Relation, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    revises: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    valid_until: Option<Timestamp>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Variant {
    base: Option<String>,
    source: String,
    source_hash: BlobHash,
    kind: polylogue_model::VariantKind,
    from_lang: Option<String>,
    to_lang: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    via: Option<String>,
    method: Option<String>,
    parts: Vec<polylogue_model::NewVariantPart>,
}

impl Request {
    pub fn intent_bytes(&self) -> Result<(Vec<u8>, AcceptedHashRule)> {
        let mut write = self.write.clone();
        match &mut write {
            Write::Retired(Action::AnnotateQuery { note, .. }) => note.targets.clear(),
            Write::Retired(Action::SaveQuery { note }) => note.revises = None,
            Write::Current(current) => *current = current.intent(),
            _ => {}
        }
        Ok(match &self.authorship {
            Some(author) => (
                serde_json::to_vec(&(&self.machine, author, write))?,
                AcceptedHashRule::MachineAuthorshipAndIntent,
            ),
            None => (
                serde_json::to_vec(&(&self.machine, write))?,
                AcceptedHashRule::MachineAndIntent,
            ),
        })
    }
}
impl Request {
    pub(super) fn source3_payload(&self) -> Result<serde_json::Value> {
        let at = self
            .at
            .ok_or_else(|| invalid("retained native request has no recorded time"))?;
        let author = &self
            .authorship
            .as_ref()
            .ok_or_else(|| invalid("retained native request has no recorded authorship"))?
            .author;
        let mut value = serde_json::json!({"request":self.id,"at":at,"machine":self.machine});
        match &self.write {
            Write::Retired(
                Action::AddNote { note }
                | Action::SaveQuery { note }
                | Action::AnnotateQuery { note, .. },
            ) => {
                value["note"] = self.id.clone().into();
                value["kind"] = note.kind.as_str().into();
                value["author"] = author.to_string().into();
                value["value"] = serde_json::to_value(&note.value)?;
                value["key"] = serde_json::to_value(&note.key)?;
                value["targets"] = serde_json::to_value(&note.targets)?;
                value["revises"] = serde_json::to_value(&note.revises)?;
            }
            Write::Retired(Action::Withdraw { note, reason }) => {
                value["note"] = self.id.clone().into();
                value["kind"] = "withdrawal".into();
                value["author"] = author.to_string().into();
                value["withdraws"] = note.clone().into();
                value["value"] = serde_json::to_value(reason)?;
            }
            Write::Retired(Action::SetSetting {
                key,
                value: setting,
                machine,
                ..
            }) => {
                value["setting"] = self.id.clone().into();
                value["key"] = key.clone().into();
                value["value"] = setting.clone().into();
                value["author"] = author.to_string().into();
                value["machine"] = machine.clone().unwrap_or_else(|| "*".into()).into();
            }
            Write::Current(UserWrite::Deliver(delivery) | UserWrite::DeliveredAdvice(delivery)) => {
                value["delivery"] = self.id.clone().into();
                value["recipient_kind"] = delivery.recipient_kind.clone().into();
                value["recipient_id"] = serde_json::to_value(&delivery.recipient_id)?;
                value["sender"] = delivery.sender.clone().into();
                value["event"] = delivery.event.clone().into();
                value["cwd"] = serde_json::to_value(&delivery.cwd)?;
                value["content_hash"] = blake3::hash(delivery.text.as_bytes())
                    .to_hex()
                    .to_string()
                    .into();
                value["bytes"] = delivery.text.len().into();
                value["refs"] = serde_json::to_value(&delivery.refs)?;
            }
            Write::Current(UserWrite::DeliveryEvent {
                delivery,
                what,
                detail,
            }) => {
                value["delivery"] = delivery.clone().into();
                value["what"] = what.as_str().into();
                value["detail"] = serde_json::to_value(detail)?;
            }
            Write::Current(UserWrite::KeepResult {
                name,
                query,
                rows_json,
                read_at,
            }) => {
                value["result"] = self.id.clone().into();
                value["name"] = serde_json::to_value(name)?;
                value["query"] = query.clone().into();
                value["rows"] = serde_json::from_str(rows_json)?;
                value["read_at"] = serde_json::from_str(read_at)?;
            }
            _ => {}
        }
        Ok(value)
    }
}

impl NewNote {
    fn historical(&self) -> NoteIntent {
        NoteIntent {
            kind: self.kind,
            key: self.key.clone(),
            value: self.value.clone(),
            targets: self.targets.clone(),
            relations: self.relations.clone(),
            revises: self.revises.clone(),
            valid_until: self.valid_until,
            claimed_author: self.author.clone(),
        }
    }
}
impl Write {
    pub(super) fn is_predecessor(&self) -> bool {
        !matches!(
            self,
            Self::Current(
                UserWrite::Assert { .. }
                    | UserWrite::AssertQuery { .. }
                    | UserWrite::ReviseAssertions { .. }
                    | UserWrite::WithdrawAssertions { .. }
                    | UserWrite::SaveQuery { .. }
                    | UserWrite::SetSetting { .. }
            )
        )
    }
    pub fn source3_kind(&self) -> Option<&'static str> {
        match self {
            Self::Retired(
                Action::AddNote { note }
                | Action::SaveQuery { note }
                | Action::AnnotateQuery { note, .. },
            ) if note.author.is_none() => Some("polylogue.note"),
            Self::Retired(Action::Withdraw { .. }) => Some("polylogue.note"),
            Self::Retired(Action::SetSetting { author: None, .. }) => Some("polylogue.setting"),
            Self::Retired(Action::Variant { variant })
                if variant.author.is_none() && variant.via.is_none() =>
            {
                Some("polylogue.request")
            }
            Self::Retired(_) => None,
            Self::Current(
                UserWrite::Assert { .. }
                | UserWrite::AssertQuery { .. }
                | UserWrite::ReviseAssertions { .. }
                | UserWrite::WithdrawAssertions { .. }
                | UserWrite::SaveQuery { .. }
                | UserWrite::SetSetting { .. },
            ) => None,
            Self::Current(UserWrite::KeepResult { .. }) => Some("polylogue.kept_result"),
            Self::Current(UserWrite::Deliver(_) | UserWrite::DeliveredAdvice(_)) => {
                Some("polylogue.delivery")
            }
            Self::Current(UserWrite::DeliveryEvent { .. }) => Some("polylogue.delivery_outcome"),
            Self::Current(_) => Some("polylogue.request"),
        }
    }
    pub fn current(&self) -> Option<UserWrite> {
        match self {
            Self::Current(write) => Some(write.clone()),
            Self::Retired(Action::Variant { variant })
                if variant.author.is_none() && variant.via.is_none() =>
            {
                Some(UserWrite::Variant {
                    variant: polylogue_model::NewVariant {
                        base: variant.base.clone(),
                        source: variant.source.clone(),
                        source_hash: variant.source_hash,
                        kind: variant.kind,
                        from_lang: variant.from_lang.clone(),
                        to_lang: variant.to_lang.clone(),
                        method: variant.method.clone(),
                        parts: variant.parts.clone(),
                    },
                })
            }
            _ => None,
        }
    }
    pub fn historical(&self) -> Intent {
        match self {
            Self::Retired(Action::AddNote { note }) => Intent::Note {
                note: note.historical(),
            },
            Self::Retired(Action::SaveQuery { note }) => Intent::SavedQuery {
                note: note.historical(),
            },
            Self::Retired(Action::AnnotateQuery { query, note }) => Intent::QueryAnnotation {
                query: query.clone(),
                note: note.historical(),
            },
            Self::Retired(Action::Withdraw { note, reason }) => Intent::Withdrawal {
                note: note.clone(),
                reason: reason.clone(),
            },
            Self::Retired(Action::SetSetting {
                key,
                value,
                author,
                machine,
            }) => Intent::Setting {
                key: key.clone(),
                value: value.clone(),
                machine: machine.clone(),
                claimed_author: author.clone(),
            },
            Self::Retired(Action::Variant { variant }) => Intent::Variant {
                variant: HistoricalVariantIntent {
                    base: variant.base.clone(),
                    source: variant.source.clone(),
                    source_hash: variant.source_hash,
                    kind: variant.kind,
                    from_lang: variant.from_lang.clone(),
                    to_lang: variant.to_lang.clone(),
                    claimed_author: variant.author.clone(),
                    claimed_via: variant.via.clone(),
                    method: variant.method.clone(),
                    parts: variant.parts.clone(),
                },
            },
            Self::Current(write) => Intent::Other {
                write: Box::new(write.clone()),
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Note {
    id: String,
    first_note: String,
    kind: RecordedKind,
    key: Option<String>,
    value: Option<String>,
    author: String,
    created_at: Timestamp,
    valid_until: Option<Timestamp>,
    targets: Vec<String>,
    #[serde(default)]
    relations: Vec<(RecordedRelation, String)>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Revision {
    note: Note,
    machine: String,
    revises: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Withdrawal {
    id: String,
    first_note: String,
    machine: String,
    author: String,
    created_at: Timestamp,
    reason: Option<String>,
    withdraws: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Setting {
    id: String,
    machine: String,
    key: String,
    value: String,
    author: String,
    set_at: Timestamp,
}

pub(super) fn effects(
    source: &SourceSchema,
    request: Option<&Request>,
    bytes: &[u8],
) -> Result<ConservedEffects> {
    let mut value: serde_json::Value = crate::encoding::read_json(bytes)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("historical effect must be an object"))?;
    if object
        .get("assertions")
        .is_some_and(|value| value.as_array().is_none_or(|facts| !facts.is_empty()))
    {
        return Err(invalid(
            "the predecessor effect cannot contain current-schema assertion facts",
        ));
    }
    let notes: Vec<Revision> = serde_json::from_value(
        object
            .remove("notes")
            .unwrap_or_else(|| serde_json::json!([])),
    )?;
    let withdrawals: Vec<Withdrawal> = serde_json::from_value(
        object
            .remove("withdrawals")
            .unwrap_or_else(|| serde_json::json!([])),
    )?;
    let settings: Vec<Setting> = serde_json::from_value(
        object
            .remove("settings")
            .unwrap_or_else(|| serde_json::json!([])),
    )?;
    object
        .entry("assertions")
        .or_insert_with(|| serde_json::json!([]));
    let current: AuthoredEffect = serde_json::from_value(value)?;
    let mut historical_assertions = Vec::new();
    for revision in notes {
        let note = revision.note;
        let mut fact = HistoricalAssertionEvidence {
            id: note.id,
            root: note.first_note,
            source: source.clone(),
            kind: note.kind,
            key: note.key,
            value: note.value,
            recorded_author: Some(note.author),
            machine: Some(revision.machine),
            created_at: Some(note.created_at),
            valid_until: note.valid_until,
            targets: note.targets,
            relations: note.relations,
            links: Vec::new(),
            predecessor: None,
        };
        for target in &fact.targets {
            if let Some(reference) = recorded_reference(target) {
                fact.links.push(AssertionLink {
                    role: LinkRole::About,
                    reference,
                    seen_hash: None,
                });
            }
        }
        for previous in revision.revises {
            let reference = old_exact(&fact.root, &previous)?;
            fact.links.push(AssertionLink {
                role: LinkRole::Revises,
                reference,
                seen_hash: None,
            });
            fact.relations
                .push((RecordedRelation::Known(Relation::Revises), previous));
        }
        // Unresolved predecessor spellings remain declared historical evidence;
        // local arrival order cannot turn them into different immutable links.
        for (role, other) in &fact.relations {
            if let RecordedRelation::Known(role) = role {
                let role = match role {
                    Relation::Answers => Some(LinkRole::Answers),
                    Relation::Challenges => Some(LinkRole::Challenges),
                    Relation::DerivedFrom => Some(LinkRole::DerivedFrom),
                    Relation::Blocks => Some(LinkRole::Blocks),
                    Relation::Corrects => Some(LinkRole::Corrects),
                    Relation::MemberOf => Some(LinkRole::MemberOf),
                    Relation::SlotOf => Some(LinkRole::SlotOf),
                    _ => None,
                };
                if let (Some(role), Some(reference)) = (role, recorded_reference(other)) {
                    fact.links.push(AssertionLink {
                        role,
                        reference,
                        seen_hash: None,
                    });
                }
            }
        }
        canonical_links(&mut fact)?;
        historical_assertions.push(fact);
    }
    for withdrawal in withdrawals {
        let mut fact = HistoricalAssertionEvidence {
            id: withdrawal.id,
            root: withdrawal.first_note,
            source: source.clone(),
            kind: RecordedKind::Known(NoteKind::Withdrawal),
            key: None,
            value: withdrawal.reason,
            recorded_author: Some(withdrawal.author),
            machine: Some(withdrawal.machine),
            created_at: Some(withdrawal.created_at),
            valid_until: None,
            targets: vec![],
            relations: vec![],
            links: vec![],
            predecessor: None,
        };
        for target in withdrawal.withdraws {
            let reference = if let Some(root) = target.strip_prefix("note:") {
                if root.is_empty() {
                    return Err(invalid("historical withdrawal target is empty"));
                }
                Reference::Assertion(RevisionReference {
                    root: root.into(),
                    revision: None,
                })
            } else {
                Reference::parse(&target).map_err(|error| invalid(&error.to_string()))?
            };
            fact.links.push(AssertionLink {
                role: LinkRole::Withdraws,
                reference,
                seen_hash: None,
            });
            fact.relations
                .push((RecordedRelation::Known(Relation::Withdraws), target));
        }
        canonical_links(&mut fact)?;
        historical_assertions.push(fact);
    }
    for setting in settings {
        historical_assertions.push(HistoricalAssertionEvidence {
            id: setting.id.clone(),
            root: setting.id,
            source: source.clone(),
            kind: RecordedKind::Known(NoteKind::Setting),
            key: Some(setting.key),
            value: Some(setting.value),
            recorded_author: Some(setting.author),
            machine: request.map(|request| request.machine.clone()),
            created_at: Some(setting.set_at),
            valid_until: None,
            targets: vec![],
            relations: vec![],
            links: vec![],
            predecessor: Some(serde_json::json!({"machine":setting.machine})),
        });
    }
    Ok(ConservedEffects {
        current,
        historical_assertions,
    })
}

fn recorded_reference(text: &str) -> Option<Reference> {
    // Request identities were package metadata, not predecessor target syntax.
    // Keep that original spelling as historical evidence instead of rebinding it.
    Reference::parse(text)
        .ok()
        .filter(|reference| !matches!(reference, Reference::Request(_)))
}

fn old_exact(root: &str, value: &str) -> Result<Reference> {
    if let Some(id) = value.strip_prefix("note:") {
        if id.is_empty() {
            return Err(invalid("historical revision identity is empty"));
        }
        return Ok(Reference::Assertion(RevisionReference::exact(
            root.into(),
            id.into(),
        )));
    }
    match Reference::parse(value) {
        Ok(Reference::Assertion(revision)) if revision.revision.is_some() => {
            Ok(Reference::Assertion(revision))
        }
        _ => Err(invalid(
            "historical revision relation has no exact assertion identity",
        )),
    }
}
fn canonical_links(fact: &mut HistoricalAssertionEvidence) -> Result<()> {
    fact.links
        .sort_by_cached_key(|link| (link.role, link.reference.to_string()));
    fact.links.dedup();
    if fact.id.is_empty() || fact.root.is_empty() {
        return Err(invalid("historical assertion identity is empty"));
    }
    Ok(())
}
pub(super) fn fact_references(effects: &ConservedEffects) -> Vec<Reference> {
    effects
        .current
        .assertions
        .iter()
        .map(|fact| {
            Reference::Assertion(RevisionReference::exact(fact.root.clone(), fact.id.clone()))
        })
        .chain(effects.historical_assertions.iter().map(|fact| {
            Reference::Assertion(RevisionReference::exact(fact.root.clone(), fact.id.clone()))
        }))
        .chain(
            effects
                .current
                .kept_results
                .iter()
                .map(|fact| Reference::Result {
                    id: fact.id.clone(),
                    row: None,
                }),
        )
        .chain(
            effects
                .current
                .deliveries
                .iter()
                .map(|fact| Reference::Delivery(fact.delivery.id.clone())),
        )
        .collect()
}
