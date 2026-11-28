//! Optional file pointers. Selection uses the annotation query and
//! delivery history; only uncommitted coordination lives in memory.
use crate::brief::Prepared;
use crate::daemon::{query, scope::SharedScopeIndex};
use polylogue_model::user::LeftOut;
use polylogue_model::{Reference, SessionId, Timestamp, annotation::LinkRole};
use polylogue_store::{Reader, StoreError};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::time::Duration;

pub const BUDGET: Duration = Duration::from_millis(20);
const COOLDOWN_MS: i64 = 600_000;

#[derive(Default)]
pub struct Cooldowns(Mutex<HashMap<String, i64>>);
impl Cooldowns {
    pub fn available(
        &self,
        session: &SessionId,
        now: i64,
    ) -> Option<MutexGuard<'_, HashMap<String, i64>>> {
        let mut recent = match self.0.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return None,
        };
        recent.retain(|_, at| now.saturating_sub(*at) < COOLDOWN_MS);
        if recent.contains_key(&session.to_string()) {
            None
        } else {
            Some(recent)
        }
    }
}

pub fn empty() -> Prepared {
    Prepared {
        event: polylogue_model::DeliveryEvent::Advice,
        text: String::new(),
        refs: Vec::new(),
        left_out: Vec::new(),
        search_summary: None,
        empty: true,
        next: None,
    }
}

pub fn paths(prompt: &str) -> Vec<PathBuf> {
    prompt
        .split_whitespace()
        .filter_map(|word| {
            let word = word
                .trim_matches(|c: char| "`\"'()[]{},;!?".contains(c))
                .trim_end_matches('.');
            if word.contains("://")
                || !(word.contains('/') || Path::new(word).extension().is_some())
            {
                return None;
            }
            Some(PathBuf::from(word))
        })
        .collect()
}

fn normalize(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            component => out.push(component.as_os_str()),
        }
    }
    (!out.as_os_str().is_empty()).then_some(out)
}

fn matches_file(
    path: &str,
    tokens: &[PathBuf],
    cwd: &Path,
    links: &[polylogue_model::annotation::AnnotationLink],
    repos: &[String],
) -> bool {
    let Some(target) = normalize(Path::new(path)) else {
        return false;
    };
    for token in tokens {
        let Some(absolute) = normalize(&cwd.join(token)) else {
            continue;
        };
        if target.is_absolute() {
            if absolute == target {
                return true;
            }
            continue;
        }
        for link in links
            .iter()
            .filter(|link| matches!(link.role, LinkRole::Scope | LinkRole::About))
        {
            match &link.reference {
                Reference::Cwd(base) => {
                    let Some(base) = normalize(Path::new(base)) else {
                        continue;
                    };
                    if base.is_absolute()
                        && cwd.starts_with(&base)
                        && normalize(&base.join(&target)).as_ref() == Some(&absolute)
                    {
                        return true;
                    }
                }
                Reference::Repo(repo)
                    if repos.contains(repo)
                        && !token.is_absolute()
                        && normalize(token).as_ref() == Some(&target) =>
                {
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

fn already_given(
    reader: &Reader<'_>,
    session: &SessionId,
    reference: &Reference,
    target: &Reference,
    context: &Arc<polylogue_reading::operation::Context>,
) -> Result<bool, StoreError> {
    // This recipient is a typed identity supplied by the hook, not an
    // archive prefix: the first prompt can precede provider capture.
    let text = format!(
        "deliveries where event:advice bytes > 0 text(recipient) = {} refs:{} refs:{}",
        query::annotations::quoted(&session.to_string()),
        query::annotations::quoted(&reference.to_string()),
        query::annotations::quoted(&target.to_string())
    );
    let reference = reference.to_string();
    let target = target.to_string();
    let mut cursor = None;
    loop {
        let page = query::run(
            reader,
            query::Ask {
                text: &text,
                cursor: cursor.as_deref(),
                page_size: None,
                why: false,
                session: None,
                context: Arc::clone(context),
                embeddings: false,
                ranking_identity: None,
                near: None,
                fields: &["id".into()],
            },
        )
        .and_then(|answer| answer.value.into_rows())
        .map_err(|failure| query::annotations::store_error(&failure, &text))?;
        // History that cannot be read whole may hold this pair: a pair is
        // offered once, so an unknown answer offers nothing.
        if !page.incomplete.is_empty() || !page.notes.is_empty() {
            return Ok(true);
        }
        for row in page.rows {
            let id = row
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    StoreError::internal("advisory history lacks a delivery identity")
                })?;
            let refs = reader
                .delivery_refs(id)?
                .ok_or_else(|| StoreError::internal("advisory history changed while reading"))?;
            if refs.iter().any(|found| found == &reference)
                && refs.iter().any(|found| found == &target)
            {
                return Ok(true);
            }
        }
        match page.next {
            Some(next) => cursor = Some(next),
            None => return Ok(false),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare(
    reader: &Reader<'_>,
    scope: &SharedScopeIndex,
    prompt: &str,
    cwd: &Path,
    session: &SessionId,
    delivery: &str,
    pending: &[Vec<String>],
    context: &Arc<polylogue_reading::operation::Context>,
) -> Result<Prepared, StoreError> {
    let tokens = paths(prompt);
    if tokens.is_empty() {
        return Ok(empty());
    }
    let Some(cwd) = normalize(cwd).filter(|cwd| cwd.is_absolute()) else {
        return Ok(empty());
    };
    let Some(repos) =
        scope.try_read(|index| index.repos_under(&[cwd.to_string_lossy().into_owned()]))
    else {
        return Ok(Prepared {
            left_out: vec![LeftOut {
                what: "advisories".into(),
                reason: "scope index could not be read".into(),
            }],
            ..empty()
        });
    };
    let now: Timestamp = context.clock.now.into();
    // The query decides current revisions and the operator's latest judgment of a revision.
    let candidates = query::annotations::ids(
        reader,
        "(kind:decision OR kind:correction) judged:accepted current:true text(about) ~ \"file:\" | sort by created_at desc, id",
        context,
    )?;
    for reference in candidates {
        reader.check()?;
        if !reader.annotation_is_unexpired(&reference, now)? {
            continue;
        }
        let Some(note) = reader.annotation(&reference)? else {
            continue;
        };
        let about = query::annotations::about(&[Reference::Annotation(reference.clone())]);
        let judgments = query::annotations::ids(
            reader,
            &format!("kind:judgment author:operator {about} | sort by created_at desc, id desc"),
            context,
        )?;
        let Some(judgment) = judgments.first() else {
            continue;
        };
        if !reader.annotation_is_unexpired(judgment, now)? {
            continue;
        }
        reader.check()?;
        let reference = Reference::Annotation(reference);
        let stored_reference = reference.to_string();
        let mut selected = None;
        for link in &note.fact.links {
            if link.role != LinkRole::About {
                continue;
            }
            let Reference::File { path, .. } = &link.reference else {
                continue;
            };
            if matches_file(path, &tokens, &cwd, &note.fact.links, &repos)
                && !pending.iter().any(|refs| {
                    refs.contains(&stored_reference) && refs.contains(&link.reference.to_string())
                })
                && !already_given(reader, session, &reference, &link.reference, context)?
            {
                selected = Some(&link.reference);
                break;
            }
        }
        let Some(target) = selected else {
            continue;
        };
        let pointer = serde_json::json!({"target": target.to_string(), "read": format!("polylogue {} read", polylogue_render::shell_quote(&reference.to_string()))});
        let text = format!(
            "[polylogue advice delivery:{}] Operator-judged {}: {}\n",
            polylogue_render::terminal::escape(delivery),
            note.fact.kind.as_str(),
            crate::brief::archive_json(&pointer)
        );
        return Ok(Prepared {
            event: polylogue_model::DeliveryEvent::Advice,
            text,
            refs: vec![stored_reference, target.to_string()],
            left_out: Vec::new(),
            search_summary: None,
            empty: false,
            next: None,
        });
    }
    Ok(empty())
}

#[cfg(test)]
pub(crate) mod tests;
