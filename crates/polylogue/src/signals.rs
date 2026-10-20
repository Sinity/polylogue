//! Signals (spec user-data.md, Signals): a saved watch with a delivery
//! rule. A rule is a note of kind `signal` whose value is a JSON object:
//!
//! ```json
//! {"name": "agent waiting", "query": "sessions where live:waiting",
//!  "fires": "on_new", "for": "20m", "deliver": "notice",
//!  "interrupt": true, "quiet_hours": [22, 7],
//!  "rate_limit": {"count": 1, "window": "1h"}}
//! ```
//!
//! The evaluator follows the joint reference
//! (`polylogue-sinex-joint/prototype/signals/`): each rule fires once per
//! matching row; `for` holds a row until it has matched that long, judged
//! against the clock, so a deadline fires with no new data and a row that
//! stops matching first fires nothing; an interrupting notice waits out
//! quiet hours; past the rate limit a notice waits for the next briefing;
//! agent-written rules share one interrupt budget, past which their
//! interrupts become plain notices. Every handoff is a `delivery` record,
//! so what fired survives restarts and index rebuilds.

use crate::daemon::service::{Daemon, now_ms};
use polylogue_model::{Author, Note, NoteKind};
use polylogue_store::{Delivery, UserRequest, UserWrite};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

const MIN: i64 = 60_000;
const HOUR: i64 = 60 * MIN;

/// What a rule does when it fires. Only notices are built; the other
/// deliveries are refused by name when a rule is read.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    /// The rule note's revision family, so a revision keeps the rule.
    pub id: String,
    pub name: String,
    pub query: String,
    pub for_ms: i64,
    pub interrupt: bool,
    /// Local hours `[from, to)`, wrapping midnight when `from > to`.
    pub quiet_hours: Option<(u8, u8)>,
    pub rate_limit: Option<(usize, i64)>,
    pub operator: bool,
}

/// A rule note that could not be used, with the reason.
#[derive(Debug, Clone, PartialEq)]
pub struct Unusable {
    pub id: String,
    pub reason: String,
}

/// A duration such as `20m`, `1h`, `90s` or `2d`.
pub fn duration(text: &str) -> Option<i64> {
    let text = text.trim();
    let split = text.find(|c: char| !c.is_ascii_digit())?;
    let (n, unit) = text.split_at(split);
    let n: i64 = n.parse().ok()?;
    Some(
        n * match unit {
            "s" => 1_000,
            "m" => MIN,
            "h" => HOUR,
            "d" => 24 * HOUR,
            _ => return None,
        },
    )
}

/// Reads a `signal` note into a rule.
pub fn rule(note: &Note) -> Result<Rule, Unusable> {
    let unusable = |reason: &str| Unusable {
        id: note.root.clone(),
        reason: reason.to_owned(),
    };
    if note.kind != NoteKind::Signal {
        return Err(unusable("not a signal note"));
    }
    let v: Value = note
        .value
        .as_deref()
        .and_then(|v| serde_json::from_str(v).ok())
        .ok_or_else(|| unusable("the value is not a JSON object"))?;
    let text = |k: &str| v.get(k).and_then(Value::as_str);
    let query = text("query").ok_or_else(|| unusable("no query"))?;
    match text("fires").unwrap_or("on_new") {
        "on_new" => {}
        "on_change" | "on_threshold" => {
            return Err(unusable(
                "fires on_change and on_threshold are not built yet",
            ));
        }
        other => return Err(unusable(&format!("unknown fires: {other}"))),
    }
    match text("deliver").unwrap_or("notice") {
        "notice" => {}
        "value" | "action" => {
            return Err(unusable("deliver value and action are not built yet"));
        }
        other => return Err(unusable(&format!("unknown deliver: {other}"))),
    }
    let for_ms = match text("for") {
        Some(d) => duration(d).ok_or_else(|| unusable("for is a duration such as 20m"))?,
        None => 0,
    };
    let quiet_hours = match v.get("quiet_hours") {
        Some(q) => {
            let pair = q
                .as_array()
                .filter(|a| a.len() == 2)
                .and_then(|a| Some((a[0].as_u64()?, a[1].as_u64()?)))
                .filter(|(a, b)| *a < 24 && *b < 24)
                .ok_or_else(|| unusable("quiet_hours is [from, to] in hours"))?;
            Some((pair.0 as u8, pair.1 as u8))
        }
        None => None,
    };
    let rate_limit = match v.get("rate_limit") {
        Some(r) => {
            let count = r.get("count").and_then(Value::as_u64).filter(|c| *c >= 1);
            let window = r.get("window").and_then(Value::as_str).and_then(duration);
            match (count, window) {
                (Some(c), Some(w)) if w > 0 => Some((c as usize, w)),
                _ => return Err(unusable("rate_limit is {count, window}")),
            }
        }
        None => None,
    };
    Ok(Rule {
        id: note.root.clone(),
        name: text("name")
            .or(note.key.as_deref())
            .unwrap_or(query)
            .to_owned(),
        query: query.to_owned(),
        for_ms,
        interrupt: v.get("interrupt").and_then(Value::as_bool).unwrap_or(false),
        quiet_hours,
        rate_limit,
        operator: !matches!(note.author, Author::Agent(_)),
    })
}

/// How a fired row was handed off.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum How {
    Notice,
    Interrupt,
    /// Past the rate limit: kept for the next briefing.
    Deferred,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Handoff {
    pub rule: String,
    pub row: String,
    /// When the row was first seen matching.
    pub matched: i64,
    pub at: i64,
    pub how: How,
    /// Why the handoff is not what the rule asked for.
    pub why: Option<String>,
}

/// A delivery already recorded, as the evaluator needs it.
#[derive(Debug, Clone, PartialEq)]
pub struct Sent {
    pub rule: String,
    pub row: String,
    pub at: i64,
    pub interrupt_by_agent: bool,
}

/// The shared interrupt budget of agent-written rules
/// (`signal.interrupts_per_hour`).
pub const INTERRUPTS_PER_HOUR: usize = 4;

#[derive(Debug, Default)]
pub struct Evaluator {
    /// (rule, row) → (due, first seen matching).
    pending: HashMap<(String, String), (i64, i64)>,
    /// Pairs already handed off, or matching before the rule was first
    /// evaluated.
    consumed: HashSet<(String, String)>,
    sent: HashMap<String, Vec<i64>>,
    interrupts: Vec<i64>,
    /// Rules evaluated at least once: their first matches are a baseline.
    known: HashSet<String>,
    pub budget: Option<usize>,
}

impl Evaluator {
    /// Starts from the deliveries already recorded, so a restart neither
    /// repeats nor forgets them.
    pub fn new(history: &[Sent]) -> Self {
        let mut e = Self::default();
        for s in history {
            e.consumed.insert((s.rule.clone(), s.row.clone()));
            e.sent.entry(s.rule.clone()).or_default().push(s.at);
            e.known.insert(s.rule.clone());
            if s.interrupt_by_agent {
                e.interrupts.push(s.at);
            }
        }
        e
    }

    /// One evaluation at `now`: `matches` holds each rule's matching rows.
    /// `local_hour` gives the local hour of an instant, for quiet hours.
    pub fn step(
        &mut self,
        now: i64,
        rules: &[Rule],
        matches: &HashMap<String, Vec<String>>,
        local_hour: impl Fn(i64) -> u8,
    ) -> Vec<Handoff> {
        let budget = self.budget.unwrap_or(INTERRUPTS_PER_HOUR);
        let live: HashSet<&str> = rules.iter().map(|r| r.id.as_str()).collect();
        // A withdrawn rule's pending work is dropped.
        self.pending
            .retain(|(rule, _), _| live.contains(rule.as_str()));
        let mut out = Vec::new();
        for rule in rules {
            let rows: HashSet<&str> = matches
                .get(&rule.id)
                .map(|v| v.iter().map(String::as_str).collect())
                .unwrap_or_default();
            let first = self.known.insert(rule.id.clone());
            // A row that stopped matching: pending work is dropped, and a
            // later match is a new episode that may fire again.
            self.pending
                .retain(|(r, row), _| r != &rule.id || rows.contains(row.as_str()));
            self.consumed
                .retain(|(r, row)| r != &rule.id || rows.contains(row.as_str()));
            for row in &rows {
                let key = (rule.id.clone(), (*row).to_owned());
                if first && rule.for_ms == 0 {
                    // Rows matching when a rule is first read are its
                    // baseline, not news.
                    self.consumed.insert(key);
                    continue;
                }
                if self.consumed.contains(&key) || self.pending.contains_key(&key) {
                    continue;
                }
                self.pending.insert(key, (now + rule.for_ms, now));
            }
            let mut due: Vec<(String, String)> = self
                .pending
                .iter()
                .filter(|((r, _), (d, _))| r == &rule.id && *d <= now)
                .map(|(k, _)| k.clone())
                .collect();
            due.sort();
            for key in due {
                let (_, matched) = self.pending[&key];
                if rule.interrupt
                    && let Some(q) = rule.quiet_hours
                    && in_quiet(local_hour(now), q)
                {
                    // Re-checked at the next evaluation after quiet hours.
                    continue;
                }
                self.pending.remove(&key);
                self.consumed.insert(key.clone());
                let sent = self.sent.entry(rule.id.clone()).or_default();
                if let Some((n, window)) = rule.rate_limit
                    && sent.iter().filter(|s| now - **s < window).count() >= n
                {
                    out.push(Handoff {
                        rule: rule.id.clone(),
                        row: key.1,
                        matched,
                        at: now,
                        how: How::Deferred,
                        why: Some("rate limit: kept for the next briefing".into()),
                    });
                    continue;
                }
                let (mut how, mut why) = if rule.interrupt {
                    (How::Interrupt, None)
                } else {
                    (How::Notice, None)
                };
                if how == How::Interrupt && !rule.operator {
                    let recent = self.interrupts.iter().filter(|s| now - **s < HOUR).count();
                    if recent >= budget {
                        how = How::Notice;
                        why = Some("the agents' shared interrupt budget is spent".into());
                    } else {
                        self.interrupts.push(now);
                    }
                }
                sent.push(now);
                out.push(Handoff {
                    rule: rule.id.clone(),
                    row: key.1,
                    matched,
                    at: now,
                    how,
                    why,
                });
            }
        }
        out
    }

    /// The earliest instant pending work becomes due, for the timer.
    pub fn next_due(&self) -> Option<i64> {
        self.pending.values().map(|(d, _)| *d).min()
    }
}

fn in_quiet(hour: u8, (from, to): (u8, u8)) -> bool {
    if from > to {
        hour >= from || hour < to
    } else {
        from <= hour && hour < to
    }
}

/// A stable key for a query row: its `id`, else its `key`, else the whole
/// row.
pub fn row_key(row: &Value) -> String {
    ["id", "key"]
        .iter()
        .find_map(|k| row.get(*k).and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| row.to_string())
}

/// The references a signal delivery records: the rule, then the row.
pub fn refs(rule: &str, row: &str) -> Vec<String> {
    vec![format!("signal:{rule}"), format!("row:{row}")]
}

/// Reads a recorded delivery's rule and row back from its references.
pub fn sent_from(refs: &[String], at: i64, interrupt_by_agent: bool) -> Option<Sent> {
    let rule = refs.iter().find_map(|r| r.strip_prefix("signal:"))?;
    let row = refs.iter().find_map(|r| r.strip_prefix("row:"))?;
    Some(Sent {
        rule: rule.to_owned(),
        row: row.to_owned(),
        at,
        interrupt_by_agent,
    })
}

/// The notice text for one handoff.
pub fn notice(rule: &Rule, h: &Handoff) -> String {
    let mut text = format!("{}: {}", rule.name, h.row);
    if let Some(why) = &h.why {
        text.push_str(&format!(" ({why})"));
    }
    text
}

/// What `status` shows of the rules: each with its author, or why it is
/// not used.
pub fn summary(rules: &[Rule], unusable: &[Unusable]) -> Value {
    json!({
        "rules": rules.iter().map(|r| json!({
            "id": r.id, "name": r.name, "query": r.query,
            "author": if r.operator { "operator" } else { "agent" },
        })).collect::<Vec<_>>(),
        "unusable": unusable.iter().map(|u| json!({"id": u.id, "reason": u.reason})).collect::<Vec<_>>(),
    })
}

/// Signal deliveries already recorded, for the evaluator to start from.
fn history(d: &Daemon) -> Vec<Sent> {
    let Ok(r) = d.reads.reader() else {
        return vec![];
    };
    let rows = r
        .query(
            "SELECT at, refs_json FROM a.delivery WHERE sender = 'signal' ORDER BY at",
            &[],
        )
        .unwrap_or_default();
    rows.iter()
        .filter_map(|row| {
            let refs: Vec<String> = serde_json::from_str(row["refs_json"].as_str()?).ok()?;
            let agent = refs.iter().any(|r| r == "interrupt:agent");
            sent_from(&refs, row["at"].as_i64()?, agent)
        })
        .collect()
}

/// Evaluates the signal rules after each change and at each `for`
/// deadline, so a duration fires with no new data (law 140), and records
/// every handoff as a delivery.
pub fn run(d: &Daemon) {
    use polylogue_model::LogicalDay;
    let day = LogicalDay::system();
    let mut eval = Evaluator::new(&history(d));
    loop {
        let now = now_ms();
        let Ok(reader) = d.reads.reader() else {
            std::thread::sleep(std::time::Duration::from_secs(30));
            continue;
        };
        let filter = polylogue_store::NoteFilter {
            kinds: vec![NoteKind::Signal],
            ..Default::default()
        };
        let mut rules: Vec<Rule> = vec![];
        let mut unusable: Vec<Unusable> = vec![];
        for note in reader.notes(&filter).unwrap_or_default() {
            match rule(&note) {
                Ok(r) => rules.push(r),
                Err(u) => unusable.push(u),
            }
        }
        let mut matches = std::collections::HashMap::new();
        let mut usable = vec![];
        for rule in rules {
            let ask = crate::daemon::query::Ask {
                text: &rule.query,
                cursor: None,
                page_size: Some(1_000),
                why: false,
                session: None,
                now,
                day: day.clone(),
                embeddings: false,
                near: None,
                fields: &[],
            };
            match crate::daemon::query::run(&reader, ask) {
                Ok(v) => {
                    let rows = v["rows"]
                        .as_array()
                        .map(|rows| rows.iter().map(row_key).collect())
                        .unwrap_or_default();
                    matches.insert(rule.id.clone(), rows);
                    usable.push(rule);
                }
                Err(f) => unusable.push(Unusable {
                    id: rule.id.clone(),
                    reason: format!("its query fails: {}", f.message),
                }),
            }
        }
        if let Ok(mut s) = d.signals.lock() {
            *s = summary(&usable, &unusable);
        }
        for h in eval.step(now, &usable, &matches, |t| day.local_hour(t)) {
            let Some(rule) = usable.iter().find(|r| r.id == h.rule) else {
                continue;
            };
            let mut refs = refs(&h.rule, &h.row);
            if h.how == How::Interrupt && !rule.operator {
                refs.push("interrupt:agent".into());
            }
            let id = polylogue_store::user_writes::new_request_id();
            let delivery = Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "signal".into(),
                event: "signal".into(),
                cwd: None,
                text: notice(rule, &h),
                refs,
                left_out: vec![],
            };
            if d.writes
                .user(UserRequest::now(id.clone(), UserWrite::Deliver(delivery)))
                .is_err()
            {
                continue;
            }
            // A notice is shown by `status`, briefings and the change feed;
            // an interrupt needs the notifier `signals.toml` names, which is
            // not built, so it stays a notice and says so.
            let event = match h.how {
                How::Notice => None,
                How::Interrupt => Some((
                    polylogue_store::DeliveryOutcome::NotSent,
                    "no notifier is configured",
                )),
                How::Deferred => Some((
                    polylogue_store::DeliveryOutcome::NotSent,
                    "rate limit: kept for the next briefing",
                )),
            };
            if let Some((what, detail)) = event {
                let _ = d.writes.user(UserRequest::now(
                    polylogue_store::user_writes::new_request_id(),
                    UserWrite::DeliveryEvent {
                        delivery: id,
                        what,
                        detail: Some(detail.into()),
                    },
                ));
            }
        }
        let wait = eval
            .next_due()
            .map_or(30_000, |due| (due - now_ms()).clamp(1_000, 30_000));
        std::thread::sleep(std::time::Duration::from_millis(wait as u64));
    }
}

#[cfg(test)]
mod tests;
