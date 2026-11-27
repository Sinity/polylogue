//! Session readings over transcript lines. A reading is composed once
//! ([`Reading::open`]): its parts, its binding to the accepted start, the
//! entries the view shows, its header and the lines that open it. Two
//! drivers lay out its entries through the same steps: a page
//! ([`cut::page`]) reads the entries its budget holds and puts the
//! continuation in its footer; a stream reads them all in windows.
//!
//! Where a page sits among the shown entries, and the counts a reading
//! shows, come from the counts the writer stores per entry class, so a page
//! reads its own entries and the bodies of its messages and events, not the
//! entries before it. A continuation names its first entry by key and binds
//! the rewrite version of every composed session, so appends never stale it
//! and rewrites always do. Nothing is written.
use super::{ReadRequest, layout_facts, output};
use crate::{Error, ReadServices};
use polylogue_model::Reason;
use polylogue_model::{MessageId, Reference, SessionId, bundle::Record, read::*};
use polylogue_store::Reader;
use polylogue_store::read::limit::ReadLimit;
use std::collections::HashSet;
use std::sync::Arc;

mod bind;
mod compose;
mod cut;
mod entries;
mod lay_out;
mod range;
mod window;
use compose::Composed;
use entries::ShownEntries;

/// What a session read covers.
pub(super) struct Scope {
    pub session: SessionId,
    /// For a message or message range: the first and last message read.
    pub range: Option<(MessageId, MessageId)>,
    /// `root:SESSION`: the logical session the root names.
    pub logical: bool,
}

impl Scope {
    /// A context is anchored by one selected message, or a session's latest.
    pub(super) fn has_context_anchor(&self) -> bool {
        self.range
            .as_ref()
            .is_none_or(|(first, last)| first == last)
    }
}

/// The session read a reference names, when it names one. `root:REF` names
/// the logical session whose root is REF.
pub(super) fn scope(reader: &Reader<'_>, target: &str) -> Result<Option<Scope>, Error> {
    if let Some(root) = target.strip_prefix("root:") {
        let reference =
            Reference::parse(&format!("session:{root}")).map_err(|error| Reason::BadReference {
                text: target.to_owned(),
                problem: error.to_string(),
            })?;
        let Reference::Session(session) = reference else {
            return Err(Reason::RootNamesSession {}.into());
        };
        return Ok(Some(Scope {
            session,
            range: None,
            logical: true,
        }));
    }
    let Ok(reference) = Reference::parse(target) else {
        return Ok(None);
    };
    let session_of = |message: &MessageId| -> Result<SessionId, Error> {
        reader
            .message_session(message)?
            .ok_or_else(|| Error::not_found(format!("no message {message}")))
    };
    Ok(match reference {
        Reference::Session(session) => Some(Scope {
            session,
            range: None,
            logical: false,
        }),
        Reference::Message(message) => Some(Scope {
            session: session_of(&message)?,
            range: Some((message.clone(), message)),
            logical: false,
        }),
        Reference::MessageRange { first, last } => {
            let session = session_of(&first)?;
            if session_of(&last)? != session {
                return Err(Reason::RangeAcrossSessions {}.into());
            }
            Some(Scope {
                session,
                range: Some((first, last)),
                logical: false,
            })
        }
        _ => None,
    })
}

/// Stops a read whose client has gone or whose deadline has passed. SQLite
/// steps are interrupted by the store; this covers the work between them.
fn still_wanted() -> Result<(), Error> {
    if let Some(limit) = ReadLimit::current() {
        limit.check()?;
    }
    Ok(())
}

pub(super) fn header(
    reader: &Reader<'_>,
    session: &SessionId,
    target: &str,
    options: &ReadOptions,
) -> Result<ReadHeader, Error> {
    let stored = reader
        .session(session)?
        .ok_or_else(|| Error::not_found(format!("no session {session}")))?;
    let title = reader
        .display_title(session)?
        .map(ValueState::known)
        .unwrap_or_default();
    Ok(ReadHeader {
        target: target.to_owned(),
        view: options.view,
        title,
        kind: HeaderKind::Session {
            session: Box::new(stored),
        },
        figures: Vec::new(),
        zone: None,
        snapshot: None,
        options: None,
    })
}

/// A session reading composed from one pinned reader, before any of its
/// entries is laid out.
pub(super) struct Reading {
    /// The request as accepted: its seen marks bound to every physical owner.
    request: ReadRequest,
    /// The request a continuation repeats: a historical context bound to
    /// its message, else the accepted request.
    continuation: ReadRequest,
    scope: Scope,
    composed: Composed,
    runs: compose::Runs,
    header: ReadHeader,
    /// The lines that open the reading, before its first entry.
    lead: Vec<ReadItem>,
    omission: Option<PositionedOmission>,
}

struct PositionedOmission {
    place: (usize, usize),
    after: bool,
    item: ReadItem,
}

impl Reading {
    /// Compose, bind to the accepted start, choose the shown runs, and
    /// prepare the header and lead lines. `opened` is the verified cursor
    /// of a continuation page.
    fn open(
        reader: &Reader<'_>,
        request: &ReadRequest,
        operation: &crate::operation::Context,
        scope: Scope,
        opened: Option<&PageCursor>,
    ) -> Result<Self, Error> {
        let options = &request.options;
        // Only an authenticated continuation may carry the original root's
        // physical option onto its already-bound message selector.
        let bound_physical =
            opened.is_some() && options.view == View::Context && options.annotation_at().is_some();
        if options.physical && !request.target.starts_with("root:") && !bound_physical {
            return Err(Reason::PhysicalNeedsRoot {}.into());
        }
        let scope =
            match bind::session_read(request).filter(|bound| opened.is_some() && bound.logical) {
                Some(bound) => Scope {
                    session: bound.session.clone(),
                    range: None,
                    logical: true,
                },
                None => Scope {
                    logical: scope.logical && !options.physical,
                    ..scope
                },
            };
        let mut composed = compose::compose(reader, &scope, options)?;
        let mut accepted = bind::bind_seen(reader, request, operation, &composed)?;
        range::trim_end(reader, &mut accepted, operation, &scope, &mut composed)?;
        let bound = if options.view == View::Context && options.annotation_at().is_some() {
            bind::bind_context(reader, &accepted, &mut composed)?
        } else {
            None
        };
        bind::trim_start(reader, &accepted, &mut composed)?;
        still_wanted()?;
        let mut runs = compose::runs(reader, &composed, options)?;
        let omission = window::apply(reader, &composed, &mut runs, &mut accepted, &scope.session)?;
        still_wanted()?;
        let read_session = reader
            .session(&scope.session)?
            .ok_or_else(|| Error::not_found(format!("no session {}", scope.session)))?;
        let inherited_messages = if !options.inherited() && options.view != View::Context {
            let mut total = 0;
            for (at, part) in composed.parts.iter().enumerate() {
                if part.inherited {
                    total += composed.count(
                        reader,
                        at,
                        (part.start, part.end),
                        EntryClass::is_header,
                    )?;
                }
            }
            total
        } else {
            0
        };
        let mut lead = lay_out::lead_items(
            &composed,
            &runs,
            options,
            &request.target,
            &read_session,
            inherited_messages,
            if options.annotations() {
                layout_facts::annotations(
                    reader,
                    &Reference::Session(read_session.id.clone()),
                    operation.ran_at,
                    options.annotation_at(),
                )?
            } else {
                vec![]
            },
        );
        let mut header = header(reader, &scope.session, &request.target, options)?;
        if scope.logical {
            header.kind = compose::logical_header(reader, &scope.session, &composed)?;
        }
        let omission = if let Some(omission) = omission {
            let item = ReadItem {
                source: Source::record(&request.target),
                kind: ItemKind::Omission {
                    count: ValueState::known(omission.count),
                    unit: "message".into(),
                    why: "outside the requested message window".into(),
                    continue_call: None,
                },
            };
            if options.first.unwrap_or(0) == 0 {
                lead.push(item);
                None
            } else if let Some(place) = omission.next {
                Some(PositionedOmission {
                    place,
                    after: false,
                    item,
                })
            } else {
                let shown = ShownEntries::new(reader, &composed, &runs, ShowProfile::of(options))?;
                shown
                    .last_before((usize::MAX, usize::MAX))?
                    .map(|entry| PositionedOmission {
                        place: entry.place(),
                        after: true,
                        item,
                    })
            }
        } else {
            None
        };
        Ok(Self {
            continuation: bound.unwrap_or_else(|| accepted.clone()),
            request: accepted,
            scope,
            composed,
            runs,
            header,
            lead,
            omission,
        })
    }

    /// The entries the reading shows, in order.
    fn shown<'r>(&'r self, reader: &'r Reader<'r>) -> Result<ShownEntries<'r, 'r>, Error> {
        ShownEntries::new(
            reader,
            &self.composed,
            &self.runs,
            ShowProfile::of(&self.request.options),
        )
    }

    /// The items of `entries`, laid out and decorated in reading order, one
    /// list per entry; with `originals`, the source record each entry's
    /// message or event comes from, beside its first entry.
    #[allow(clippy::too_many_arguments)]
    fn lay_out(
        &self,
        reader: &Reader<'_>,
        services: &impl ReadServices,
        shown: &ShownEntries<'_, '_>,
        entries: &[entries::PageEntry],
        at: polylogue_model::Timestamp,
        cache: &mut Option<lay_out::PreparedRecord>,
        originals: bool,
    ) -> Result<lay_out::LaidOut, Error> {
        let options = &self.request.options;
        let mut laid = lay_out::lay_out(
            reader,
            services,
            &self.composed,
            &self.runs,
            entries,
            options,
            at,
            cache,
            self.request.start.uses_seen(),
            originals,
        )?;
        still_wanted()?;
        lay_out::decorate(
            services,
            reader,
            &self.composed,
            shown,
            entries,
            &mut laid.items,
            options,
        )?;
        if let Some(omission) = &self.omission
            && let Some(slot) = entries
                .iter()
                .position(|entry| entry.place() == omission.place)
        {
            let items = &mut laid.items[slot];
            if omission.after {
                items.push(omission.item.clone());
            } else {
                items.insert(0, omission.item.clone());
            }
        }
        still_wanted()?;
        Ok(laid)
    }
}

/// One page of a session reading and the seen marks its display leaves.
pub(super) fn page(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    scope: Scope,
) -> Result<(ReadPage, Vec<polylogue_model::user::SeenMark>), Error> {
    let (_, _, digest) = super::page_terms(request)?;
    let key = super::cursor_seal(reader)?;
    let opened = request
        .cursor
        .as_deref()
        .map(|text| {
            let opened = PageCursor::open(text, &key).map_err(super::cursor_failure)?;
            opened
                .check(&request.target, &digest)
                .map_err(super::cursor_failure)?;
            Ok::<_, Error>(opened)
        })
        .transpose()?;
    let reading = Reading::open(reader, request, operation, scope, opened.as_ref())?;
    cut::page(reader, services, request, operation, &reading, opened, key)
}

/// The whole session reading as a stream, laid out window by window. Only a
/// small entry window and its current source body are held; each source
/// record goes out once, before the first item it holds.
pub(super) fn stream(
    reader: &Reader<'_>,
    services: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    scope: Scope,
    snapshot: polylogue_model::bundle::ReadSnapshot,
    output: &mut dyn std::io::Write,
) -> Result<ReadFooter, Error> {
    let reading = Reading::open(reader, request, operation, scope, None)?;
    let shown = reading.shown(reader)?;
    let token = reader.scope_token();
    let check = || {
        token.check()?;
        reader.check()?;
        still_wanted()
    };
    let mut header = reading.header.clone();
    header.snapshot = Some(Box::new(snapshot));
    header.options = Some(Box::new(request.options.clone()));
    let mut stream = output::Stream::begin(output, &check, &header)?;
    let mut emitted = HashSet::new();
    // A composed header does not replace its physical sources' metadata. The
    // main session may occur again in the parts; send its original only once.
    let session = &reading.scope.session;
    for id in
        std::iter::once(session).chain(reading.composed.parts.iter().map(|part| &part.session))
    {
        if emitted.insert(format!("session:{id}")) {
            stream.record(&ReadRecord::Original {
                record: Box::new(super::records::session_record(reader, id)?),
                parts: vec![],
            })?;
        }
    }
    for item in &reading.lead {
        stream.record(&ReadRecord::Presentation {
            parts: vec![ReadPart::Item(item.clone())],
        })?;
    }
    let mut blobs = (request.delivery == ReadDelivery::Bundle)
        .then(|| super::blobs::Blobs::new(reader))
        .transpose()?;
    let mut cache = None;
    let mut from = shown.start();
    loop {
        check()?;
        let window = shown.page_after(from, u64::MAX, entries::FIRST_WINDOW)?;
        if window.entries.is_empty() {
            break;
        }
        let laid = reading.lay_out(
            reader,
            services,
            &shown,
            &window.entries,
            operation.ran_at,
            &mut cache,
            true,
        )?;
        let mut records = laid.records.into_iter().peekable();
        for (slot, items) in laid.items.into_iter().enumerate() {
            if records.peek().is_some_and(|(at, _)| *at == slot) {
                let (_, record) = records.next().expect("the slot has its prepared record");
                let reference = record
                    .record()
                    .and_then(Record::reference)
                    .ok_or_else(|| Error::internal("a composed source has no reference"))?;
                if emitted.insert(reference.to_owned()) {
                    // Copied membership can name a canonical source session
                    // outside the display parts. Preserve that source header too.
                    let source_session = match record.record() {
                        Some(Record::Message(row)) => Some(&row.message.session),
                        Some(Record::Event(row)) => row.event.session.as_ref(),
                        _ => None,
                    };
                    if let Some(id) = source_session
                        && emitted.insert(format!("session:{id}"))
                    {
                        stream.record(&ReadRecord::Original {
                            record: Box::new(super::records::session_record(reader, id)?),
                            parts: vec![],
                        })?;
                    }
                    stream.record(&record)?;
                    if let Some(blobs) = &mut blobs {
                        blobs.record(&record, &mut stream, &check)?;
                    }
                }
            }
            for item in items {
                stream.record(&ReadRecord::Presentation {
                    parts: vec![ReadPart::Item(item)],
                })?;
            }
        }
        match window.next {
            Some(next) => from = next.place(),
            None => break,
        }
    }
    check()?;
    stream.finish(ReadFooter::default())
}

#[cfg(test)]
mod tests;
