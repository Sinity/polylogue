//! Actual source classification is separate from portable fragment claims.
use super::{Expansion, ProjectionEvidence, ReadError, ReadOptions, ValueState, View};
use crate::{BlockKind, PromptKind, Role, bundle::TextFragmentRecord};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FragmentOrigin {
    /// None means an actual non-message container, such as a Document.
    pub message: Option<FragmentMessageOrigin>,
    /// Exhaustive original body ranges over the selected interval, including
    /// accessible-empty and adjacent bodies needed to prove separator gaps.
    /// Missing or unavailable bodies cannot be skipped to mint known origin.
    pub bodies: Vec<FragmentBodyOrigin>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FragmentMessageOrigin {
    pub role: Role,
    pub prompt_kind: Option<PromptKind>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FragmentBodyOrigin {
    pub start: u64,
    pub end: u64,
    pub kind: BlockKind,
}
impl FragmentOrigin {
    pub fn validate(&self, start: u64, end: u64) -> Result<(), ReadError> {
        let invalid = || {
            ReadError::InvalidEvidence(
                "fragment origin does not prove the selected source recipe".into(),
            )
        };
        if start > end {
            return Err(invalid());
        }
        let Some(first) = self.bodies.first() else {
            return if start == 0 && end == 0 {
                Ok(())
            } else {
                Err(invalid())
            };
        };
        let last = self.bodies.last().expect("nonempty bodies");
        if first.start > start || last.end < end {
            return Err(invalid());
        }
        let mut prior_end: Option<u64> = None;
        for body in &self.bodies {
            if body.start > body.end
                || !matches!(
                    body.kind,
                    BlockKind::Text | BlockKind::Thinking | BlockKind::ToolResult
                )
                || prior_end.is_some_and(|prior| {
                    prior.checked_add(crate::TEXT_BODY_SEPARATOR.len() as u64) != Some(body.start)
                })
            {
                return Err(invalid());
            }
            prior_end = Some(body.end);
        }
        Ok(())
    }
}

pub(crate) fn project_fragment(
    row: &TextFragmentRecord,
    options: &ReadOptions,
    evidence: &ProjectionEvidence,
) -> Result<(), ReadError> {
    row.validate()
        .map_err(|error| ReadError::InvalidEvidence(error.to_string()))?;
    let unsupported = || ReadError::UnsupportedTarget {
        view: options.view,
        target: row.reference.clone(),
        accepted_views: vec![View::Transcript, View::Raw],
    };
    if matches!(
        options.view,
        View::Summary | View::Links | View::Changes | View::Live
    ) {
        return Err(unsupported());
    }
    let ValueState::Value { value: origin } = &evidence.fragment_origin else {
        return if matches!(options.view, View::Transcript | View::Raw) {
            Ok(())
        } else {
            Err(ReadError::MissingEvidence {
                view: options.view,
                evidence: "exact source role and body kinds".into(),
                reference: row.reference.clone(),
            })
        };
    };
    let coverage = row.fragment.coverage;
    origin.validate(coverage.start, coverage.end)?;
    let message_container = matches!(
        &row.fragment.container,
        crate::Reference::Message(_) | crate::Reference::Block { .. }
    );
    if message_container != origin.message.is_some() {
        return Err(ReadError::InvalidEvidence(
            "fragment origin disagrees with its container family".into(),
        ));
    }
    if options.view == View::Raw {
        return Ok(());
    }
    if matches!(options.view, View::Dialogue | View::Context) {
        let message = origin.message.as_ref().ok_or_else(unsupported)?;
        if !options
            .view
            .keeps_message_metadata(message.role, message.prompt_kind)
        {
            return Err(unsupported());
        }
    }
    let harness = origin.message.as_ref().is_some_and(|message| {
        message.role == Role::System || message.prompt_kind.is_some_and(PromptKind::is_harness)
    });
    let missing = |evidence: &str| ReadError::MissingEvidence {
        view: options.view,
        evidence: evidence.into(),
        reference: row.reference.clone(),
    };
    if harness && options.harness() != Expansion::Full {
        return Err(ReadError::RequiredExpansion {
            view: options.view,
            target: row.reference.clone(),
            option: "harness".into(),
        });
    }
    for body in &origin.bodies {
        if body.start < coverage.end
            && body.end > coverage.start
            && !options.text_kind_visible(body.kind)
        {
            if !options.view.block_kinds().contains(&body.kind) {
                return Err(unsupported());
            }
            return Err(ReadError::RequiredExpansion {
                view: options.view,
                target: row.reference.clone(),
                option: match body.kind {
                    BlockKind::Thinking => "thinking",
                    BlockKind::ToolResult => "tools",
                    _ => unreachable!("available source text kind"),
                }
                .into(),
            });
        }
    }
    if options.view == View::Context {
        let context = evidence
            .context
            .as_ref()
            .ok_or_else(|| missing("context membership"))?;
        if !context.selected {
            return Err(unsupported());
        }
    }
    Ok(())
}
