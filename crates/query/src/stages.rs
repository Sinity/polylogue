//! The stages after a selection, in the order they apply (docs/query.md:
//! Stages, Combining queries). A stage points at the bound transforms it
//! lowers rather than copying them; the store reads those transforms and
//! lowers each stage to one statement over the rows entering it.
use crate::{QueryError, ast::*};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    /// The indexes of the bound transforms this stage lowers.
    pub transforms: Range<usize>,
    pub kind: StageKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// One transform over the rows entering it.
    Rows,
    /// `| group by K` with the `first N` or `agg …` straight after it, which
    /// reads the rows entering the grouping, group by group.
    PerGroup,
    /// Consecutive `| union` and `| fuse` stages: one reciprocal-rank
    /// fusion over the rows entering them and every operand.
    Fusion,
}

/// The stages of `transforms` from index `from` on. A `| measure` ends
/// the query.
pub fn stages(transforms: &[Transform], from: usize) -> Result<Vec<Stage>, QueryError> {
    let mut stages = vec![];
    let mut index = from;
    while let Some(transform) = transforms.get(index) {
        if index > from && measures(&transforms[index - 1]) {
            return Err(QueryError::Invalid {
                message: "`| measure WHAT` is the last stage".into(),
                position: transform.span.start,
            });
        }
        let (end, kind) = match &transform.kind {
            TransformKind::Group { .. }
                if transforms.get(index + 1).is_some_and(|next| {
                    matches!(
                        next.kind,
                        TransformKind::First { .. } | TransformKind::Aggregate { .. }
                    )
                }) =>
            {
                (index + 2, StageKind::PerGroup)
            }
            kind if fused(kind) => (
                index
                    + transforms[index..]
                        .iter()
                        .take_while(|next| fused(&next.kind))
                        .count(),
                StageKind::Fusion,
            ),
            _ => (index + 1, StageKind::Rows),
        };
        stages.push(Stage {
            transforms: index..end,
            kind,
        });
        index = end;
    }
    Ok(stages)
}

fn measures(transform: &Transform) -> bool {
    matches!(&transform.kind, TransformKind::Extension { name, .. } if name == crate::catalog::MEASURE)
}

fn fused(kind: &TransformKind) -> bool {
    matches!(
        kind,
        TransformKind::Combine {
            how: Combination::Union | Combination::Fuse,
            ..
        }
    )
}

#[cfg(test)]
mod tests;
