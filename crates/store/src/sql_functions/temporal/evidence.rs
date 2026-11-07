//! Source evidence and the possible latest time of a set of source periods.
use super::{Failure, period};
use crate::query::fields::NativeTimeStorage;
use polylogue_model::{Event, Precision, TimeEvidence, TimeQuality, Timestamp};
use query_language::time::Time;
use rusqlite::{
    Error,
    functions::{Aggregate, Context},
};
use serde::{Deserialize, Serialize};

/// A derived time may have several possible source-quality classes. It does
/// not turn those sources into a fictitious single native provenance class.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Evidence {
    pub at: Option<Timestamp>,
    pub quality: Option<TimeQuality>,
    pub precision: Option<Precision>,
    pub until: Option<Timestamp>,
}
impl Evidence {
    pub fn period(&self) -> Option<Time> {
        period(self.at, self.precision, self.until)
    }
    fn native(at: Option<Timestamp>, source: TimeEvidence) -> Result<Self, Failure> {
        source.validate(at)?;
        Ok(Self {
            at,
            quality: Some(source.quality),
            precision: source.precision,
            until: source.until,
        })
    }
    fn validate(&self) -> Result<(), Failure> {
        if self.at.is_none() && (self.until.is_some() || self.precision.is_some()) {
            return Err("derived time bounds require a known time".into());
        }
        if self
            .until
            .is_some_and(|until| self.at.is_none_or(|at| until <= at))
        {
            return Err("derived time end must follow its start".into());
        }
        match self.precision {
            Some(Precision::Instant) if self.until.is_some() => {
                Err("a derived instant has no upper bound".into())
            }
            Some(Precision::Bounded) if self.until.is_none() => {
                Err("a derived period needs its upper bound".into())
            }
            _ => Ok(()),
        }
    }
}

pub(super) fn read(bytes: Option<&[u8]>, storage: &str) -> Result<Evidence, Failure> {
    let storage = NativeTimeStorage::from_code(storage).ok_or("unknown native time storage")?;
    let Some(bytes) = bytes else {
        return Ok(Evidence::default());
    };
    match storage {
        NativeTimeStorage::Tuple => {
            let (at, source) = serde_json::from_slice(bytes)?;
            Evidence::native(at, source)
        }
        NativeTimeStorage::Event => {
            let event: Event = serde_json::from_slice(&crate::encoding::unpack_bytes(bytes)?)?;
            Evidence::native(event.at, event.time_evidence())
        }
        NativeTimeStorage::Latest => {
            let evidence: Evidence = serde_json::from_slice(bytes)?;
            evidence.validate()?;
            Ok(evidence)
        }
    }
}

#[derive(Clone, Copy)]
enum Upper {
    Bounded { at: Timestamp, inclusive: bool },
    Unbounded,
}
impl Upper {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unbounded, _) | (_, Self::Unbounded) => Self::Unbounded,
            (
                Self::Bounded {
                    at: a,
                    inclusive: ai,
                },
                Self::Bounded {
                    at: b,
                    inclusive: bi,
                },
            ) => {
                if a > b {
                    self
                } else if a < b {
                    other
                } else {
                    Self::Bounded {
                        at: a,
                        inclusive: ai || bi,
                    }
                }
            }
        }
    }
    fn can_be_latest(self, lower: Timestamp) -> bool {
        match self {
            Self::Unbounded => true,
            Self::Bounded { at, inclusive } => at > lower || (at == lower && inclusive),
        }
    }
}

#[derive(Default)]
pub(super) struct LatestState {
    lower: Option<Timestamp>,
    upper: Option<Upper>,
    // There is at most one bound per member of the closed TimeQuality set.
    qualities: Vec<(TimeQuality, Upper)>,
}
impl LatestState {
    fn add(&mut self, evidence: Evidence) {
        let Some(at) = evidence.at else { return };
        let upper = match evidence.period() {
            Some(Time::Instant { at }) => Upper::Bounded {
                at: at.into(),
                inclusive: true,
            },
            Some(Time::Span { end, .. }) => Upper::Bounded {
                at: end.into(),
                inclusive: false,
            },
            None => Upper::Unbounded,
        };
        self.lower = Some(self.lower.map_or(at, |old| old.max(at)));
        self.upper = Some(self.upper.map_or(upper, |old| old.merge(upper)));
        if let Some(quality) = evidence.quality {
            if let Some((_, bound)) = self.qualities.iter_mut().find(|(q, _)| *q == quality) {
                *bound = bound.merge(upper);
            } else {
                self.qualities.push((quality, upper));
            }
        }
    }
    fn finish(self) -> Option<Evidence> {
        let at = self.lower?;
        let mut candidates = self
            .qualities
            .into_iter()
            .filter(|(_, bound)| bound.can_be_latest(at));
        let quality = candidates
            .next()
            .and_then(|(quality, _)| candidates.next().is_none().then_some(quality));
        let (precision, until) = match self.upper? {
            Upper::Unbounded => (None, None),
            Upper::Bounded { at: end, .. } if end > at => (Some(Precision::Bounded), Some(end)),
            Upper::Bounded { .. } => (Some(Precision::Instant), None),
        };
        Some(Evidence {
            at: Some(at),
            quality,
            precision,
            until,
        })
    }
}

pub(super) struct Latest;
impl Aggregate<LatestState, Option<Vec<u8>>> for Latest {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<LatestState> {
        Ok(LatestState::default())
    }
    fn step(&self, ctx: &mut Context<'_>, state: &mut LatestState) -> rusqlite::Result<()> {
        let bytes: Option<Vec<u8>> = ctx.get(0)?;
        state.add(read(bytes.as_deref(), "tuple").map_err(Error::UserFunctionError)?);
        Ok(())
    }
    fn finalize(
        &self,
        _: &mut Context<'_>,
        state: Option<LatestState>,
    ) -> rusqlite::Result<Option<Vec<u8>>> {
        state
            .and_then(LatestState::finish)
            .map(|value| {
                serde_json::to_vec(&value).map_err(|e| Error::UserFunctionError(Box::new(e)))
            })
            .transpose()
    }
}
