use super::*;
use query_language::{
    ast::{Source, SourceColumns},
    time::{Clock, DaySetting},
};
use std::collections::BTreeMap;

#[test]
fn explanation_uses_the_same_closed_saved_binding_as_lowering() {
    let options = Options {
        clock: Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            DaySetting::new(
                jiff::tz::TimeZone::get("Etc/UTC").unwrap(),
                jiff::civil::Time::midnight(),
            )
            .unwrap(),
        ),
        origins: vec!["test".into()],
        page_size: None,
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        fields: vec![],
        kept: BTreeMap::new(),
        provider_catalog: None,
        saved: {
            let mut bindings = polylogue_query::Bindings::default();
            let plan = polylogue_query::plan::prepare(
                "sessions | group by origin | count",
                &bindings,
                |_| Ok(()),
            )
            .unwrap();
            bindings
                .add_saved("daily", &plan, "revision", "family")
                .unwrap();
            bindings.saved
        },
    };
    let explanation = query("query:daily | where count > 1", &options).unwrap();
    assert!(matches!(
        explanation.parsed.source,
        Source::Saved {
            output: SourceColumns::Known,
            ..
        }
    ));
    assert_eq!(explanation.units, [ast::Unit::Sessions]);
    assert_eq!(explanation.program.unit, ast::Unit::Sessions);
    assert_eq!(explanation.program.shape, crate::query::Shape::Groups);
}
