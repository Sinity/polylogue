use super::*;
use crate::fields::FIELDS;

#[test]
fn every_unit_is_declared_once() {
    for u in Unit::ALL {
        assert_eq!(UNITS.iter().filter(|s| s.unit == u).count(), 1, "{u:?}");
    }
    assert_eq!(UNITS.len(), Unit::ALL.len());
}

#[test]
fn names_round_trip() {
    for s in UNITS {
        assert_eq!(Unit::from_name(s.name), Some(s.unit));
        assert_eq!(Unit::from_name(s.singular), Some(s.unit));
    }
}

#[test]
fn units_in_a_session_relate_to_it() {
    for s in UNITS.iter().filter(|s| s.in_session) {
        let r = s.relation("session").expect("session relation");
        assert_eq!(r.target, Unit::Sessions, "{:?}", s.unit);
    }
}

#[test]
fn units_outside_a_session_scope_to_sessions_or_are_sessions() {
    for s in UNITS
        .iter()
        .filter(|s| !s.in_session && s.unit != Unit::Sessions)
    {
        assert!(s.session_scope.is_some(), "{:?}", s.unit);
    }
}

#[test]
fn templates_name_only_known_aliases() {
    let a = Aliases::plain();
    for s in UNITS {
        for t in [s.key, s.time, s.session_pk]
            .into_iter()
            .chain(s.since)
            .chain(s.until)
        {
            let x = a.expand(t).replace("{v}", "0");
            assert!(!x.contains('{'), "{:?}: {t}", s.unit);
        }
        assert!(!s.from(&a).contains('{'), "{:?} from", s.unit);
        for (name, e, _) in s.columns {
            assert!(!e.render(&a, 0).contains('{'), "{:?}.{name}", s.unit);
        }
    }
    for f in FIELDS {
        if let Some(x) = f.expr(&a, 0) {
            assert!(!x.contains('{'), "{:?}.{}", f.unit, f.name);
        }
    }
}

#[test]
fn the_grammar_names_every_unit() {
    for s in UNITS {
        let q = crate::parse(s.name).unwrap_or_else(|e| panic!("{}: {e}", s.name));
        assert!(
            matches!(q.source, crate::ast::Source::Unit { unit, .. } if unit == s.unit),
            "{} parses as {:?}",
            s.name,
            q.source
        );
        // `exists UNIT(…)` names the row units by their singular.
        if !matches!(s.unit, Unit::Sessions | Unit::Changes) {
            let text = format!("exists {}(x)", s.singular);
            crate::parse(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
        }
    }
}
