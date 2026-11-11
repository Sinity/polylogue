use super::*;
use crate::query::fields::{FieldId, rule};

#[test]
fn units_in_a_session_relate_to_it() {
    for unit in Unit::all().filter(|unit| fields::has_session(*unit)) {
        let r = spec(unit).relation("session").expect("session relation");
        assert_eq!(r.target, Unit::Sessions, "{unit:?}");
    }
}

#[test]
fn sessionless_units_keep_explicit_association_or_independence() {
    for unit in Unit::all().filter(|unit| {
        !fields::has_session(*unit) && !matches!(unit, Unit::Sessions | Unit::Documents)
    }) {
        assert!(spec(unit).session_scope.is_some(), "{unit:?}");
    }
    let documents = spec(Unit::Documents);
    assert!(!fields::has_session(Unit::Documents));
    assert!(documents.session_scope.is_none());
    assert!(polylogue_query::defaults::declared(Unit::Documents).is_empty());
    assert!(documents.relations.is_empty());
}

#[test]
fn templates_name_only_known_aliases() {
    let a = Aliases::plain();
    for unit in Unit::all() {
        let s = spec(unit);
        for t in [s.key, s.time, s.session_pk]
            .into_iter()
            .chain(s.identity.iter().copied())
            .chain(s.since)
            .chain(s.until)
        {
            let x = a.expand(t).replace("{v}", "0");
            assert!(!x.contains('{'), "{unit:?}: {t}");
        }
        assert!(!s.from(&a).contains('{'), "{unit:?} from");
        for (name, e, _) in s.columns {
            assert!(!e.render(&a, 0).contains('{'), "{unit:?}.{name}");
        }
    }
    for &id in FieldId::ALL {
        if let Some(x) = rule(id).expr(&a, 0) {
            assert!(!x.contains('{'), "{id:?}");
        }
    }
}

#[test]
fn the_grammar_names_every_unit() {
    for u in Unit::all() {
        let name = u.name();
        let q = crate::query::parse(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            matches!(q.source, crate::query::ast::Source::Unit { ref unit, .. } if unit == name),
            "{name} parses as {:?}",
            q.source
        );
        // `exists UNIT(…)` names the row units by their one name.
        if !matches!(u, Unit::Sessions | Unit::Changes) {
            let text = format!("exists {name}(x)");
            crate::query::parse(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
        }
    }
}

#[test]
fn identities_never_use_the_row_key() {
    for unit in Unit::all() {
        let s = spec(unit);
        for term in s.identity {
            assert!(!term.contains(".pk"), "{unit:?}: {term}");
            assert_ne!(*term, s.key, "{unit:?}");
        }
    }
}
