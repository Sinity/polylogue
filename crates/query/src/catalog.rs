//! The language vocabulary is projected from the product's execution registry.
use crate::{ast::Unit, fields, units};
use query_language::catalog::{Catalog, Field, FieldKind, ReferenceSyntax, UnknownRule};

fn field(value: &fields::Field, name: String) -> Field {
    use fields::Kind;
    Field {
        name,
        kind: match value.kind {
            Kind::Match | Kind::Special => FieldKind::Match,
            Kind::Prefix => FieldKind::Reference,
            Kind::Substring => FieldKind::Substring,
            Kind::Subtree => FieldKind::Subtree,
            Kind::Number => FieldKind::Number,
            Kind::Money => FieldKind::Money,
            Kind::Duration => FieldKind::Duration,
            Kind::Time => FieldKind::Time,
            Kind::Bool => FieldKind::Bool,
        },
        values: value
            .values
            .into_iter()
            .flatten()
            .map(|v| (*v).to_owned())
            .collect(),
        unknown: value
            .stored_unknown
            .map_or(UnknownRule::Unknown, |v| UnknownRule::Word(v.into())),
        help: value.help.into(),
        read_as: value.help.into(),
        unit: Some(value.unit.name().into()),
    }
}

pub fn catalog(origins: &[String], commands: &[String]) -> Catalog {
    let mut catalog = Catalog::new("sessions");
    catalog.recall_last_unit = Some("messages".into());
    catalog.commands = commands.to_vec();
    catalog.units = Unit::all()
        .map(|unit| {
            let spec = units::spec(unit);
            let mut declarations: Vec<Field> = fields::FIELDS
                .iter()
                .filter(|f| f.unit == unit)
                .map(|f| field(f, f.name.into()))
                .collect();
            if fields::has_session(unit) {
                for f in fields::FIELDS.iter().filter(|f| f.unit == Unit::Sessions) {
                    if !declarations.iter().any(|d| d.name == f.name) {
                        declarations.push(field(f, f.name.into()));
                    }
                }
            }
            for relation in spec.relations {
                for f in fields::FIELDS.iter().filter(|f| f.unit == relation.target) {
                    declarations.push(field(f, format!("{}.{}", relation.prefix, f.name)));
                }
            }
            // These reference atoms are compiled at the product identity seam.
            for name in origins
                .iter()
                .map(String::as_str)
                .chain(["session", "message", "block", "note"])
            {
                if !declarations.iter().any(|f| f.name == name) {
                    declarations.push(Field {
                        name: name.into(),
                        kind: FieldKind::Reference,
                        ..Field::default()
                    });
                }
            }
            query_language::catalog::Unit {
                name: spec.name.into(),
                aliases: vec![spec.singular.into()],
                sequence_unit: Some("tool_calls".into()),
                fields: declarations,
                identity: spec
                    .identity
                    .iter()
                    .filter_map(|sql| column_name(spec, sql))
                    .map(str::to_owned)
                    .collect(),
                default_order: column_name(spec, spec.time)
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                main_time: column_name(spec, spec.time).map(str::to_owned),
                ..query_language::catalog::Unit::default()
            }
        })
        .collect();
    catalog.references = vec![ReferenceSyntax {
        prefix: "@".into(),
        marker: None,
    }];
    catalog
}

pub use query_language::catalog::{AGGREGATES, KEY_FUNCTIONS, STAGES};
pub use query_language::complete::{Completion, CompletionKind};
pub fn complete(text: &str, position: usize, catalog: &Catalog) -> Vec<Completion> {
    query_language::complete::complete(text, position, catalog)
}

fn column_name<'a>(unit: &'a units::UnitSpec, sql: &str) -> Option<&'a str> {
    unit.columns.iter().find_map(|(name, expr, _)| match expr {
        fields::Expr::Template(template) | fields::Expr::Reference(template, _)
            if *template == sql =>
        {
            Some(*name)
        }
        _ => None,
    })
}
