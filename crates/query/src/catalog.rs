//! The language vocabulary is projected from the product's execution registry.
use crate::{ast::Unit, fields, units};
use query_language::catalog::{Catalog, Field, FieldKind, ReferenceForm, UnknownRule};

pub const MEASURE: &str = "measure";

fn field(value: &fields::Field, name: String) -> Field {
    use fields::Kind;
    let target = fields::reference_target(value.unit, value.name);
    let kind = if value.value_kind == Some(crate::ColKind::Ids) {
        FieldKind::List(Box::new(FieldKind::Reference))
    } else if target.is_some() && value.column_kind() == crate::ColKind::Text {
        FieldKind::Reference
    } else if value.unit == Unit::Sessions && value.name == "cwd" {
        FieldKind::Subtree
    } else {
        match value.kind {
            Kind::Match | Kind::Special => FieldKind::Exact,
            Kind::Prefix => FieldKind::Reference,
            Kind::Substring => FieldKind::Substring,
            Kind::Subtree => FieldKind::Subtree,
            Kind::Number => FieldKind::Number,
            Kind::Money => FieldKind::Money,
            Kind::Duration => FieldKind::Duration,
            Kind::Time => FieldKind::Time,
            Kind::Bool => FieldKind::Bool,
        }
    };
    let reference = matches!(kind, FieldKind::Reference)
        || matches!(&kind, FieldKind::List(element) if **element == FieldKind::Reference);
    Field {
        name,
        kind,
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
        unit: if reference {
            target.map(|unit| unit.name().into())
        } else {
            Some(value.unit.name().into())
        },
    }
}

pub fn catalog(origins: &[String]) -> Catalog {
    let mut catalog = Catalog::new("sessions");
    catalog.namespace = Some("pl".into());
    catalog.functions.push(query_language::catalog::Function {
        name: "program".into(),
        arguments: 1,
        result: FieldKind::Exact,
        help: "The program invoked by a command".into(),
    });
    use query_language::catalog::{Extension, ExtensionPosition, ExtensionSlot};
    catalog.extensions.push(Extension {
        name: MEASURE.into(),
        position: ExtensionPosition::Transform,
        arguments: vec![
            ExtensionSlot::Value,
            ExtensionSlot::Optional(vec![
                ExtensionSlot::Keyword("by".into()),
                ExtensionSlot::List(Box::new(ExtensionSlot::Expression)),
            ]),
        ],
        help: "Measure the selected sessions".into(),
    });
    catalog.units = Unit::all()
        .map(|unit| {
            let spec = units::spec(unit);
            let mut declarations: Vec<Field> = fields::FIELDS
                .iter()
                .filter(|f| f.unit == unit)
                .map(|f| field(f, f.name.into()))
                .collect();
            if fields::has_session(unit) {
                if !declarations.iter().any(|field| field.name == "session") {
                    declarations.push(Field {
                        name: "session".into(),
                        kind: FieldKind::Reference,
                        unit: fields::reference_target(unit, "session")
                            .map(|unit| unit.name().into()),
                        help: "The session containing this row".into(),
                        ..Field::default()
                    });
                }
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
            query_language::catalog::Unit {
                name: spec.name.into(),
                sequence_unit: Some("tool_calls".into()),
                order_scope: (unit == Unit::ToolCalls).then(|| "session_tool_calls".into()),
                fields: declarations,
                identity: spec
                    .identity
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect(),
                default_order: spec.main_time.into_iter().map(str::to_owned).collect(),
                main_time: spec.main_time.map(str::to_owned),
                ..query_language::catalog::Unit::default()
            }
        })
        .collect();
    catalog
        .units
        .extend(units::DESIGNED_UNITS.iter().map(|designed| {
            query_language::catalog::Unit {
                name: designed.name.into(),
                fields: designed
                    .columns
                    .iter()
                    .map(|column| Field {
                        name: column.name.into(),
                        kind: crate::compile::source::column_kind(column.kind),
                        help: format!("{} of a designed {} row", column.name, designed.name),
                        ..Field::default()
                    })
                    .collect(),
                help: format!("{} (not built yet)", designed.name),
                ..query_language::catalog::Unit::default()
            }
        }));
    catalog.references = [
        ("@", None),
        ("session:", Some(":")),
        ("message:", Some(":")),
        ("block:", None),
        ("document:", Some(":")),
        ("event:", None),
        ("span:", Some("#")),
        ("assertion:", None),
        ("delivery:", None),
        ("run:", None),
        ("page:", None),
        ("file:", Some("@")),
        ("https://", None),
        ("http://", None),
    ]
    .into_iter()
    .map(|(prefix, marker)| ReferenceForm::Prefix {
        prefix: prefix.into(),
        marker: marker.map(str::to_owned),
    })
    .collect();
    catalog.references.extend(
        origins
            .iter()
            .map(|origin| ReferenceForm::prefix(format!("{origin}:"))),
    );
    catalog.references.extend(
        ["self", "last", "latest"]
            .into_iter()
            .map(ReferenceForm::word),
    );
    catalog
        .references
        .push(ReferenceForm::Identifier { min_length: 8 });
    catalog
}

pub use query_language::catalog::{AGGREGATES, FUNCTIONS, TRANSFORMS};
pub use query_language::complete::{Completion, CompletionKind};
pub fn complete(text: &str, position: usize, catalog: &Catalog) -> Vec<Completion> {
    query_language::complete::complete(text, position, catalog)
}

/// The row domain a read target selects; every target has one.
pub fn target_unit(target: &polylogue_model::read::ReadTarget) -> Unit {
    use polylogue_model::read::ReadTarget;
    match target {
        ReadTarget::Session(_) => Unit::Sessions,
        ReadTarget::Message(_) | ReadTarget::MessageRange { .. } => Unit::Messages,
        ReadTarget::Block { .. } => Unit::Blocks,
        ReadTarget::Document(_) => Unit::Documents,
        ReadTarget::Event(_) => Unit::Events,
        ReadTarget::Assertion { .. } => Unit::Assertions,
        ReadTarget::Delivery(_) => Unit::Deliveries,
    }
}

/// The native row domain of a typed address. This is pure product vocabulary;
/// membership and source availability remain Store responsibilities.
pub fn reference_unit(reference: &polylogue_model::Reference) -> Option<Unit> {
    use polylogue_model::Reference;
    Some(match reference {
        Reference::Session(_) => Unit::Sessions,
        Reference::Message(_) | Reference::MessageRange { .. } => Unit::Messages,
        Reference::Block { .. } => Unit::Blocks,
        Reference::Document(_) => Unit::Documents,
        Reference::Event(_) => Unit::Events,
        Reference::Assertion(_) => Unit::Assertions,
        Reference::Delivery(_) => Unit::Deliveries,
        Reference::Span { container, .. } => return reference_unit(container),
        _ => return None,
    })
}
