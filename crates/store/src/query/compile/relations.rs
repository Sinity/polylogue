//! Relational stage boundaries. Logical output names are independent of SQL slots.
use super::*;

mod order;

#[derive(Clone)]
pub(super) struct NativeSource {
    pub prefix: String,
    pub unit: Unit,
    pub key: String,
}

pub(super) struct Relation {
    pub program: Program,
    pub sources: Vec<NativeSource>,
    pub open_fields: bool,
    next_slot: usize,
}

impl Relation {
    pub fn new(program: Program) -> Self {
        let sources = if matches!(program.row_basis, RowBasis::Native { .. }) {
            vec![NativeSource {
                prefix: String::new(),
                unit: program.unit,
                key: "_pk".into(),
            }]
        } else {
            vec![]
        };
        Self {
            program,
            sources,
            open_fields: true,
            next_slot: 0,
        }
    }
    pub fn slot(&mut self) -> String {
        self.next_slot += 1;
        format!("_rel_value{}", self.next_slot)
    }
    pub fn field(
        &self,
        compiler: &mut Compiler<'_>,
        name: &str,
        at: usize,
    ) -> Result<(String, ColKind), QueryError> {
        self.field_on(compiler, name, at, "r")
    }
    pub fn field_on(
        &self,
        compiler: &mut Compiler<'_>,
        name: &str,
        at: usize,
        alias: &str,
    ) -> Result<(String, ColKind), QueryError> {
        if let Some(column) = self.program.columns.iter().find(|c| c.name == name) {
            let sql = format!("{alias}.{}", column_sql(self.program.column_slot(name)));
            return Ok((
                if column.kind == ColKind::Money {
                    format!("({sql}) * 1e9")
                } else {
                    sql
                },
                column.kind,
            ));
        }
        if self.open_fields {
            for source in self.sources.iter().rev() {
                if let Some(field) = name.strip_prefix(&source.prefix) {
                    let context = RowContext {
                        unit: source.unit,
                        a: compiler.fresh(),
                    };
                    if let Ok((value, kind)) = compiler.expr(field, &context, at) {
                        let key = context.a.expand(units::spec(source.unit).key);
                        return Ok((
                            format!(
                                "(SELECT {value} FROM {} WHERE {key} = {alias}.{})",
                                base_from(source.unit, &context.a),
                                column_sql(&source.key)
                            ),
                            kind,
                        ));
                    }
                }
            }
        }
        Err(QueryError::Invalid {
            message: format!("unknown relation column {name:?}"),
            position: at,
        })
    }
    pub fn reference(&self, value: &Expression) -> Option<ColumnReference> {
        match value {
            Expression::Field { name } => self
                .program
                .columns
                .iter()
                .find(|c| &c.name == name)
                .and_then(|c| c.reference)
                .or_else(|| {
                    self.open_fields
                        .then(|| {
                            self.sources.iter().rev().find_map(|source| {
                                name.strip_prefix(&source.prefix).and_then(|name| {
                                    Compiler::field_reference(name, &RowContext::plain(source.unit))
                                })
                            })
                        })
                        .flatten()
                }),
            Expression::Function { name, arguments } if name == "case_eq" => {
                arguments.get(2).and_then(|arg| self.reference(arg))
            }
            _ => None,
        }
    }
}

impl<'a> Compiler<'a> {
    pub(super) fn program_with(
        self,
        query: &Query,
        preset: Option<MatchRel>,
    ) -> Result<Program, QueryError> {
        self.native_program_with(query, preset)
    }
}
