# Decision: parser library for the query language

Status: **decided.** Decided by the spec authors (a technical choice); the operator may overrule.

The existing grammar is a ~75-line Lark grammar plus hand-written splitting
of pipelines and the `with` clause. In Rust:

- **pest**: grammar in a separate PEG file, close to how the Lark grammar
  reads; error messages with positions; mature (2.x).
- **chumsky**: parser combinators in Rust code; excellent error recovery and
  messages; 1.0 still in alpha.
- **hand-written recursive descent**: most control over error messages, more
  code.

Decision: **pest** for the first version: the grammar stays a readable
file the spec can quote, and the whole language (including pipelines and
`with`) goes into one grammar instead of being split by hand as today. Revisit
if error messages are not good enough for agents, which matter most here.
