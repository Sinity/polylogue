//! Provider-owned point projection. Each advance performs at most one state
//! lookup and emits at most one whole admitted cell or annotation. It never
//! assembles a result string, response, or wrapper children array. The caller
//! pins the ReadView and cursor to one committed parser/evidence revision.
//!
//! Message projection does not consume session-scoped unresolved native items
//! or Effects obligations. Those remain separately retained; neither this
//! cursor nor byte EOF certifies normalization, source rank, or version hashes.
use super::keyed::{Cell, Key, MemoryReadView, ReadView};
use polylogue_model::RetainedJson as Value;
use polylogue_model::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Message,
    Annotated,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
enum Phase {
    Header,
    MessageWitness(String),
    Block,
    BlockPresence,
    PendingResult,
    ChunkZero,
    ChunkAfter(usize),
    Continuation,
    ChildCount,
    Child(usize),
    AdvanceBlock,
    Done,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum Scope {
    Message,
    Header,
    Block(usize),
}
/// Persist only with its pinned predecessor and after the emitted unit commits.
/// A read failure fuses this cursor; restart from the last committed cursor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cursor {
    message: usize,
    mode: Mode,
    scope: Scope,
    phase: Phase,
    id: Option<MessageId>,
    blocks: usize,
    block: usize,
    call: Option<String>,
    wrapper: bool,
    children: usize,
    failed: bool,
    after_presence: Option<Phase>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unit {
    Header(Message),
    MessageWitness(super::presence::Witness),
    BlockPresence {
        index: usize,
        group: Option<super::presence::Group>,
    },
    Block {
        index: usize,
        value: Block,
    },
    /// The result's effective output is replaced by the following chunk units.
    /// No chunk means absent output; an empty chunk means present empty text.
    ResultStart {
        block: usize,
    },
    ResultChunk {
        block: usize,
        slot: usize,
        text: String,
    },
    /// Closes the chunk stream, including a stream with no chunks.
    ResultEnd {
        block: usize,
    },
    Continuation {
        call_id: String,
    },
    /// Replaces a previous entry for this wrapper, as the source map did.
    WrapperStart {
        call_id: String,
        children: usize,
    },
    WrapperChild {
        call_id: String,
        index: usize,
        child: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Unit(Unit),
    Continue,
    Absent,
    Done,
}
impl Cursor {
    pub fn new(message: usize, mode: Mode) -> Self {
        Self {
            message,
            mode,
            scope: Scope::Message,
            phase: Phase::Header,
            id: None,
            blocks: 0,
            block: 0,
            call: None,
            wrapper: false,
            children: 0,
            failed: false,
            after_presence: None,
        }
    }
    /// Select only the current header; no witness or block enumeration.
    pub fn header(message: usize, mode: Mode) -> Self {
        let mut cursor = Self::new(message, mode);
        cursor.scope = Scope::Header;
        cursor
    }
    /// Select one block through this same interpretation/projection machine.
    /// Header/ID reanchor, presence, result chunks and annotations are unchanged.
    pub fn block(message: usize, block: usize, mode: Mode) -> Self {
        let mut cursor = Self::new(message, mode);
        cursor.scope = Scope::Block(block);
        cursor
    }
    pub fn advance(&mut self, view: &impl ReadView) -> Result<Step, String> {
        if self.failed {
            return Err("Codex projection cursor already failed".into());
        }
        let result = self.step(view);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn cell(&self, view: &impl ReadView, key: &Key) -> Result<Option<Cell>, String> {
        let value = view.read(key)?;
        if value.as_ref().is_some_and(|v| !v.valid_for(key)) {
            return Err(format!("invalid Codex projection cell {key:?}"));
        }
        Ok(value)
    }
    fn call_id(&self) -> Result<String, String> {
        self.call
            .clone()
            .ok_or_else(|| "missing projection call identity".into())
    }
    fn chunk(&self, slot: usize, value: Cell) -> Result<Step, String> {
        match value {
            Cell::Text(text) => Ok(Step::Unit(Unit::ResultChunk {
                block: self.block,
                slot,
                text,
            })),
            _ => Err("invalid Codex output chunk".into()),
        }
    }
    fn step(&mut self, view: &impl ReadView) -> Result<Step, String> {
        match self.phase {
            Phase::Header => {
                let Some(Cell::Header { message, blocks }) =
                    self.cell(view, &Key::Message(self.message))?
                else {
                    self.phase = Phase::Done;
                    return Ok(Step::Absent);
                };
                self.id = Some(message.id.clone());
                self.blocks = blocks;
                self.phase = match self.scope {
                    Scope::Message => Phase::MessageWitness(String::new()),
                    Scope::Header => Phase::Done,
                    Scope::Block(index) => {
                        if index >= blocks {
                            return Err(
                                "selected Codex projection block exceeds header count".into()
                            );
                        }
                        self.block = index;
                        Phase::Block
                    }
                };
                Ok(Step::Unit(Unit::Header(message)))
            }
            Phase::MessageWitness(ref after) => {
                let key = Key::Presence(self.message, after.clone());
                match view.next(&key)? {
                    Some((Key::Presence(message, name), Cell::Witness(w)))
                        if message == self.message && name > *after =>
                    {
                        self.phase = Phase::MessageWitness(name);
                        Ok(Step::Unit(Unit::MessageWitness(w)))
                    }
                    Some((found, _)) if found <= key || !found.same_family(&key) => {
                        Err("invalid Codex presence neighbor".into())
                    }
                    Some((Key::Presence(message, _), _)) if message == self.message => {
                        Err("invalid Codex message witness".into())
                    }
                    _ => {
                        self.phase = Phase::Block;
                        Ok(Step::Continue)
                    }
                }
            }
            Phase::BlockPresence => {
                let value = self.cell(view, &Key::PresenceBlock(self.message, self.block))?;
                self.phase = self
                    .after_presence
                    .take()
                    .ok_or("missing Codex block projection phase")?;
                let group = match value {
                    Some(Cell::Presence(group)) => Some(group),
                    None => None,
                    _ => return Err("invalid Codex block presence".into()),
                };
                Ok(Step::Unit(Unit::BlockPresence {
                    index: self.block,
                    group,
                }))
            }
            Phase::Block => {
                if self.block > self.blocks {
                    return Err("invalid Codex projection block cursor".into());
                }
                if self.block == self.blocks {
                    self.phase = Phase::Done;
                    return Ok(Step::Done);
                }
                let Some(Cell::Block(mut value)) =
                    self.cell(view, &Key::Block(self.message, self.block))?
                else {
                    return Err("missing Codex projection block".into());
                };
                let id = self
                    .id
                    .clone()
                    .ok_or("missing projection message identity")?;
                value.id = match value.id.slot() {
                    BlockSlot::Item(n) => BlockId::new(id, *n),
                    BlockSlot::Reasoning(n) => BlockId::reasoning(id, *n),
                    BlockSlot::Attachment(key) => {
                        BlockId::attachment(id, key).map_err(|e| e.to_string())?
                    }
                };
                self.call = None;
                self.wrapper = false;
                self.phase = match &value.content {
                    BlockContent::ToolResult(_) => Phase::PendingResult,
                    BlockContent::ToolCall(c)
                        if self.mode == Mode::Annotated && c.call_id.is_some() =>
                    {
                        self.call = c.call_id.as_ref().map(|id| id.as_str().to_owned());
                        self.wrapper = c.tool.as_deref() == Some("exec");
                        Phase::Continuation
                    }
                    _ => Phase::AdvanceBlock,
                };
                self.after_presence = Some(self.phase.clone());
                self.phase = Phase::BlockPresence;
                Ok(Step::Unit(Unit::Block {
                    index: self.block,
                    value,
                }))
            }
            Phase::PendingResult => {
                let pending = self
                    .cell(view, &Key::PendingResult(self.message, self.block))?
                    .is_some();
                self.phase = if pending {
                    Phase::ChunkZero
                } else {
                    Phase::AdvanceBlock
                };
                Ok(if pending {
                    Step::Unit(Unit::ResultStart { block: self.block })
                } else {
                    Step::Continue
                })
            }
            Phase::ChunkZero => {
                let value = self.cell(view, &Key::OutputChunk(self.message, self.block, 0))?;
                self.phase = Phase::ChunkAfter(0);
                value.map_or(Ok(Step::Continue), |v| self.chunk(0, v))
            }
            Phase::ChunkAfter(slot) => {
                let key = Key::OutputChunk(self.message, self.block, slot);
                match view.next(&key)? {
                    Some((Key::OutputChunk(message, block, next), value))
                        if message == self.message && block == self.block && next > slot =>
                    {
                        self.phase = Phase::ChunkAfter(next);
                        self.chunk(next, value)
                    }
                    Some((found, _)) if !found.same_family(&key) || found <= key => {
                        Err("invalid Codex projection neighbor order/family".into())
                    }
                    _ => {
                        self.phase = Phase::AdvanceBlock;
                        Ok(Step::Unit(Unit::ResultEnd { block: self.block }))
                    }
                }
            }
            Phase::Continuation => {
                let call_id = self.call_id()?;
                let continued = self
                    .cell(view, &Key::Continuation(call_id.clone()))?
                    .is_some();
                self.phase = if self.wrapper {
                    Phase::ChildCount
                } else {
                    Phase::AdvanceBlock
                };
                Ok(if continued {
                    Step::Unit(Unit::Continuation { call_id })
                } else {
                    Step::Continue
                })
            }
            Phase::ChildCount => {
                let call_id = self.call_id()?;
                self.children = match self.cell(view, &Key::ChildCount(call_id.clone()))? {
                    Some(Cell::Number(n)) => n,
                    None => 0,
                    _ => return Err("invalid Codex child count".into()),
                };
                self.phase = if self.children > 0 {
                    Phase::Child(0)
                } else {
                    Phase::AdvanceBlock
                };
                Ok(if self.children > 0 {
                    Step::Unit(Unit::WrapperStart {
                        call_id,
                        children: self.children,
                    })
                } else {
                    Step::Continue
                })
            }
            Phase::Child(index) => {
                if index >= self.children {
                    return Err("invalid Codex projection child cursor".into());
                }
                let call_id = self.call_id()?;
                let Some(Cell::Text(child)) =
                    self.cell(view, &Key::Child(call_id.clone(), index))?
                else {
                    return Err("missing Codex wrapper child".into());
                };
                self.phase = if index + 1 == self.children {
                    Phase::AdvanceBlock
                } else {
                    Phase::Child(index + 1)
                };
                Ok(Step::Unit(Unit::WrapperChild {
                    call_id,
                    index,
                    child,
                }))
            }
            Phase::AdvanceBlock => {
                if matches!(self.scope, Scope::Block(_)) {
                    self.phase = Phase::Done;
                    return Ok(Step::Continue);
                }
                self.block = self
                    .block
                    .checked_add(1)
                    .ok_or("Codex projection block index overflow")?;
                self.phase = Phase::Block;
                Ok(Step::Continue)
            }
            Phase::Done => Ok(Step::Done),
        }
    }
}

/// Explicit unbounded inspection collector. Production finalizers consume
/// units directly; only this helper assembles full strings and arrays.
pub fn collect(
    view: &impl ReadView,
    message: usize,
    mode: Mode,
) -> Result<Option<Message>, String> {
    let mut cursor = Cursor::new(message, mode);
    let mut result: Option<Message> = None;
    let mut continued = Vec::new();
    let mut children = std::collections::BTreeMap::new();
    loop {
        match cursor.advance(view)? {
            Step::Absent => return Ok(None),
            Step::Done => break,
            Step::Continue => {}
            Step::Unit(Unit::Header(message)) => result = Some(message),
            Step::Unit(unit) => {
                let m = result.as_mut().ok_or("projection unit before header")?;
                match unit {
                    Unit::Block { value, .. } => m.blocks.push(value),
                    Unit::ResultStart { block } => {
                        let Some(Block {
                            content: BlockContent::ToolResult(r),
                            ..
                        }) = m.blocks.get_mut(block)
                        else {
                            return Err("projection result owner missing".into());
                        };
                        r.output = None;
                    }
                    Unit::ResultChunk { block, text, .. } => {
                        let Some(Block {
                            content: BlockContent::ToolResult(r),
                            ..
                        }) = m.blocks.get_mut(block)
                        else {
                            return Err("projection chunk owner missing".into());
                        };
                        match &mut r.output {
                            None => r.output = Some(ToolOutput::Text(text)),
                            Some(ToolOutput::Text(output)) => output.push_str(&text),
                            _ => return Err("projection chunk replaced nontext output".into()),
                        }
                    }
                    Unit::ResultEnd { .. }
                    | Unit::MessageWitness(_)
                    | Unit::BlockPresence { .. } => {}
                    Unit::Continuation { call_id } => continued.push(Value::from(call_id)),
                    Unit::WrapperStart { call_id, .. } => {
                        children.insert(call_id.into(), Value::Array(Vec::new()));
                    }
                    Unit::WrapperChild { call_id, child, .. } => {
                        let Some(Value::Array(ids)) =
                            children.get_mut(&polylogue_model::NativeId::unicode(&call_id))
                        else {
                            return Err("projection child before wrapper".into());
                        };
                        ids.push(Value::from(child));
                    }
                    Unit::Header(_) => return Err("duplicate projection header".into()),
                }
            }
        }
    }
    let Some(mut m) = result else {
        return Err("projection completed without header".into());
    };
    if !continued.is_empty() || !children.is_empty() {
        let attrs = m.attrs.get_or_insert_with(|| serde_json::json!({}).into());
        if let Some(o) = attrs.as_object_mut() {
            if !continued.is_empty() {
                o.insert("continuation_calls".into(), Value::Array(continued));
            }
            if !children.is_empty() {
                o.insert("code_children".into(), Value::Object(children));
            }
        }
    }
    Ok(Some(m))
}
impl MemoryReadView {
    pub fn message(&self, i: usize) -> Result<Option<Message>, String> {
        collect(self, i, Mode::Message)
    }
    pub fn projected_message(&self, i: usize) -> Result<Option<Message>, String> {
        collect(self, i, Mode::Annotated)
    }
}
