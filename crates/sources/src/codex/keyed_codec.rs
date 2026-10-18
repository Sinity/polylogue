//! Byte encoding of Codex state keys. The store sorts cells by these bytes, so
//! byte order must equal `Key` order. Changing the encoding needs a new state
//! layout version.
use super::keyed::Key;

impl Key {
    pub fn family(&self) -> &'static str {
        match self {
            Self::CompletedMessage(_, _) => "completed_message",
            Self::PromptEcho(_) => "prompt_echo",
            Self::ResponseScope(_) => "response_scope",
            Self::Operation(_) => "operation",
            Self::UnifiedOperation(_) => "unified_operation",
            Self::OperationBlock(_, _) => "operation_block",
            Self::Message(_) => "message",
            Self::Block(_, _) => "block",
            Self::Call(_) => "call",
            Self::CallBlock(_, _, _) => "call_block",
            Self::Reasoning(_) => "reasoning",
            Self::Running(_) => "running",
            Self::Outcome(_) => "outcome",
            Self::Continuation(_) => "continuation",
            Self::PendingStdin(_) => "pending_stdin",
            Self::Wrapper(_) => "wrapper",
            Self::WrapperCall(_, _) => "wrapper_call",
            Self::Child(_, _) => "child",
            Self::ChildCount(_) => "child_count",
            Self::Occurrence(_) => "occurrence",
            Self::Model(_) => "model",
            Self::Mutable(_, _) => "mutable",
            Self::Prompt => "prompt",
            Self::BlockOrder(_) => "block_order",
            Self::Evidence(_, _, _) => "evidence",
            Self::EvidenceHead(_) => "evidence_head",
            Self::EvidenceItem(_, _) => "evidence_item",
            Self::Context(_) => "context",
            Self::CallAssociation(_) => "call_association",
            Self::StatusEvidence(_) => "status_evidence",
            Self::Result(_) => "result",
            Self::NativeItem(_, _) => "native_item",
            Self::OutputChunk(_, _, _) => "output_chunk",
            Self::PendingResult(_, _) => "pending_result",
            Self::ContinuationTarget(_) => "continuation_target",
            Self::Presence(_, _) => "presence",
            Self::PresenceBlock(_, _) => "presence_block",
        }
    }
    /// Encoded size, without a limit.
    pub fn ordered_len(&self) -> usize {
        self.ordered_len_limited(usize::MAX).unwrap_or(usize::MAX)
    }
    /// Encoded size, or `Err` with the needed size (or a lower bound) when it
    /// exceeds `allowance`. Checks a cheap lower bound before scanning strings.
    pub fn ordered_len_limited(&self, allowance: usize) -> Result<usize, usize> {
        if let Self::NativeItem(scope, id) | Self::CompletedMessage(scope, id) = self {
            let lower = 5usize.saturating_add(scope.len()).saturating_add(id.len());
            if lower > allowance {
                return Err(lower);
            }
            let required =
                lower.saturating_add(scope.bytes().chain(id.bytes()).filter(|b| *b == 0).count());
            return if required > allowance {
                Err(required)
            } else {
                Ok(required)
            };
        }
        let (fixed, text): (usize, Option<&str>) = match self {
            Self::Message(_)
            | Self::Wrapper(_)
            | Self::Model(_)
            | Self::BlockOrder(_)
            | Self::EvidenceHead(_)
            | Self::Context(_)
            | Self::PromptEcho(_) => (9, None),
            Self::Evidence(_, _, _) | Self::OutputChunk(_, _, _) => (25, None),
            Self::PendingResult(_, _) => (17, None),
            Self::EvidenceItem(_, s) | Self::Presence(_, s) => (11, Some(s)),
            Self::Block(_, _) | Self::PresenceBlock(_, _) | Self::OperationBlock(_, _) => {
                (17, None)
            }
            Self::Call(s)
            | Self::Reasoning(s)
            | Self::Running(s)
            | Self::Outcome(s)
            | Self::Continuation(s)
            | Self::PendingStdin(s)
            | Self::ChildCount(s)
            | Self::Occurrence(s)
            | Self::CallAssociation(s)
            | Self::StatusEvidence(s)
            | Self::Result(s)
            | Self::ContinuationTarget(s)
            | Self::Operation(s)
            | Self::UnifiedOperation(s)
            | Self::ResponseScope(s) => (3, Some(s)),
            Self::CallBlock(s, _, _) => (19, Some(s)),
            Self::WrapperCall(s, _) | Self::Child(s, _) | Self::Mutable(_, s) => (11, Some(s)),
            Self::Prompt => (1, None),
            Self::NativeItem(_, _) | Self::CompletedMessage(_, _) => unreachable!("handled above"),
        };
        let lower = fixed.saturating_add(text.map_or(0, str::len));
        if lower > allowance {
            return Err(lower);
        }
        let required =
            lower.saturating_add(text.map_or(0, |s| s.bytes().filter(|b| *b == 0).count()));
        if required > allowance {
            Err(required)
        } else {
            Ok(required)
        }
    }
    /// Write the encoding to `w`.
    pub fn write_ordered(&self, out: &mut impl std::io::Write) -> std::io::Result<()> {
        fn number(out: &mut impl std::io::Write, n: usize) -> std::io::Result<()> {
            out.write_all(&(n as u64).to_be_bytes())
        }
        fn text(out: &mut impl std::io::Write, s: &str) -> std::io::Result<()> {
            let mut start = 0;
            for (i, b) in s.bytes().enumerate() {
                if b == 0 {
                    out.write_all(&s.as_bytes()[start..i])?;
                    out.write_all(&[0, 1])?;
                    start = i + 1;
                }
            }
            out.write_all(&s.as_bytes()[start..])?;
            out.write_all(&[0, 0])
        }
        match self {
            Self::ResponseScope(id) => {
                out.write_all(&[36])?;
                text(out, id)?;
            }
            Self::CompletedMessage(turn, id) => {
                out.write_all(&[34])?;
                text(out, turn)?;
                text(out, id)?;
            }
            Self::PromptEcho(message) => {
                out.write_all(&[35])?;
                number(out, *message)?;
            }
            Self::UnifiedOperation(s) => {
                out.write_all(&[33])?;
                text(out, s)?;
            }
            Self::OperationBlock(m, b) => {
                out.write_all(&[32])?;
                number(out, *m)?;
                number(out, *b)?;
            }
            Self::Operation(s) => {
                out.write_all(&[31])?;
                text(out, s)?;
            }
            Self::Message(n) => {
                out.write_all(&[0])?;
                number(out, *n)?;
            }
            Self::Block(n, b) => {
                out.write_all(&[1])?;
                number(out, *n)?;
                number(out, *b)?;
            }
            Self::Call(s) => {
                out.write_all(&[2])?;
                text(out, s)?;
            }
            Self::CallBlock(s, n, b) => {
                out.write_all(&[3])?;
                text(out, s)?;
                number(out, *n)?;
                number(out, *b)?;
            }
            Self::Reasoning(s) => {
                out.write_all(&[4])?;
                text(out, s)?;
            }
            Self::Running(s) => {
                out.write_all(&[5])?;
                text(out, s)?;
            }
            Self::Outcome(s) => {
                out.write_all(&[6])?;
                text(out, s)?;
            }
            Self::Continuation(s) => {
                out.write_all(&[7])?;
                text(out, s)?;
            }
            Self::PendingStdin(s) => {
                out.write_all(&[8])?;
                text(out, s)?;
            }
            Self::Wrapper(n) => {
                out.write_all(&[9])?;
                number(out, *n)?;
            }
            Self::WrapperCall(s, n) => {
                out.write_all(&[10])?;
                text(out, s)?;
                number(out, *n)?;
            }
            Self::Child(s, n) => {
                out.write_all(&[11])?;
                text(out, s)?;
                number(out, *n)?;
            }
            Self::ChildCount(s) => {
                out.write_all(&[12])?;
                text(out, s)?;
            }
            Self::Occurrence(s) => {
                out.write_all(&[13])?;
                text(out, s)?;
            }
            Self::Model(n) => {
                out.write_all(&[14])?;
                number(out, *n)?;
            }
            Self::Mutable(n, s) => {
                out.write_all(&[15])?;
                number(out, *n)?;
                text(out, s)?;
            }
            Self::Prompt => out.write_all(&[16])?,
            Self::BlockOrder(n) => {
                out.write_all(&[17])?;
                number(out, *n)?;
            }
            Self::Evidence(m, p, n) => {
                out.write_all(&[18])?;
                number(out, *m)?;
                number(out, *p)?;
                number(out, *n)?;
            }
            Self::EvidenceHead(n) => {
                out.write_all(&[19])?;
                number(out, *n)?;
            }
            Self::EvidenceItem(n, s) => {
                out.write_all(&[20])?;
                number(out, *n)?;
                text(out, s)?;
            }
            Self::Context(n) => {
                out.write_all(&[21])?;
                number(out, *n)?;
            }
            Self::CallAssociation(s) => {
                out.write_all(&[22])?;
                text(out, s)?;
            }
            Self::StatusEvidence(s) => {
                out.write_all(&[23])?;
                text(out, s)?;
            }
            Self::Result(s) => {
                out.write_all(&[24])?;
                text(out, s)?;
            }
            Self::NativeItem(scope, id) => {
                out.write_all(&[25])?;
                text(out, scope)?;
                text(out, id)?;
            }
            Self::OutputChunk(i, b, n) => {
                out.write_all(&[26])?;
                number(out, *i)?;
                number(out, *b)?;
                number(out, *n)?;
            }
            Self::PendingResult(i, b) => {
                out.write_all(&[27])?;
                number(out, *i)?;
                number(out, *b)?;
            }
            Self::ContinuationTarget(s) => {
                out.write_all(&[28])?;
                text(out, s)?;
            }
            Self::Presence(i, s) => {
                out.write_all(&[29])?;
                number(out, *i)?;
                text(out, s)?;
            }
            Self::PresenceBlock(i, b) => {
                out.write_all(&[30])?;
                number(out, *i)?;
                number(out, *b)?;
            }
        }
        Ok(())
    }
    /// The encoding as a `Vec`; byte order equals `Key` order.
    pub fn ordered_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.ordered_len());
        self.write_ordered(&mut out)
            .expect("Vec byte writes are infallible");
        out
    }
    pub fn from_ordered_bytes(bytes: &[u8]) -> Result<Self, String> {
        struct Reader<'a>(&'a [u8]);
        impl Reader<'_> {
            fn byte(&mut self) -> Result<u8, String> {
                let Some((v, rest)) = self.0.split_first() else {
                    return Err("truncated Codex cell key".into());
                };
                self.0 = rest;
                Ok(*v)
            }
            fn number(&mut self) -> Result<usize, String> {
                if self.0.len() < 8 {
                    return Err("truncated Codex cell ordinal".into());
                }
                let (n, rest) = self.0.split_at(8);
                self.0 = rest;
                usize::try_from(u64::from_be_bytes(n.try_into().unwrap()))
                    .map_err(|_| "Codex cell ordinal exceeds platform capacity".into())
            }
            fn text(&mut self) -> Result<String, String> {
                let mut out = Vec::new();
                loop {
                    match self.byte()? {
                        0 => match self.byte()? {
                            0 => {
                                return String::from_utf8(out)
                                    .map_err(|_| "invalid UTF8 in Codex cell key".into());
                            }
                            1 => out.push(0),
                            _ => return Err("invalid Codex cell string escape".into()),
                        },
                        b => out.push(b),
                    }
                }
            }
        }
        let mut r = Reader(bytes);
        let key = match r.byte()? {
            0 => Self::Message(r.number()?),
            1 => Self::Block(r.number()?, r.number()?),
            2 => Self::Call(r.text()?),
            3 => Self::CallBlock(r.text()?, r.number()?, r.number()?),
            4 => Self::Reasoning(r.text()?),
            5 => Self::Running(r.text()?),
            6 => Self::Outcome(r.text()?),
            7 => Self::Continuation(r.text()?),
            8 => Self::PendingStdin(r.text()?),
            9 => Self::Wrapper(r.number()?),
            10 => Self::WrapperCall(r.text()?, r.number()?),
            11 => Self::Child(r.text()?, r.number()?),
            12 => Self::ChildCount(r.text()?),
            13 => Self::Occurrence(r.text()?),
            14 => Self::Model(r.number()?),
            15 => Self::Mutable(r.number()?, r.text()?),
            16 => Self::Prompt,
            17 => Self::BlockOrder(r.number()?),
            18 => Self::Evidence(r.number()?, r.number()?, r.number()?),
            19 => Self::EvidenceHead(r.number()?),
            20 => Self::EvidenceItem(r.number()?, r.text()?),
            21 => Self::Context(r.number()?),
            22 => Self::CallAssociation(r.text()?),
            23 => Self::StatusEvidence(r.text()?),
            24 => Self::Result(r.text()?),
            25 => Self::NativeItem(r.text()?, r.text()?),
            26 => Self::OutputChunk(r.number()?, r.number()?, r.number()?),
            27 => Self::PendingResult(r.number()?, r.number()?),
            28 => Self::ContinuationTarget(r.text()?),
            29 => Self::Presence(r.number()?, r.text()?),
            30 => Self::PresenceBlock(r.number()?, r.number()?),
            31 => Self::Operation(r.text()?),
            32 => Self::OperationBlock(r.number()?, r.number()?),
            33 => Self::UnifiedOperation(r.text()?),
            34 => Self::CompletedMessage(r.text()?, r.text()?),
            35 => Self::PromptEcho(r.number()?),
            36 => Self::ResponseScope(r.text()?),
            _ => return Err("unknown Codex cell key tag".into()),
        };
        if !r.0.is_empty() {
            return Err("trailing bytes in Codex cell key".into());
        }
        Ok(key)
    }
}
