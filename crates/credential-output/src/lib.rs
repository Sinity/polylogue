//! Streaming masking for the fixed `credential-output-v1` catalog.
//!
//! Text that might still become a credential is held back, up to 512 UTF-8
//! bytes; going past that returns `InvalidData` without writing the held text.
//! Recognized secrets and PEM bodies are replaced as they stream, so they do
//! not accumulate. Any error closes the stream.
use std::collections::VecDeque;
use std::io::{self, Write};

mod word_ranges;
use word_ranges::is_word_char;

#[cfg(test)]
mod tests;

/// Most UTF-8 bytes held back while a match is undecided. Also caps open URL
/// and PEM header matches and the length of one PEM header.
const CARRY: usize = 512;
/// Most open token matches.
const MAX_CANDIDATES: usize = 2048;
/// Catalog families, numbered in catalog order. URL is 19 and PEM is 20; they
/// are matched by `Url` and `PemHeader`, not by patterns.
const FAMILIES: usize = 22;

fn whitespace(c: char) -> bool {
    // Python's str.isspace also counts U+001C..U+001F.
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn limit_reached() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "credential masking limit reached: too much undecided text",
    )
}

#[derive(Clone, Copy)]
enum Class {
    /// One character; the flag makes ASCII letters case-insensitive.
    Literal(char, bool),
    Alnum,
    Under,
    Url,
    Base,
    BasePad,
    Bearer,
    Hex,
    LowerHex,
    Digit,
    Dash,
    Space,
    Bare,
    User,
    /// Inside the given quote; the flag also stops at `:` (user part).
    Quoted(char, bool),
}

impl Class {
    fn accepts(self, c: char) -> bool {
        match self {
            Self::Literal(want, insensitive) => {
                c == want || insensitive && c.is_ascii() && c.eq_ignore_ascii_case(&want)
            }
            Self::Alnum => c.is_ascii_alphanumeric(),
            Self::Under => c.is_ascii_alphanumeric() || c == '_',
            Self::Url => c.is_ascii_alphanumeric() || "_-".contains(c),
            Self::Base => c.is_ascii_alphanumeric() || "+/".contains(c),
            Self::BasePad => c.is_ascii_alphanumeric() || "+/=".contains(c),
            Self::Bearer => c.is_ascii_alphanumeric() || "._~+/=-".contains(c),
            Self::Hex => c.is_ascii_hexdigit(),
            Self::LowerHex => c.is_ascii_digit() || ('a'..='f').contains(&c),
            Self::Digit => c.is_ascii_digit(),
            Self::Dash => c.is_ascii_alphanumeric() || c == '-',
            Self::Space => c == ' ' || c == '\t',
            Self::Bare => !whitespace(c) && c != '\'' && c != '"',
            Self::User => !whitespace(c) && !":'\"".contains(c),
            Self::Quoted(quote, user) => {
                c != quote && c != '\r' && c != '\n' && (!user || c != ':')
            }
        }
    }
}

#[derive(Clone)]
struct Part {
    class: Class,
    min: usize,
    max: usize,
}

fn run(class: Class, min: usize) -> Part {
    Part {
        class,
        min,
        max: usize::MAX,
    }
}

fn fixed(class: Class, count: usize) -> Part {
    Part {
        class,
        min: count,
        max: count,
    }
}

/// What must precede a match.
#[derive(Clone, Copy)]
enum Boundary {
    /// Start of block or a non-word character.
    Word,
    /// Start of block or whitespace.
    Space,
}

struct Pattern {
    family: usize,
    kind: &'static str,
    boundary: Boundary,
    parts: Vec<Part>,
    /// Index of the first masked part; earlier parts (labels) stay visible.
    secret: usize,
}

/// Builds a pattern's parts in order.
#[derive(Default)]
struct Shape {
    parts: Vec<Part>,
    secret: usize,
}

impl Shape {
    fn text(mut self, text: &str, insensitive: bool) -> Self {
        self.parts.extend(
            text.chars()
                .map(|c| fixed(Class::Literal(c, insensitive), 1)),
        );
        self
    }
    fn lit(self, text: &str) -> Self {
        self.text(text, false)
    }
    fn nocase(self, text: &str) -> Self {
        self.text(text, true)
    }
    fn then(mut self, part: Part) -> Self {
        self.parts.push(part);
        self
    }
    /// Mask from the next part on.
    fn secret(mut self) -> Self {
        self.secret = self.parts.len();
        self
    }
    fn build(self, family: usize, kind: &'static str, boundary: Boundary) -> Pattern {
        Pattern {
            family,
            kind,
            boundary,
            parts: self.parts,
            secret: self.secret,
        }
    }
    fn word(self, family: usize, kind: &'static str) -> Pattern {
        self.build(family, kind, Boundary::Word)
    }
}

fn patterns() -> Vec<Pattern> {
    use Class::*;
    let mut p = Vec::new();
    for prefix in ["AKIA", "ASIA", "ABIA", "ACCA"] {
        let shape = Shape::default().nocase(prefix).then(run(Alnum, 16));
        p.push(shape.word(0, "AWS access key"));
    }
    let github = ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"];
    let stripe = [
        "sk_test_", "sk_live_", "pk_test_", "pk_live_", "rk_test_", "rk_live_",
    ];
    let slack = ["xoxb-", "xoxp-", "xoxs-", "xoxa-", "xoxr-"];
    let openai = ["sk-proj-", "sk-svcacct-", "sk-admin-"];
    for (family, kind, prefixes, tail, min) in [
        (2, "GitHub token", &github[..], Under, 36),
        (3, "GitLab token", &["glpat-"], Url, 20),
        (4, "npm token", &["npm_"], Alnum, 36),
        (5, "Stripe key", &stripe, Alnum, 24),
        (6, "Slack token", &slack, Dash, 1),
        (8, "Google API key", &["AIza"], Url, 35),
        (14, "OpenAI key", &openai, Url, 20),
        (14, "OpenAI key", &["sk-"], Alnum, 48),
        (15, "Hugging Face token", &["hf_"], Alnum, 34),
        (18, "Twilio key", &["SK"], LowerHex, 32),
    ] {
        for prefix in prefixes {
            let shape = Shape::default().lit(prefix).then(run(tail, min));
            p.push(shape.word(family, kind));
        }
    }
    for prefix in ["sk-ant-api", "sk-ant-oat", "sk-ant-sid"] {
        p.push(
            Shape::default()
                .lit(prefix)
                .then(fixed(Digit, 2))
                .lit("-")
                .then(run(Url, 20))
                .word(13, "Anthropic key"),
        );
    }
    p.push(
        Shape::default()
            .then(fixed(Hex, 32))
            .nocase("-us")
            .then(run(Digit, 1))
            .word(16, "Mailchimp key"),
    );
    p.push(
        Shape::default()
            .lit("eyJ")
            .then(run(Url, 1))
            .lit(".eyJ")
            .then(run(Url, 1))
            .lit(".")
            .then(run(Url, 1))
            .word(7, "JWT"),
    );
    p.push(
        Shape::default()
            .lit("SG.")
            .then(run(Url, 20))
            .lit(".")
            .then(run(Url, 20))
            .word(17, "SendGrid token"),
    );
    for label in ["aws_secret_access_key", "secret_access_key", "aws_secret"] {
        for delimiter in [":", "="] {
            for quote in ["", "\"", "'"] {
                p.push(
                    Shape::default()
                        .nocase(label)
                        .then(run(Space, 0))
                        .lit(delimiter)
                        .then(run(Space, 0))
                        .lit(quote)
                        .secret()
                        .then(run(BasePad, 40))
                        .word(1, "AWS secret key"),
                );
            }
        }
    }
    p.push(
        Shape::default()
            .lit("AccountKey=")
            .secret()
            .then(run(BasePad, 44))
            .word(9, "Azure connection key"),
    );
    p.push(
        Shape::default()
            .nocase("Authorization")
            .then(run(Space, 0))
            .lit(":")
            .then(run(Space, 0))
            .nocase("Basic")
            .then(run(Space, 1))
            .secret()
            .then(fixed(Base, 8))
            .then(run(BasePad, 0))
            .word(11, "Basic authorization"),
    );
    p.push(
        Shape::default()
            .nocase("Bearer")
            .then(run(Space, 1))
            .secret()
            .then(run(Bearer, 20))
            .word(12, "Bearer token"),
    );
    let secret_flags = [
        "--password",
        "--token",
        "--secret",
        "--api-key",
        "--auth-token",
    ];
    for (family, kind, flags) in [
        (10, "CLI secret", &secret_flags[..]),
        (21, "CLI credentials", &["-u", "--user"][..]),
    ] {
        // Family 21 values are `user:password`; the user part stops at `:`.
        let user = family == 21;
        for flag in flags {
            for equals in [false, true] {
                for quote in [None, Some('\''), Some('"')] {
                    let mut shape = Shape::default().lit(flag);
                    shape = if equals {
                        shape.lit("=")
                    } else {
                        shape.then(run(Space, 1))
                    };
                    shape = shape.secret();
                    let quote_text = quote.map(String::from).unwrap_or_default();
                    shape = shape.lit(&quote_text);
                    let first = quote.map_or(if user { User } else { Bare }, |q| Quoted(q, user));
                    shape = shape.then(run(first, 1));
                    if user {
                        let rest = quote.map_or(Bare, |q| Quoted(q, false));
                        shape = shape.lit(":").then(run(rest, 1));
                    }
                    p.push(shape.lit(&quote_text).build(family, kind, Boundary::Space));
                }
            }
        }
    }
    p
}

/// One in-progress match of one pattern.
struct Candidate {
    pattern: usize,
    start: usize,
    /// Byte offset where the masked part starts.
    secret: usize,
    part: usize,
    count: usize,
    /// End offset of the longest complete match so far.
    accepted: Option<usize>,
    alive: bool,
    /// Its masked region has been recorded; it may still grow.
    committed: bool,
}

impl Candidate {
    fn step(&mut self, c: char, pos: usize, p: &Pattern) {
        while self.part < p.parts.len() {
            let part = &p.parts[self.part];
            if self.count < part.max && part.class.accepts(c) {
                if self.part == p.secret && self.count == 0 {
                    self.secret = pos;
                }
                // Unbounded runs only need to know the minimum was reached.
                self.count = self.count.saturating_add(1).min(if part.max == usize::MAX {
                    part.min.max(1)
                } else {
                    part.max
                });
                if self.count >= part.min && p.parts[self.part + 1..].iter().all(|x| x.min == 0) {
                    self.accepted = Some(pos + c.len_utf8());
                }
                return;
            }
            if self.count < part.min {
                self.alive = false;
                return;
            }
            self.part += 1;
            self.count = 0;
        }
        self.alive = false;
    }
}

/// A byte range to replace with one marker.
#[derive(Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
    kind: &'static str,
}

/// A held-back character.
struct Cell {
    pos: usize,
    c: char,
    region: Option<Region>,
}

/// A possible `scheme://user:password@host` match; masks `user:password`.
#[derive(Default)]
struct Url {
    /// Offset after `//`.
    authority: Option<usize>,
    /// 0: scheme, 1: after `:`, 2: after first `/`, 3: authority.
    stage: u8,
    colon: Option<usize>,
    /// Last `@` in the authority.
    at: Option<usize>,
}

impl Url {
    /// Returns false when `c` ends the match.
    fn step(&mut self, c: char, pos: usize) -> bool {
        match self.stage {
            0 if c.is_ascii_alphanumeric() || "+.-".contains(c) => true,
            0 if c == ':' => {
                self.stage = 1;
                true
            }
            1 | 2 if c == '/' => {
                self.stage += 1;
                if self.stage == 3 {
                    self.authority = Some(pos + 1);
                }
                true
            }
            3 if !whitespace(c) && !"/?#".contains(c) => {
                if c == ':' && self.colon.is_none() {
                    self.colon = Some(pos);
                }
                if c == '@' {
                    self.at = Some(pos);
                }
                true
            }
            _ => false,
        }
    }
    fn region(&self) -> Option<(usize, usize)> {
        let start = self.authority?;
        let colon = self.colon?;
        let at = self.at?;
        (colon > start && at > colon + 1).then_some((start, at))
    }
}

/// A possible `-----BEGIN ... PRIVATE KEY-----` line.
struct PemHeader {
    start: usize,
    text: String,
}

impl PemHeader {
    /// `Ok(Some(end_line))` when the header is complete, `Ok(None)` while it
    /// may still match, `Err` when it cannot.
    fn step(&mut self, c: char) -> Result<Option<String>, ()> {
        self.text.push(c);
        let prefix = "-----BEGIN ";
        if self.text.len() <= prefix.len() {
            return if prefix.starts_with(&self.text) {
                Ok(None)
            } else {
                Err(())
            };
        }
        let tail = &self.text[prefix.len()..];
        let label_end = tail.find('-').unwrap_or(tail.len());
        let label = &tail[..label_end];
        if !label.bytes().all(|b| b.is_ascii_uppercase() || b == b' ')
            || label.starts_with(' ')
            || label.contains("  ")
        {
            return Err(());
        }
        if label_end == tail.len() {
            return Ok(None);
        }
        if !(label == "PRIVATE KEY" || label.ends_with(" PRIVATE KEY")) {
            return Err(());
        }
        let dashes = &tail[label_end..];
        if !"-----".starts_with(dashes) {
            return Err(());
        }
        Ok((dashes.len() == 5).then(|| format!("-----END {label}-----")))
    }
}

/// An open PEM block; everything is masked until `end` is seen.
struct Pem {
    start: usize,
    end: String,
    /// The last `end.len()` characters.
    tail: VecDeque<char>,
}

/// Masks one logical UTF-8 string written in pieces.
///
/// Piece boundaries never end a match. `block_boundary` ends token matches
/// but keeps an open PEM block. `finish` ends the string. After any error the
/// stream is closed and later calls return `BrokenPipe` without writing.
///
/// Each character and each complete marker reaches the sink in one
/// `write_all` call, so a sink that fits output to a display size can refuse
/// a whole marker rather than cut it.
pub struct Masker {
    patterns: Vec<Pattern>,
    candidates: Vec<Candidate>,
    pending: VecDeque<Cell>,
    urls: Vec<Url>,
    headers: Vec<PemHeader>,
    pem: Option<Pem>,
    position: usize,
    previous: Option<char>,
    /// The region whose marker was written last; overlapping regions extend it.
    last_marker: Option<Region>,
    closed: bool,
}

impl Default for Masker {
    fn default() -> Self {
        Self::new()
    }
}

impl Masker {
    pub fn new() -> Self {
        Self {
            patterns: patterns(),
            candidates: Vec::new(),
            pending: VecDeque::new(),
            urls: Vec::new(),
            headers: Vec::new(),
            pem: None,
            position: 0,
            previous: None,
            last_marker: None,
            closed: false,
        }
    }

    pub fn push(&mut self, text: &str, sink: &mut impl Write) -> io::Result<()> {
        self.guarded(sink, |m, sink| {
            text.chars().try_for_each(|c| m.step(c, sink))
        })
    }

    pub fn block_boundary(&mut self, sink: &mut impl Write) -> io::Result<()> {
        self.guarded(sink, Self::boundary)
    }

    pub fn finish(&mut self, sink: &mut impl Write) -> io::Result<()> {
        self.guarded(sink, |m, sink| {
            m.closed = true;
            m.boundary(sink)
        })
    }

    /// Refuses calls on a closed stream and closes it on error.
    fn guarded<W: Write>(
        &mut self,
        sink: &mut W,
        op: impl FnOnce(&mut Self, &mut W) -> io::Result<()>,
    ) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "masking stream is closed",
            ));
        }
        let result = op(self, sink);
        if result.is_err() {
            self.closed = true;
        }
        result
    }

    fn boundary(&mut self, sink: &mut impl Write) -> io::Result<()> {
        for url in std::mem::take(&mut self.urls) {
            if let Some((a, b)) = url.region() {
                self.cover(a, b, "URL credentials");
            }
        }
        for candidate in &mut self.candidates {
            candidate.alive = false;
        }
        self.commit_candidates();
        self.candidates.clear();
        self.headers.clear();
        self.previous = None;
        self.drain(usize::MAX, sink)?;
        if self.pem.is_none() {
            self.last_marker = None;
        }
        Ok(())
    }

    /// Marks `a..b` for masking, merged with every overlapping region,
    /// including one whose marker was already written. Adjacent regions stay
    /// separate.
    fn cover(&mut self, mut a: usize, mut b: usize, kind: &'static str) {
        loop {
            let old = (a, b);
            for r in self
                .pending
                .iter()
                .filter_map(|c| c.region)
                .chain(self.last_marker)
            {
                if a < r.end && r.start < b {
                    a = a.min(r.start);
                    b = b.max(r.end);
                }
            }
            if old == (a, b) {
                break;
            }
        }
        if let Some(r) = &mut self.last_marker {
            if a < r.end && r.start < b {
                r.start = a;
                r.end = b;
            }
        }
        for cell in &mut self.pending {
            if cell.pos >= a && cell.pos < b {
                cell.region = Some(Region {
                    start: a,
                    end: b,
                    kind,
                });
            }
        }
    }

    /// Records the masked region of finished matches.
    ///
    /// Within a family the earliest match wins, as with Python `re.finditer`,
    /// so a later match waits while an earlier one can still complete.
    fn commit_candidates(&mut self) {
        for family in 0..FAMILIES {
            loop {
                let earliest = self
                    .candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| {
                        self.patterns[c.pattern].family == family
                            && (c.alive || c.accepted.is_some())
                    })
                    .min_by_key(|(_, c)| c.start)
                    .map(|(i, _)| i);
                let Some(i) = earliest else {
                    break;
                };
                let c = &self.candidates[i];
                let Some(end) = c.accepted else {
                    break;
                };
                let (start, secret, kind, alive) =
                    (c.start, c.secret, self.patterns[c.pattern].kind, c.alive);
                self.cover(secret, end, kind);
                self.candidates[i].committed = true;
                self.candidates.retain(|c| {
                    self.patterns[c.pattern].family != family
                        || c.start >= end
                        || c.start == start && c.committed && c.alive
                });
                if alive {
                    break;
                }
            }
        }
        self.candidates.retain(|c| c.alive || c.accepted.is_some());
    }

    fn step(&mut self, c: char, sink: &mut impl Write) -> io::Result<()> {
        let held = self
            .pending
            .front()
            .map_or(0, |first| self.position - first.pos);
        if held + c.len_utf8() > CARRY
            || self.candidates.len() >= MAX_CANDIDATES
            || self.urls.len() >= CARRY
            || self.headers.len() >= CARRY
            || self
                .headers
                .iter()
                .any(|h| h.text.len() + c.len_utf8() > CARRY)
        {
            return Err(limit_reached());
        }
        let pos = self.position;
        self.position += c.len_utf8();
        self.pending.push_back(Cell {
            pos,
            c,
            region: None,
        });
        self.step_pem(c, pos);
        self.step_urls(c, pos);
        for candidate in &mut self.candidates {
            if candidate.alive {
                candidate.step(c, pos, &self.patterns[candidate.pattern]);
            }
        }
        self.start_candidates(c, pos)?;
        self.commit_candidates();
        let safe = self
            .candidates
            .iter()
            .filter(|c| !c.committed)
            .map(|c| c.start)
            .chain(self.urls.iter().filter_map(|u| u.authority))
            .chain(self.headers.iter().map(|h| h.start))
            .min()
            .unwrap_or(self.position);
        self.drain(safe, sink)?;
        self.previous = Some(c);
        Ok(())
    }

    fn step_pem(&mut self, c: char, pos: usize) {
        if let Some(pem) = &mut self.pem {
            pem.tail.push_back(c);
            if pem.tail.len() > pem.end.len() {
                pem.tail.pop_front();
            }
            let done = pem.tail.iter().copied().eq(pem.end.chars());
            let start = pem.start;
            self.cover(start, self.position, "PEM private key");
            if done {
                self.pem = None;
            }
            return;
        }
        if c == '-' {
            self.headers.push(PemHeader {
                start: pos,
                text: String::new(),
            });
        }
        let mut complete = None;
        self.headers.retain_mut(|h| match h.step(c) {
            Ok(Some(end)) => {
                complete = Some((h.start, end));
                false
            }
            Ok(None) => true,
            Err(()) => false,
        });
        if let Some((start, end)) = complete {
            self.cover(start, self.position, "PEM private key");
            self.pem = Some(Pem {
                start,
                end,
                tail: VecDeque::new(),
            });
            self.headers.clear();
        }
    }

    fn step_urls(&mut self, c: char, pos: usize) {
        let mut ended = Vec::new();
        self.urls.retain_mut(|u| {
            u.step(c, pos) || {
                ended.extend(u.region());
                false
            }
        });
        for (a, b) in ended {
            self.cover(a, b, "URL credentials");
        }
        if c.is_ascii_alphabetic()
            && self
                .previous
                .is_none_or(|p| !is_word_char(p) && !"+.-".contains(p))
        {
            self.urls.push(Url::default());
        }
    }

    fn start_candidates(&mut self, c: char, pos: usize) -> io::Result<()> {
        for (i, p) in self.patterns.iter().enumerate() {
            let at_boundary = match p.boundary {
                Boundary::Word => self.previous.is_none_or(|x| !is_word_char(x)),
                Boundary::Space => self.previous.is_none_or(whitespace),
            };
            // A committed match of the same family that is still growing here
            // blocks a new one from starting inside it.
            let inside_match = self.candidates.iter().any(|s| {
                s.committed
                    && s.alive
                    && self.patterns[s.pattern].family == p.family
                    && s.accepted == Some(self.position)
            });
            if !at_boundary || inside_match || !p.parts[0].class.accepts(c) {
                continue;
            }
            if self.candidates.len() == MAX_CANDIDATES {
                return Err(limit_reached());
            }
            let mut candidate = Candidate {
                pattern: i,
                start: pos,
                secret: pos,
                part: 0,
                count: 0,
                accepted: None,
                alive: true,
                committed: false,
            };
            candidate.step(c, pos, p);
            self.candidates.push(candidate);
        }
        Ok(())
    }

    /// Writes held characters before `safe` and every masked cell at the front.
    fn drain(&mut self, safe: usize, sink: &mut impl Write) -> io::Result<()> {
        while let Some(cell) = self
            .pending
            .pop_front_if(|c| c.region.is_some() || c.pos < safe)
        {
            let Some(r) = cell.region else {
                sink.write_all(cell.c.encode_utf8(&mut [0; 4]).as_bytes())?;
                continue;
            };
            match &mut self.last_marker {
                Some(last) if r.start < last.end && last.start < r.end => {
                    last.end = last.end.max(r.end);
                }
                _ => {
                    sink.write_all(format!("[credential: {}, masked]", r.kind).as_bytes())?;
                    self.last_marker = Some(r);
                }
            }
        }
        Ok(())
    }
}
