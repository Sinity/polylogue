//! Bounded streaming implementation of the fixed `credential-output-v1` catalog.
//! Raw/canonical values, detection flags and reveal authorization are separate.
//! Undecided text is limited to 512 UTF-8 bytes; exceeding that limit returns
//! InvalidData before releasing the region. Recognized runs and PEM bodies do
//! not accumulate. A sink error is returned unchanged and fuses the stream.
use std::collections::VecDeque;
use std::io::{self, Write};

const CARRY: usize = 512;
const MAX_CANDIDATES: usize = 2048;
include!("masking_unicode.rs");

fn word(c: char) -> bool {
    c == '_'
        || WORD_RANGES
            .binary_search_by(|&(a, b)| {
                if c < a {
                    std::cmp::Ordering::Greater
                } else if c > b {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .is_ok()
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
#[derive(Clone, Copy)]
enum Class {
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
#[derive(Clone, Copy)]
enum Boundary {
    Word,
    Space,
}
struct Pattern {
    family: usize,
    kind: &'static str,
    boundary: Boundary,
    parts: Vec<Part>,
    secret: usize,
}
fn literal(parts: &mut Vec<Part>, text: &str, insensitive: bool) {
    parts.extend(
        text.chars()
            .map(|c| fixed(Class::Literal(c, insensitive), 1)),
    );
}
fn add(
    patterns: &mut Vec<Pattern>,
    family: usize,
    kind: &'static str,
    prefix: &str,
    insensitive: bool,
    tail: Class,
    min: usize,
) {
    let mut parts = Vec::new();
    literal(&mut parts, prefix, insensitive);
    parts.push(run(tail, min));
    patterns.push(Pattern {
        family,
        kind,
        boundary: Boundary::Word,
        parts,
        secret: 0,
    });
}
// Family slots follow the pinned catalog order; URL and PEM use slots 19/20.
fn patterns() -> Vec<Pattern> {
    use Class::*;
    let mut p = Vec::new();
    for prefix in ["AKIA", "ASIA", "ABIA", "ACCA"] {
        add(&mut p, 0, "AWS access key", prefix, true, Alnum, 16);
    }
    for prefix in ["ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"] {
        add(&mut p, 2, "GitHub token", prefix, false, Under, 36);
    }
    add(&mut p, 3, "GitLab token", "glpat-", false, Url, 20);
    add(&mut p, 4, "npm token", "npm_", false, Alnum, 36);
    for prefix in [
        "sk_test_", "sk_live_", "pk_test_", "pk_live_", "rk_test_", "rk_live_",
    ] {
        add(&mut p, 5, "Stripe key", prefix, false, Alnum, 24);
    }
    for prefix in ["xoxb-", "xoxp-", "xoxs-", "xoxa-", "xoxr-"] {
        add(&mut p, 6, "Slack token", prefix, false, Dash, 1);
    }
    add(&mut p, 8, "Google API key", "AIza", false, Url, 35);
    for prefix in ["sk-ant-api", "sk-ant-oat", "sk-ant-sid"] {
        let mut parts = Vec::new();
        literal(&mut parts, prefix, false);
        parts.push(fixed(Digit, 2));
        literal(&mut parts, "-", false);
        parts.push(run(Url, 20));
        p.push(Pattern {
            family: 13,
            kind: "Anthropic key",
            boundary: Boundary::Word,
            parts,
            secret: 0,
        });
    }
    for prefix in ["sk-proj-", "sk-svcacct-", "sk-admin-"] {
        add(&mut p, 14, "OpenAI key", prefix, false, Url, 20);
    }
    add(&mut p, 14, "OpenAI key", "sk-", false, Alnum, 48);
    add(&mut p, 15, "Hugging Face token", "hf_", false, Alnum, 34);
    add(&mut p, 18, "Twilio key", "SK", false, LowerHex, 32);
    let mut parts = vec![fixed(Hex, 32)];
    literal(&mut parts, "-us", true);
    parts.push(run(Digit, 1));
    p.push(Pattern {
        family: 16,
        kind: "Mailchimp key",
        boundary: Boundary::Word,
        parts,
        secret: 0,
    });
    for (family, kind, prefix, first_min, second_prefix, second_min) in [
        (7, "JWT", "eyJ", 1, ".eyJ", 1),
        (17, "SendGrid token", "SG.", 20, ".", 20),
    ] {
        let mut parts = Vec::new();
        literal(&mut parts, prefix, false);
        parts.push(run(Url, first_min));
        literal(&mut parts, second_prefix, false);
        parts.push(run(Url, second_min));
        if family == 7 {
            literal(&mut parts, ".", false);
            parts.push(run(Url, 1));
        }
        p.push(Pattern {
            family,
            kind,
            boundary: Boundary::Word,
            parts,
            secret: 0,
        });
    }
    for label in ["aws_secret_access_key", "secret_access_key", "aws_secret"] {
        for delimiter in [":", "="] {
            for quote in ["", "\"", "'"] {
                let mut parts = Vec::new();
                literal(&mut parts, label, true);
                parts.push(run(Space, 0));
                literal(&mut parts, delimiter, false);
                parts.push(run(Space, 0));
                literal(&mut parts, quote, false);
                let secret = parts.len();
                parts.push(run(BasePad, 40));
                p.push(Pattern {
                    family: 1,
                    kind: "AWS secret key",
                    boundary: Boundary::Word,
                    parts,
                    secret,
                });
            }
        }
    }
    let mut parts = Vec::new();
    literal(&mut parts, "AccountKey=", false);
    let secret = parts.len();
    parts.push(run(BasePad, 44));
    p.push(Pattern {
        family: 9,
        kind: "Azure connection key",
        boundary: Boundary::Word,
        parts,
        secret,
    });
    let mut parts = Vec::new();
    literal(&mut parts, "Authorization", true);
    parts.push(run(Space, 0));
    literal(&mut parts, ":", false);
    parts.push(run(Space, 0));
    literal(&mut parts, "Basic", true);
    parts.push(run(Space, 1));
    let secret = parts.len();
    parts.push(fixed(Base, 8));
    parts.push(run(BasePad, 0));
    p.push(Pattern {
        family: 11,
        kind: "Basic authorization",
        boundary: Boundary::Word,
        parts,
        secret,
    });
    let mut parts = Vec::new();
    literal(&mut parts, "Bearer", true);
    parts.push(run(Space, 1));
    let secret = parts.len();
    parts.push(run(Bearer, 20));
    p.push(Pattern {
        family: 12,
        kind: "Bearer token",
        boundary: Boundary::Word,
        parts,
        secret,
    });
    for (family, kind, flags) in [
        (
            10,
            "CLI secret",
            &[
                "--password",
                "--token",
                "--secret",
                "--api-key",
                "--auth-token",
            ][..],
        ),
        (21, "CLI credentials", &["-u", "--user"][..]),
    ] {
        for flag in flags {
            for equals in [false, true] {
                for quote in [None, Some('\''), Some('"')] {
                    let mut parts = Vec::new();
                    literal(&mut parts, flag, false);
                    if equals {
                        literal(&mut parts, "=", false);
                    } else {
                        parts.push(run(Space, 1));
                    }
                    let secret = parts.len();
                    if let Some(q) = quote {
                        literal(&mut parts, &q.to_string(), false);
                    }
                    let class = quote.map_or(if family == 21 { User } else { Bare }, |q| {
                        Quoted(q, family == 21)
                    });
                    parts.push(run(class, 1));
                    if family == 21 {
                        literal(&mut parts, ":", false);
                        parts.push(run(quote.map_or(Bare, |q| Quoted(q, false)), 1));
                    }
                    if let Some(q) = quote {
                        literal(&mut parts, &q.to_string(), false);
                    }
                    p.push(Pattern {
                        family,
                        kind,
                        boundary: Boundary::Space,
                        parts,
                        secret,
                    });
                }
            }
        }
    }
    p
}
struct Candidate {
    pattern: usize,
    start: usize,
    secret: usize,
    part: usize,
    count: usize,
    accepted: Option<usize>,
    alive: bool,
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
#[derive(Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
    kind: &'static str,
}
struct Cell {
    pos: usize,
    c: char,
    region: Option<Region>,
}
struct Url {
    authority: Option<usize>,
    stage: u8,
    colon: Option<usize>,
    at: Option<usize>,
}
impl Url {
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
struct PemHeader {
    start: usize,
    text: String,
}
impl PemHeader {
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
struct Pem {
    start: usize,
    end: String,
    tail: VecDeque<char>,
}

/// One logical UTF-8 display stream. Pieces do not terminate matches. A block
/// boundary terminates ordinary recognizers; matching PEM suppression survives.
/// Every error permanently closes the stream. Sink errors retain their identity.
pub struct Masker {
    patterns: Vec<Pattern>,
    candidates: Vec<Candidate>,
    pending: VecDeque<Cell>,
    urls: Vec<Url>,
    headers: Vec<PemHeader>,
    pem: Option<Pem>,
    position: usize,
    previous: Option<char>,
    rendered: Option<Region>,
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
            rendered: None,
            closed: false,
        }
    }
    fn check_open(&self) -> io::Result<()> {
        if self.closed {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "masking stream is closed",
            ))
        } else {
            Ok(())
        }
    }
    pub fn push(&mut self, text: &str, sink: &mut impl Write) -> io::Result<()> {
        self.check_open()?;
        let result = text.chars().try_for_each(|c| self.step(c, sink));
        if result.is_err() {
            self.closed = true;
        }
        result
    }
    pub fn finish(&mut self, sink: &mut impl Write) -> io::Result<()> {
        self.check_open()?;
        self.closed = true;
        self.boundary(sink)
    }
    pub fn block_boundary(&mut self, sink: &mut impl Write) -> io::Result<()> {
        self.check_open()?;
        let result = self.boundary(sink);
        if result.is_err() {
            self.closed = true;
        }
        result
    }
    fn boundary(&mut self, sink: &mut impl Write) -> io::Result<()> {
        let urls = std::mem::take(&mut self.urls);
        for url in urls {
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
            self.rendered = None;
        }
        Ok(())
    }
    fn cover(&mut self, mut a: usize, mut b: usize, kind: &'static str) {
        // Merge overlaps (not adjacency), including regions already streamed.
        loop {
            let old = (a, b);
            for r in self
                .pending
                .iter()
                .filter_map(|c| c.region)
                .chain(self.rendered)
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
        if let Some(r) = &mut self.rendered {
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
    fn commit_candidates(&mut self) {
        // Python finditer chooses the earliest match per family. A later match
        // must wait while an earlier incomplete candidate can still succeed.
        for family in 0..22 {
            loop {
                let selected = self
                    .candidates
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| {
                        self.patterns[c.pattern].family == family
                            && (c.alive || c.accepted.is_some())
                    })
                    .min_by_key(|(_, c)| c.start)
                    .map(|(i, _)| i);
                let Some(i) = selected else {
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
        if self
            .pending
            .front()
            .is_some_and(|first| self.position - first.pos + c.len_utf8() > CARRY)
            || self.candidates.len() >= MAX_CANDIDATES
            || self.urls.len() >= CARRY
            || self.headers.len() >= CARRY
            || self
                .headers
                .iter()
                .any(|h| h.text.len() + c.len_utf8() > CARRY)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "undecided credential state exceeds fixed bound",
            ));
        }
        let pos = self.position;
        self.position += c.len_utf8();
        self.pending.push_back(Cell {
            pos,
            c,
            region: None,
        });
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
        } else {
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
        let mut ended = Vec::new();
        self.urls.retain_mut(|u| {
            if u.step(c, pos) {
                true
            } else {
                if let Some(r) = u.region() {
                    ended.push(r);
                }
                false
            }
        });
        for (a, b) in ended {
            self.cover(a, b, "URL credentials");
        }
        if c.is_ascii_alphabetic() && self.previous.is_none_or(|p| !word(p) && !"+.-".contains(p)) {
            self.urls.push(Url {
                authority: None,
                stage: 0,
                colon: None,
                at: None,
            });
        }
        for candidate in &mut self.candidates {
            if candidate.alive {
                candidate.step(c, pos, &self.patterns[candidate.pattern]);
            }
        }
        for (i, p) in self.patterns.iter().enumerate() {
            let boundary = match p.boundary {
                Boundary::Word => self.previous.is_none_or(|x| !word(x)),
                Boundary::Space => self.previous.is_none_or(whitespace),
            };
            let occupied = self.candidates.iter().any(|s| {
                s.committed
                    && s.alive
                    && self.patterns[s.pattern].family == p.family
                    && s.accepted == Some(self.position)
            });
            if boundary && !occupied && p.parts[0].class.accepts(c) {
                let mut state = Candidate {
                    pattern: i,
                    start: pos,
                    secret: pos,
                    part: 0,
                    count: 0,
                    accepted: None,
                    alive: true,
                    committed: false,
                };
                if self.candidates.len() == MAX_CANDIDATES {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "undecided credential candidates exceed fixed bound",
                    ));
                }
                state.step(c, pos, p);
                self.candidates.push(state);
            }
        }
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
        if self
            .pending
            .front()
            .is_some_and(|first| self.position - first.pos > CARRY)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "undecided credential region exceeds 512 bytes",
            ));
        }
        Ok(())
    }
    fn drain(&mut self, safe: usize, sink: &mut impl Write) -> io::Result<()> {
        while self
            .pending
            .front()
            .is_some_and(|c| c.region.is_some() || c.pos < safe)
        {
            let cell = self.pending.pop_front().unwrap();
            if let Some(r) = cell.region {
                if !self
                    .rendered
                    .is_some_and(|last| r.start < last.end && last.start < r.end)
                {
                    write!(sink, "[credential: {}, masked]", r.kind)?;
                    self.rendered = Some(r);
                } else if let Some(last) = &mut self.rendered {
                    last.end = last.end.max(r.end);
                }
            } else {
                sink.write_all(cell.c.encode_utf8(&mut [0; 4]).as_bytes())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn masked(parts: &[&str]) -> String {
        let mut masker = Masker::new();
        let mut out = Vec::new();
        for text in parts {
            masker.push(text, &mut out).unwrap();
        }
        masker.finish(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }
    fn normalized(text: &str) -> String {
        let mut rest = text;
        let mut out = String::new();
        while let Some(start) = rest.find("[credential: ") {
            out.push_str(&rest[..start]);
            out.push_str("<masked>");
            let end = rest[start..].find(", masked]").unwrap() + start + ", masked]".len();
            rest = &rest[end..];
        }
        out.push_str(rest);
        out
    }
    // Model an actual UTF-8 transport decoder: only complete scalar values are
    // passed to the &str API, keeping at most three undecoded bytes between frames.
    fn byte_chunks(input: &str, chunks: &[usize]) -> String {
        let mut masker = Masker::new();
        let mut out = Vec::new();
        let mut offset = 0;
        let mut buffered = Vec::new();
        for &size in chunks {
            let end = (offset + size).min(input.len());
            buffered.extend_from_slice(&input.as_bytes()[offset..end]);
            offset = end;
            let valid = match std::str::from_utf8(&buffered) {
                Ok(s) => s.len(),
                Err(e) => {
                    assert!(e.error_len().is_none());
                    e.valid_up_to()
                }
            };
            masker
                .push(std::str::from_utf8(&buffered[..valid]).unwrap(), &mut out)
                .unwrap();
            buffered.drain(..valid);
            assert!(buffered.len() <= 3);
        }
        assert_eq!(offset, input.len());
        assert!(buffered.is_empty());
        masker.finish(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }
    #[test]
    fn fixed_catalog_independent_extents_at_every_byte_cut_and_small_chunk_size() {
        let corpus: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/credential-output-v1/conformance.json"
        ))
        .unwrap();
        assert_eq!(corpus["catalog_version"], "credential-output-v1");
        assert_eq!(corpus["fixture_encoding"], "text-parts-v1");
        let cases = corpus["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 132);
        for case in cases {
            let text: String = case["text_parts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|part| part.as_str().unwrap())
                .collect();
            let text = text.as_str();
            let id = case["id"].as_str().unwrap();
            let mut expected = String::new();
            let mut offset = 0;
            for region in case["regions_utf8"].as_array().unwrap() {
                let a = region[0].as_u64().unwrap() as usize;
                let b = region[1].as_u64().unwrap() as usize;
                expected.push_str(&text[offset..a]);
                expected.push_str("<masked>");
                offset = b;
            }
            expected.push_str(&text[offset..]);
            for cut in 0..=text.len() {
                assert_eq!(
                    normalized(&byte_chunks(text, &[cut, text.len() - cut])),
                    expected,
                    "{id}, byte cut {cut}"
                );
            }
            for width in 1..=8 {
                let chunks: Vec<_> = (0..text.len())
                    .step_by(width)
                    .map(|i| width.min(text.len() - i))
                    .collect();
                assert_eq!(
                    normalized(&byte_chunks(text, &chunks)),
                    expected,
                    "{id}, chunk width {width}"
                );
            }
        }
    }
    #[test]
    fn unicode_boundary_matches_python_letters_numbers_underscore_and_whitespace() {
        let token = format!("ghp_{}", "A".repeat(36));
        for prefix in ['\u{345}', '\u{301}', '\u{203f}', '\u{200d}'] {
            assert_eq!(
                normalized(&masked(&[&format!("{prefix}{token}")])),
                format!("{prefix}<masked>")
            );
        }
        for prefix in ['π', '雪', 'Ⅷ', '²', '_', '9'] {
            let text = format!("{prefix}{token}");
            assert_eq!(masked(&[&text]), text);
        }
        for prefix in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{85}', '\u{a0}'] {
            assert_eq!(
                normalized(&masked(&[&format!("{prefix}--secret=x")])),
                format!("{prefix}--secret=<masked>")
            );
        }
    }
    #[test]
    fn contextual_labels_short_secrets_and_unclosed_quotes_preserve_exact_text() {
        for (input, expected) in [
            (
                "run --secret=x --user iris --key /home/iris/id.pub",
                "run --secret=<masked> --user iris --key /home/iris/id.pub",
            ),
            ("curl --user='a:b'", "curl --user=<masked>"),
            ("--secret='' --token=\"\"", "--secret='' --token=\"\""),
            (
                "--secret=\"outer --secret=inner",
                "--secret=\"outer --secret=<masked>",
            ),
            (
                "token=ast_node secret=chapter user@example.invalid",
                "token=ast_node secret=chapter user@example.invalid",
            ),
        ] {
            assert_eq!(normalized(&masked(&[input])), expected);
        }
    }
    #[test]
    fn overlap_unions_and_adjacency_are_distinct() {
        let input = format!("xoxb-A-ghp_{}!tail", "A".repeat(36));
        assert_eq!(normalized(&masked(&[&input])), "<masked>!tail");
        let input = format!(
            "-----BEGIN PRIVATE KEY-----x-----END PRIVATE KEY-----ghp_{}",
            "A".repeat(36)
        );
        assert_eq!(normalized(&masked(&[&input])), "<masked><masked>");
        let input = format!("https://u:ghp_{}!private@host/path", "A".repeat(36));
        assert_eq!(normalized(&masked(&[&input])), "https://<masked>@host/path");
        let crossing =
            "-----BEGIN PRIVATE KEY----- --secret=\"inside -----END PRIVATE KEY----- after\" tail";
        // Independently authored PEM [0,70), CLI secret [37,77), union [0,77).
        assert_eq!(&crossing[77..], " tail");
        for cut in 0..=crossing.len() {
            assert_eq!(
                normalized(&masked(&[&crossing[..cut], &crossing[cut..]])),
                "<masked> tail",
                "PEM/context overlap cut {cut}"
            );
        }
        let unfinished = format!("https://u:ghp_{}", "A".repeat(36));
        assert_eq!(normalized(&masked(&[&unfinished])), "https://u:<masked>");
    }
    #[test]
    fn block_boundaries_terminate_prefixes_and_keep_dynamic_pem_body_and_end_progress() {
        let mut m = Masker::new();
        let mut out = Vec::new();
        m.push("short ghp_", &mut out).unwrap();
        m.block_boundary(&mut out).unwrap();
        m.push(
            "-----BEGIN VENDOR ENCRYPTED PRIVATE KEY-----body-----END VEND",
            &mut out,
        )
        .unwrap();
        m.block_boundary(&mut out).unwrap();
        m.push("OR ENCRYPTED PRIVATE KEY----- visible", &mut out)
            .unwrap();
        m.finish(&mut out).unwrap();
        assert_eq!(
            normalized(std::str::from_utf8(&out).unwrap()),
            "short ghp_<masked> visible"
        );
        for cut in 0..=40 {
            let raw = format!("ghp_{}", "A".repeat(36));
            let mut m = Masker::new();
            let mut out = Vec::new();
            m.push(&raw[..cut], &mut out).unwrap();
            m.block_boundary(&mut out).unwrap();
            m.push(&raw[cut..], &mut out).unwrap();
            m.finish(&mut out).unwrap();
            let expected = if cut == 0 || cut == 40 {
                "<masked>".to_owned()
            } else {
                raw.clone()
            };
            assert_eq!(
                normalized(std::str::from_utf8(&out).unwrap()),
                expected,
                "block cut {cut}"
            );
        }
    }
    #[test]
    fn known_runs_and_pem_bodies_stream_with_fixed_state() {
        for begin in [
            format!("ghp_{}", "A".repeat(36)),
            "Bearer AAAAAAAAAAAAAAAAAAAA".into(),
            "--secret=x".into(),
            "-----BEGIN VENDOR PRIVATE KEY-----".into(),
        ] {
            let mut m = Masker::new();
            let mut out = Vec::new();
            m.push(&begin, &mut out).unwrap();
            let initial = out.len();
            for _ in 0..256 {
                m.push(&"A".repeat(4096), &mut out).unwrap();
                assert!(m.pending.len() <= CARRY);
                assert!(m.candidates.len() <= MAX_CANDIDATES);
                assert!(m.headers.iter().all(|h| h.text.len() <= CARRY));
            }
            m.finish(&mut out).unwrap();
            assert_eq!(out.len(), initial, "{begin}");
        }
        // Schemes themselves are preserved, so an arbitrarily long ordinary
        // ASCII word needs no undecided text carry.
        let ordinary = "Z".repeat(4096);
        assert_eq!(masked(&[&ordinary]), ordinary);
    }
    #[test]
    fn undecided_regions_refuse_without_releasing_them_and_fuse() {
        for (prefix, visible) in [
            ("before https://u:", "before https://"),
            ("before eyJA", "before "),
            ("before SG.A", "before "),
            ("before --secret=\"x", "before "),
            ("before -----BEGIN VENDOR ", "before "),
        ] {
            let mut m = Masker::new();
            let mut out = Vec::new();
            m.push(prefix, &mut out).unwrap();
            let error = m.push(&"A".repeat(CARRY), &mut out).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert_eq!(std::str::from_utf8(&out).unwrap(), visible, "{prefix}");
            let calls = out.len();
            assert!(m.push("later", &mut out).is_err());
            assert!(m.block_boundary(&mut out).is_err());
            assert!(m.finish(&mut out).is_err());
            assert_eq!(out.len(), calls);
        }
    }
    #[test]
    fn adversarial_incomplete_candidates_inside_pem_have_a_fixed_refusal_bound() {
        let mut m = Masker::new();
        let mut out = Vec::new();
        m.push("-----BEGIN PRIVATE KEY-----", &mut out).unwrap();
        let error = m
            .push(&"eyJA-".repeat(MAX_CANDIDATES + 1), &mut out)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(m.candidates.len() <= MAX_CANDIDATES + 1);
        assert_eq!(normalized(std::str::from_utf8(&out).unwrap()), "<masked>");
    }
    #[test]
    fn original_sink_errors_fuse_push_finish_and_boundary_without_another_write() {
        #[derive(Debug)]
        struct Original;
        impl std::fmt::Display for Original {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("original")
            }
        }
        impl std::error::Error for Original {}
        struct Broken {
            calls: usize,
        }
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                Err(io::Error::other(Original))
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("no flush")
            }
        }
        for operation in [0, 1, 2, 3] {
            let mut m = Masker::new();
            let mut sink = Broken { calls: 0 };
            let error = match operation {
                0 => m.push("visible ", &mut sink).unwrap_err(),
                1 => {
                    m.push("ghp_", &mut sink).unwrap();
                    m.finish(&mut sink).unwrap_err()
                }
                2 => {
                    m.push("ghp_", &mut sink).unwrap();
                    m.block_boundary(&mut sink).unwrap_err()
                }
                _ => m
                    .push(&format!("ghp_{}", "A".repeat(36)), &mut sink)
                    .unwrap_err(),
            };
            assert!(error.get_ref().unwrap().is::<Original>());
            assert!(m.push("later", &mut sink).is_err());
            assert!(m.finish(&mut sink).is_err());
            assert!(m.block_boundary(&mut sink).is_err());
            assert_eq!(sink.calls, 1);
        }
    }
}
