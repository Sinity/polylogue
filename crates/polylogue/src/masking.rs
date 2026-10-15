//! Streaming display masking, independent of retained raw/canonical content.
//!
//! This deliberately selects the high-precision prefix shapes from Sinex's
//! credential catalog (c997df89764aa47d624355ae04fbe65e6f9a2bfa), not its PII,
//! generic assignments, JWT/Bearer engine, or vendored Gitleaks rules. URL
//! credentials support explicit lowercase HTTP(S) schemes, not all schemes.
//! Unlike catalog maximum-length matches, a recognized token's entire same-
//! alphabet tail is suppressed, including arbitrarily long malformed tokens.
//! PEM recognizes the six explicit headers below and suppresses through their
//! matching END header (or EOF); arbitrary vendor header labels are unsupported.
//! Work is linear in input bytes with fixed rule/header factors. Carry is fixed
//! size; neither token bodies nor PEM bodies accumulate. Undecided URL
//! credentials refuse at 512 bytes without releasing that undecided region.
//! Detection/index status and reveal authorization are outside this module.
use std::io::{self, Write};

const CARRY: usize = 512;

#[derive(Clone, Copy)]
enum Alphabet {
    Alnum,
    Underscore,
    Url,
}
impl Alphabet {
    fn contains(self, b: u8) -> bool {
        b.is_ascii_alphanumeric()
            || match self {
                Self::Alnum => false,
                Self::Underscore => b == b'_',
                Self::Url => b == b'_' || b == b'-',
            }
    }
}
#[derive(Clone, Copy)]
struct Rule {
    prefix: &'static [u8],
    minimum: usize,
    alphabet: Alphabet,
    kind: &'static str,
    aws: bool,
    end: Option<&'static [u8]>,
}
const fn token(
    prefix: &'static [u8],
    minimum: usize,
    alphabet: Alphabet,
    kind: &'static str,
) -> Rule {
    Rule {
        prefix,
        minimum,
        alphabet,
        kind,
        aws: false,
        end: None,
    }
}
const fn aws(prefix: &'static [u8]) -> Rule {
    Rule {
        prefix,
        minimum: 16,
        alphabet: Alphabet::Alnum,
        kind: "AWS access key",
        aws: true,
        end: None,
    }
}
const fn pem(prefix: &'static [u8], end: &'static [u8]) -> Rule {
    Rule {
        prefix,
        minimum: 0,
        alphabet: Alphabet::Alnum,
        kind: "PEM private key",
        aws: false,
        end: Some(end),
    }
}
const RULES: &[Rule] = &[
    token(b"ghp_", 36, Alphabet::Underscore, "GitHub token"),
    token(b"gho_", 36, Alphabet::Underscore, "GitHub token"),
    token(b"ghu_", 36, Alphabet::Underscore, "GitHub token"),
    token(b"ghs_", 36, Alphabet::Underscore, "GitHub token"),
    token(b"ghr_", 36, Alphabet::Underscore, "GitHub token"),
    token(b"github_pat_", 36, Alphabet::Underscore, "GitHub token"),
    token(b"glpat-", 20, Alphabet::Url, "GitLab token"),
    token(b"npm_", 36, Alphabet::Alnum, "npm token"),
    token(b"hf_", 34, Alphabet::Alnum, "Hugging Face token"),
    token(b"AIza", 35, Alphabet::Url, "Google API key"),
    token(b"sk-ant-api??-", 20, Alphabet::Url, "Anthropic key"),
    token(b"sk-ant-oat??-", 20, Alphabet::Url, "Anthropic key"),
    token(b"sk-ant-sid??-", 20, Alphabet::Url, "Anthropic key"),
    token(b"sk-proj-", 20, Alphabet::Url, "OpenAI key"),
    token(b"sk-svcacct-", 20, Alphabet::Url, "OpenAI key"),
    token(b"sk-admin-", 20, Alphabet::Url, "OpenAI key"),
    token(b"sk-", 48, Alphabet::Alnum, "OpenAI key"),
    token(b"sk_live_", 24, Alphabet::Alnum, "Stripe key"),
    token(b"sk_test_", 24, Alphabet::Alnum, "Stripe key"),
    token(b"pk_live_", 24, Alphabet::Alnum, "Stripe key"),
    token(b"pk_test_", 24, Alphabet::Alnum, "Stripe key"),
    token(b"rk_live_", 24, Alphabet::Alnum, "Stripe key"),
    token(b"rk_test_", 24, Alphabet::Alnum, "Stripe key"),
    aws(b"AKIA"),
    aws(b"ASIA"),
    aws(b"ABIA"),
    aws(b"ACCA"),
    pem(b"-----BEGIN PRIVATE KEY-----", b"-----END PRIVATE KEY-----"),
    pem(
        b"-----BEGIN RSA PRIVATE KEY-----",
        b"-----END RSA PRIVATE KEY-----",
    ),
    pem(
        b"-----BEGIN EC PRIVATE KEY-----",
        b"-----END EC PRIVATE KEY-----",
    ),
    pem(
        b"-----BEGIN DSA PRIVATE KEY-----",
        b"-----END DSA PRIVATE KEY-----",
    ),
    pem(
        b"-----BEGIN OPENSSH PRIVATE KEY-----",
        b"-----END OPENSSH PRIVATE KEY-----",
    ),
    pem(
        b"-----BEGIN ENCRYPTED PRIVATE KEY-----",
        b"-----END ENCRYPTED PRIVATE KEY-----",
    ),
];

#[derive(Default)]
enum Mode {
    #[default]
    Text,
    Token(Alphabet),
    Pem {
        end: &'static [u8],
        tail: [u8; CARRY],
        len: usize,
    },
}
/// One logical display-text stream. Feed UTF-8 pieces in order, then finish once.
/// Output is UTF-8 collectively; `Write` calls can split a multibyte character.
/// A sink error is returned unchanged and permanently fuses this stream.
pub struct Masker {
    mode: Mode,
    pending: [u8; CARRY],
    len: usize,
    previous_word: bool,
    covered: usize,
    urls: bool,
    closed: bool,
}
impl Default for Masker {
    fn default() -> Self {
        Self {
            mode: Mode::Text,
            pending: [0; CARRY],
            len: 0,
            previous_word: false,
            covered: 0,
            urls: true,
            closed: false,
        }
    }
}
impl Masker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, text: &str, sink: &mut impl Write) -> io::Result<()> {
        self.check_open()?;
        let result = self.push_inner(text.as_bytes(), sink);
        if result.is_err() {
            self.closed = true;
        }
        result
    }
    pub fn finish(&mut self, sink: &mut impl Write) -> io::Result<()> {
        self.check_open()?;
        self.closed = true;
        self.flush_ordinary(sink)
    }
    /// A display block separator terminates token runs and short prefixes.
    /// PEM suppression survives, including a header inside an incomplete URL.
    pub fn block_boundary(&mut self, sink: &mut impl Write) -> io::Result<()> {
        self.check_open()?;
        let result = self.flush_ordinary(sink);
        if result.is_err() {
            self.closed = true;
        }
        result
    }
    fn flush_ordinary(&mut self, sink: &mut impl Write) -> io::Result<()> {
        if matches!(self.mode, Mode::Pem { .. }) {
            return Ok(());
        }
        // An unfinished URL is not itself a credential match. Revisit its
        // bounded undecided bytes through the SAME matcher with URL prefixes
        // disabled so nested complete tokens cannot escape at EOF/separators.
        let pending = self.pending;
        let start = self.covered;
        let end = self.len;
        if start > 0 {
            self.previous_word = word(pending[start - 1]);
        }
        self.len = 0;
        self.covered = 0;
        self.mode = Mode::Text;
        self.urls = false;
        let result = self.push_inner(&pending[start..end], sink);
        self.urls = true;
        result?;
        if !matches!(self.mode, Mode::Pem { .. }) {
            sink.write_all(&self.pending[self.covered..self.len])?;
            self.len = 0;
            self.covered = 0;
            self.mode = Mode::Text;
            self.previous_word = false;
        }
        Ok(())
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
    fn push_inner(&mut self, bytes: &[u8], sink: &mut impl Write) -> io::Result<()> {
        for &byte in bytes {
            let suppressed = match &mut self.mode {
                Mode::Token(alphabet) if alphabet.contains(byte) => true,
                Mode::Pem { end, tail, len } => {
                    if *len == end.len() {
                        tail.copy_within(1..*len, 0);
                        *len -= 1;
                    }
                    tail[*len] = byte;
                    *len += 1;
                    let done = &tail[..*len] == *end;
                    self.previous_word = word(byte);
                    if done {
                        self.mode = Mode::Text;
                    }
                    continue;
                }
                _ => {
                    self.mode = Mode::Text;
                    false
                }
            };
            // Every possible rule completes within CARRY. Nonmatches emit the
            // first byte before accepting another; there is no growing carry.
            if self.len == CARRY {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "undecided credential region exceeds 512 bytes",
                ));
            }
            self.pending[self.len] = byte;
            self.len += 1;
            if suppressed {
                self.covered = self.len;
            }
            loop {
                let url = if self.urls {
                    url_status(&self.pending[..self.len])
                } else {
                    UrlStatus::No
                };
                let mut possible = url != UrlStatus::No;
                if url == UrlStatus::Matched {
                    if self.covered == 0 {
                        marker(sink, "URL credentials", &self.pending[..4])?;
                    }
                    self.previous_word = false;
                    self.len = 0;
                    self.covered = 0;
                    self.mode = Mode::Text;
                    break;
                }
                let mut selected = None;
                for rule in RULES {
                    if rule.aws && self.previous_word {
                        continue;
                    }
                    if compatible(*rule, &self.pending[..self.len]) {
                        possible = true;
                        if self.len == rule.prefix.len() + rule.minimum {
                            selected = Some(*rule);
                            break;
                        }
                    }
                }
                if let Some(rule) = selected {
                    if self.covered == 0 {
                        marker(sink, rule.kind, &self.pending[..4])?;
                    }
                    self.previous_word = word(self.pending[self.len - 1]);
                    self.len = 0;
                    self.covered = 0;
                    self.mode = match rule.end {
                        Some(end) => Mode::Pem {
                            end,
                            tail: [0; CARRY],
                            len: 0,
                        },
                        None => Mode::Token(rule.alphabet),
                    };
                    break;
                }
                if possible {
                    break;
                }
                if self.covered == 0 {
                    sink.write_all(&self.pending[..1])?;
                } else {
                    self.covered -= 1;
                }
                self.previous_word = word(self.pending[0]);
                self.pending.copy_within(1..self.len, 0);
                self.len -= 1;
                if self.len == 0 {
                    break;
                }
            }
        }
        Ok(())
    }
}
fn marker(sink: &mut impl Write, kind: &str, first: &[u8]) -> io::Result<()> {
    sink.write_all(b"[credential: ")?;
    sink.write_all(kind.as_bytes())?;
    sink.write_all(b" ")?;
    sink.write_all(first)?;
    sink.write_all("…, masked]".as_bytes())
}
#[derive(PartialEq, Eq)]
enum UrlStatus {
    No,
    Possible,
    Matched,
}
// Explicit HTTP(S) subset. A slash before ':' identifies an ordinary URL path;
// whitespace terminates this free-text credential shape. Ambiguous long URLs
// refuse rather than release their undecided bytes.
fn url_status(bytes: &[u8]) -> UrlStatus {
    let Some(prefix) = [b"https://".as_slice(), b"http://".as_slice()]
        .into_iter()
        .find(|prefix| prefix.starts_with(bytes) || bytes.starts_with(prefix))
    else {
        return UrlStatus::No;
    };
    if bytes.len() <= prefix.len() {
        return UrlStatus::Possible;
    }
    let mut colon = false;
    let mut user = false;
    let mut password = false;
    for &b in &bytes[prefix.len()..] {
        if b.is_ascii_whitespace() {
            return UrlStatus::No;
        }
        if !colon {
            match b {
                b':' if user => colon = true,
                b':' | b'@' | b'/' => return UrlStatus::No,
                _ => user = true,
            }
        } else if b == b'@' {
            return if password {
                UrlStatus::Matched
            } else {
                UrlStatus::No
            };
        } else {
            password = true;
        }
    }
    UrlStatus::Possible
}
fn word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte >= 128
}
fn compatible(rule: Rule, bytes: &[u8]) -> bool {
    for (i, &byte) in bytes.iter().enumerate() {
        if i < rule.prefix.len() {
            let expected = rule.prefix[i];
            if !(expected == b'?' && byte.is_ascii_digit()
                || if rule.aws {
                    byte.eq_ignore_ascii_case(&expected)
                } else {
                    byte == expected
                })
            {
                return false;
            }
        } else if !rule.alphabet.contains(byte) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    fn masked(parts: &[&str]) -> String {
        let mut out = Vec::new();
        let mut masker = Masker::new();
        for part in parts {
            masker.push(part, &mut out).unwrap();
        }
        masker.finish(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    }
    #[test]
    fn credentials_mask_at_every_piece_cut_and_preserve_surrounding_unicode() {
        for (raw, kind, first) in [
            (format!("ghp_{}", "A".repeat(36)), "GitHub token", "ghp_"),
            (format!("AKIA{}", "A".repeat(16)), "AWS access key", "AKIA"),
            (
                format!("sk-ant-api03-{}", "A".repeat(20)),
                "Anthropic key",
                "sk-a",
            ),
            (format!("sk-proj-{}", "A".repeat(20)), "OpenAI key", "sk-p"),
            (format!("sk-{}", "A".repeat(48)), "OpenAI key", "sk-A"),
        ] {
            let input = format!("żółw {raw} 雪\n");
            let expected = format!("żółw [credential: {kind} {first}…, masked] 雪\n");
            for cut in input
                .char_indices()
                .map(|(i, _)| i)
                .chain(std::iter::once(input.len()))
            {
                assert_eq!(
                    masked(&[&input[..cut], &input[cut..]]),
                    expected,
                    "cut {cut}"
                );
            }
            let prefix = "x".repeat(4094);
            assert_eq!(masked(&[&prefix, &input]), format!("{prefix}{expected}"));
        }
    }
    #[test]
    fn nonsecrets_and_short_prefixes_remain_exact_and_no_pii_is_masked() {
        let raw = "żółw user@example.invalid /home/example token=ordinary ghp_short sk-ant-api03-short AKIA123 -----BEGIN PUBLIC KEY-----";
        assert_eq!(masked(&[raw]), raw);
        assert_eq!(
            masked(&["gh", "p_", &"A".repeat(35)]),
            format!("ghp_{}", "A".repeat(35))
        );
        assert_eq!(
            masked(&[&format!("xAKIA{}", "A".repeat(16))]),
            format!("xAKIA{}", "A".repeat(16))
        );
    }
    #[test]
    fn pem_body_and_end_marker_survive_all_cuts_without_body_carry() {
        let input =
            "before -----BEGIN RSA PRIVATE KEY-----\nQUJD\n-----END RSA PRIVATE KEY----- after";
        let expected = "before [credential: PEM private key ----…, masked] after";
        for cut in 0..=input.len() {
            assert_eq!(masked(&[&input[..cut], &input[cut..]]), expected);
        }
        assert_eq!(
            masked(&["-----BEGIN PRIVATE KEY-----", "truncated"]),
            "[credential: PEM private key ----…, masked]"
        );
    }
    #[test]
    fn arbitrary_token_and_pem_lengths_have_fixed_carry_and_no_tail_leak() {
        for begin in [
            format!("ghp_{}", "A".repeat(36)),
            "-----BEGIN EC PRIVATE KEY-----".into(),
        ] {
            let mut masker = Masker::new();
            let mut output = Vec::new();
            masker.push(&begin, &mut output).unwrap();
            let initial = output.len();
            for _ in 0..64 {
                masker.push(&"A".repeat(4096), &mut output).unwrap();
                assert!(masker.len <= CARRY);
            }
            masker.finish(&mut output).unwrap();
            assert_eq!(output.len(), initial);
        }
    }
    #[test]
    fn overlapping_url_and_token_spans_form_one_mask_at_every_cut() {
        let input = format!("before https://u:ghp_{}!private@host after", "A".repeat(36));
        let expected = "before [credential: URL credentials http…, masked]host after";
        for cut in 0..=input.len() {
            assert_eq!(masked(&[&input[..cut], &input[cut..]]), expected);
        }
        let unfinished = format!("https://u:ghp_{}", "A".repeat(36));
        assert_eq!(
            masked(&[&unfinished]),
            "https://u:[credential: GitHub token ghp_…, masked]"
        );
        // The nested OpenAI prefix begins inside GitHub's token alphabet but
        // extends past its end. Suppression must continue through both spans.
        let overlap = format!("ghp_{}sk-proj-{}!tail", "A".repeat(36), "B".repeat(20));
        for cut in 0..=overlap.len() {
            assert_eq!(
                masked(&[&overlap[..cut], &overlap[cut..]]),
                "[credential: GitHub token ghp_…, masked]!tail"
            );
        }
    }
    #[test]
    fn block_boundary_flushes_short_text_but_keeps_pem_body_suppressed() {
        let mut masker = Masker::new();
        let mut out = Vec::new();
        masker.push("short ghp_", &mut out).unwrap();
        masker.block_boundary(&mut out).unwrap();
        assert_eq!(out, b"short ghp_");
        masker
            .push("-----BEGIN EC PRIVATE KEY-----body", &mut out)
            .unwrap();
        masker.block_boundary(&mut out).unwrap();
        masker
            .push("continued-----END EC PRIVATE KEY----- visible", &mut out)
            .unwrap();
        masker.finish(&mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "short ghp_[credential: PEM private key ----…, masked] visible"
        );
    }
    #[test]
    fn undecided_url_refuses_without_releasing_prefix_and_fuses() {
        let mut masker = Masker::new();
        let mut out = Vec::new();
        masker.push("before https://u:", &mut out).unwrap();
        let error = masker.push(&"A".repeat(CARRY), &mut out).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(out, b"before ");
        assert!(masker.finish(&mut out).is_err());
        assert_eq!(out, b"before ");
    }
    #[test]
    fn actual_4096_byte_pieces_do_not_change_matches() {
        let text = format!("{}ghp_{} end", "a".repeat(4090), "A".repeat(36));
        let parts = text
            .as_bytes()
            .chunks(4096)
            .map(|p| std::str::from_utf8(p).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            masked(&parts),
            format!(
                "{}[credential: GitHub token ghp_…, masked] end",
                "a".repeat(4090)
            )
        );
        for rule in RULES {
            assert!(rule.prefix.len() + rule.minimum <= CARRY);
            if let Some(end) = rule.end {
                assert!(end.len() <= CARRY);
            }
        }
    }
    #[test]
    fn sink_failure_is_original_and_fuses_without_later_writes() {
        #[derive(Debug)]
        struct Marker;
        impl std::fmt::Display for Marker {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("marker")
            }
        }
        impl std::error::Error for Marker {}
        struct Broken {
            calls: usize,
        }
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                Err(io::Error::other(Marker))
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("no flush")
            }
        }
        let mut sink = Broken { calls: 0 };
        let mut masker = Masker::new();
        let error = masker.push("visible", &mut sink).unwrap_err();
        assert!(error.get_ref().unwrap().is::<Marker>());
        assert!(masker.push("later", &mut sink).is_err());
        assert!(masker.finish(&mut sink).is_err());
        assert_eq!(sink.calls, 1);
    }
}
