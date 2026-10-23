use super::*;
use crate::excise::text_matches::{Error, HitSink, Scope, TextMatches};

fn finish(scanner: TextMatches, mut hits: super::super::matching::Hits<'_>) -> Result<()> {
    let accepted = scanner.finish(&mut hits).map_err(|e| match e {
        Error::Sink(error) => error,
        Error::Syntax(message) => StoreError::Invalid(message),
    })?;
    hits.finish(accepted.scope)
}

fn staged(plan: &Plan) -> i64 {
    plan.conn
        .query_row("SELECT count(*) FROM candidate_matches", [], |r| r.get(0))
        .unwrap()
}

#[test]
fn completed_values_keep_hits_on_disk_until_chain_completion() {
    let plan = Plan::new().unwrap();
    let mut hits = plan.hits("input", 1);
    let mut scanner = TextMatches::new("secret", &mut hits).unwrap();
    scanner.feed(b"[", &mut hits).unwrap();
    for _ in 0..20000 {
        scanner.feed(br#""secret","#, &mut hits).unwrap();
        assert!(scanner.retained_bytes() < 4096);
    }
    assert_eq!(staged(&plan), 20000);
    assert!(plan.counts().unwrap().is_empty());
    scanner.feed(b"]", &mut hits).unwrap();
    finish(scanner, hits).unwrap();
    assert_eq!(plan.counts().unwrap()["input"], 20000);
    assert_eq!(staged(&plan), 0);
}

#[test]
fn huge_tentative_values_keys_and_whitespace_keep_only_bounded_matcher_state() {
    for tail in [b"{}".as_slice(), b"plain", b""] {
        let plan = Plan::new().unwrap();
        let mut hits = plan.hits("input", 1);
        let mut scanner = TextMatches::new(" ", &mut hits).unwrap();
        for _ in 0..128 {
            scanner.feed(&[b' '; 1024], &mut hits).unwrap();
            assert!(scanner.retained_bytes() < 4096);
        }
        assert_eq!(staged(&plan), 128 * 1024);
        scanner.feed(tail, &mut hits).unwrap();
        finish(scanner, hits).unwrap();
        assert_eq!(
            plan.counts().unwrap().get("input").copied().unwrap_or(0),
            if tail == b"{}" { 0 } else { 128 * 1024 }
        );
        assert_eq!(staged(&plan), 0);
    }
    for tail in [b":0]".as_slice(), b",0]", b""] {
        let plan = Plan::new().unwrap();
        let mut hits = plan.hits("input", 1);
        let mut scanner = TextMatches::new("secret", &mut hits).unwrap();
        scanner.feed(b"[\"", &mut hits).unwrap();
        for _ in 0..128 {
            scanner.feed(&b"secret".repeat(1024), &mut hits).unwrap();
            assert!(scanner.retained_bytes() < 4096);
        }
        scanner.feed(b"\"", &mut hits).unwrap();
        for _ in 0..128 {
            scanner.feed(&[b' '; 1024], &mut hits).unwrap();
            assert!(scanner.retained_bytes() < 4096);
        }
        assert_eq!(staged(&plan), 128 * 1024);
        let fed = scanner.feed(tail, &mut hits);
        if tail.starts_with(b":") {
            assert!(matches!(fed, Err(Error::Syntax(_))));
            assert_eq!(staged(&plan), 0);
        } else {
            fed.unwrap();
            finish(scanner, hits).unwrap();
            assert_eq!(plan.counts().unwrap()["input"], 128 * 1024);
            assert_eq!(staged(&plan), 0);
        }
    }
}

#[test]
fn huge_nested_candidates_accept_or_discard_exact_hits_without_retaining_them() {
    for valid in [false, true] {
        let plan = Plan::new().unwrap();
        let mut hits = plan.hits("input", 1);
        let mut scanner = TextMatches::new("secret", &mut hits).unwrap();
        let prefix = serde_json::to_string(r#"{"v":""#).unwrap();
        scanner
            .feed(&prefix.as_bytes()[..prefix.len() - 1], &mut hits)
            .unwrap();
        for _ in 0..128 {
            let part = serde_json::to_string(&r#"s\u0065cret"#.repeat(1024)).unwrap();
            scanner
                .feed(&part.as_bytes()[1..part.len() - 1], &mut hits)
                .unwrap();
            assert!(scanner.retained_bytes() < 8192);
        }
        assert_eq!(staged(&plan), 128 * 1024);
        let suffix = serde_json::to_string(if valid {
            r#""}"#
        } else {
            r#"","later":"unfinished"#
        })
        .unwrap();
        scanner.feed(&suffix.as_bytes()[1..], &mut hits).unwrap();
        finish(scanner, hits).unwrap();
        assert_eq!(
            plan.counts().unwrap().get("input").copied().unwrap_or(0),
            if valid { 128 * 1024 } else { 0 }
        );
        assert_eq!(staged(&plan), 0);
    }
}

#[test]
fn disk_scopes_preserve_all_splits_equal_hits_overlaps_and_nested_rejection() {
    for source in [r#"{"v":"ababa"}"#, r#"{"v":"{\"v\":\"ababa\"}"}"#] {
        for cut in 0..=source.len() {
            let plan = Plan::new().unwrap();
            let mut hits = plan.hits("input", 1);
            let mut scanner = TextMatches::new("aba", &mut hits).unwrap();
            scanner.feed(&source.as_bytes()[..cut], &mut hits).unwrap();
            scanner.feed(&source.as_bytes()[cut..], &mut hits).unwrap();
            finish(scanner, hits).unwrap();
            assert_eq!(plan.counts().unwrap()["input"], 2, "cut {cut}: {source}");
            assert_eq!(staged(&plan), 0);
        }
    }
    for (source, needle, expected) in [
        (r#"{"v":"é\uD83D\uDE00"}"#, "é😀", 1),
        (
            r#"{"v":"{\"v\":\"secret s\\u0065cret\",\"later\":\"unfinished"}"#,
            "secret",
            1,
        ),
    ] {
        for cut in 0..=source.len() {
            let plan = Plan::new().unwrap();
            let mut hits = plan.hits("input", 1);
            let mut scanner = TextMatches::new(needle, &mut hits).unwrap();
            scanner.feed(&source.as_bytes()[..cut], &mut hits).unwrap();
            scanner.feed(&source.as_bytes()[cut..], &mut hits).unwrap();
            finish(scanner, hits).unwrap();
            assert_eq!(
                plan.counts().unwrap()["input"],
                expected,
                "cut {cut}: {source}"
            );
            assert_eq!(staged(&plan), 0);
        }
    }
    for bytes in [
        br#"{"v":"{\"secret\":\"value\"}"}"#.as_slice(),
        br#"{"secret":"value"}"#,
    ] {
        let plan = Plan::new().unwrap();
        let mut hits = plan.hits("input", 1);
        let mut scanner = TextMatches::new("secret", &mut hits).unwrap();
        assert!(matches!(
            scanner.feed(bytes, &mut hits),
            Err(Error::Syntax(_))
        ));
    }
}

struct Failing<'a> {
    inner: super::super::matching::Hits<'a>,
    plan: &'a Plan,
    action: &'static str,
    failed: bool,
}

impl Failing<'_> {
    fn check(&mut self, action: &str, scope: Scope) -> Result<()> {
        let has_hits: bool = self.plan.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM candidate_matches WHERE scope=?1)",
            [scope as i64],
            |r| r.get(0),
        )?;
        if action == self.action && (action == "hit" || has_hits) {
            self.failed = true;
            return Err(StoreError::Io(std::io::Error::other(
                "synthetic scratch failure",
            )));
        }
        Ok(())
    }
}

impl HitSink for Failing<'_> {
    fn scope(&mut self) -> Result<Scope> {
        self.inner.scope()
    }
    fn hit(&mut self, scope: Scope, start: u64, end: u64) -> Result<()> {
        self.check("hit", scope)?;
        self.inner.hit(scope, start, end)
    }
    fn accept(&mut self, scope: Scope, parent: Scope) -> Result<()> {
        self.check("accept", scope)?;
        self.inner.accept(scope, parent)
    }
    fn discard(&mut self, scope: Scope) -> Result<()> {
        self.check("discard", scope)?;
        self.inner.discard(scope)
    }
}

#[test]
fn nested_sqlite_write_failure_is_not_rejected_embedded_syntax() {
    let plan = Plan::new().unwrap();
    plan.conn
        .execute_batch(
            "CREATE TRIGGER fail_hit BEFORE INSERT ON candidate_matches
        BEGIN SELECT RAISE(FAIL,'synthetic scratch write failure'); END;",
        )
        .unwrap();
    let mut hits = plan.hits("input", 1);
    let mut scanner = TextMatches::new("secret", &mut hits).unwrap();
    let bytes = serde_json::to_vec(&serde_json::json!({"v": r#"{"v":"s\u0065cret"}"#})).unwrap();
    assert!(matches!(
        scanner.feed(&bytes, &mut hits),
        Err(Error::Sink(StoreError::Sqlite(_)))
    ));
    assert!(plan.counts().unwrap().is_empty());
}

#[test]
fn nested_hit_accept_and_discard_failures_abort_preflight_as_storage_errors() {
    for action in ["hit", "accept", "discard"] {
        let plan = Plan::new().unwrap();
        let mut sink = Failing {
            inner: plan.hits("input", 1),
            plan: &plan,
            action,
            failed: false,
        };
        let mut scanner = TextMatches::new("secret", &mut sink).unwrap();
        let inner = if action == "discard" {
            r#"{"v":"s\u0065cret","later":"unfinished"#
        } else {
            r#"{"v":"s\u0065cret"}"#
        };
        let source = serde_json::to_vec(&serde_json::json!({"v":inner})).unwrap();
        let result = scanner
            .feed(&source, &mut sink)
            .and_then(|()| scanner.finish(&mut sink).map(|_| ()));
        assert!(sink.failed, "{action}");
        assert!(
            matches!(result, Err(Error::Sink(StoreError::Io(_)))),
            "{action}: {result:?}"
        );
        assert!(plan.counts().unwrap().is_empty());
    }
}
