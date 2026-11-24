use super::*;

fn link(role: LinkRole, reference: &str, quote: Option<&str>, form: Option<&str>) -> TextLink {
    TextLink {
        role,
        reference: Reference::parse(reference).unwrap(),
        quote: quote.map(str::to_owned),
        form: form.map(str::to_owned),
    }
}

#[test]
fn reference_links_keep_text_order_and_each_role_once() {
    let text = "[message](message:codex:m) <sinex:ev/opaque> \
                [again](message:codex:m) ![card](message:codex:m \"card\") \
                ![again](message:codex:m \"excerpt\")";
    assert_eq!(
        links(text).unwrap(),
        [
            link(LinkRole::Mentions, "message:codex:m", None, None),
            link(LinkRole::Mentions, "sinex:ev/opaque", None, None),
            link(LinkRole::Embeds, "message:codex:m", None, Some("card")),
        ],
    );
}

#[test]
fn external_urls_keep_their_spelling_and_other_markdown_stays_plain() {
    assert_eq!(
        links(
            "[web](HTTPS://Example.test/a?utm_source=x) [local](guide.md) \
               [mail](mailto:operator@example.test) ![diagram](image.png \"not a form\")"
        )
        .unwrap(),
        [link(
            LinkRole::Mentions,
            "url:HTTPS://Example.test/a?utm_source=x",
            None,
            None,
        )],
    );
}

#[test]
fn an_external_url_reference_is_the_written_text_unchanged() {
    let found = links("[web](HTTPS://Example.test/a?utm_source=x)").unwrap();
    assert_eq!(
        found[0].reference,
        Reference::Url("HTTPS://Example.test/a?utm_source=x".into())
    );
}

#[test]
fn code_never_supplies_reference_links() {
    let text = "`[inline](block:codex:m#bad)`\n\n\
                ```query\n[block](message:codex:ignored)\n```\n\n\
                [real](message:codex:kept)";
    assert_eq!(
        links(text).unwrap(),
        [link(LinkRole::Mentions, "message:codex:kept", None, None)],
    );
}

#[test]
fn footnotes_cite_with_the_first_straight_or_curly_quote() {
    let text = "One.[^one] Two.[^two]\n\n\
                [^one]: “retained *words*” <span:message:codex:m#c=0-14>\n\
                [^two]: <span:message:codex:other#c=0-5> \"after the reference\" and \"not this\"\n";
    assert_eq!(
        links(text).unwrap(),
        [
            link(
                LinkRole::Cites,
                "span:message:codex:m#c=0-14",
                Some("retained words"),
                None,
            ),
            link(
                LinkRole::Cites,
                "span:message:codex:other#c=0-5",
                Some("after the reference"),
                None,
            ),
        ],
    );
}

#[test]
fn a_mention_and_a_citation_remain_distinct_and_repeated_cites_keep_first_quote() {
    let text = "[message](message:codex:m). One.[^one] Two.[^two]\n\n\
                [^one]: \"first\" <span:message:codex:m#c=0-5>\n\
                [^two]: \"second\" <span:message:codex:m#c=0-5>\n";
    assert_eq!(
        links(text).unwrap(),
        [
            link(LinkRole::Mentions, "message:codex:m", None, None),
            link(
                LinkRole::Cites,
                "span:message:codex:m#c=0-5",
                Some("first"),
                None
            ),
        ],
    );
}

#[test]
fn embeds_accept_the_declared_forms_and_report_unknown_forms() {
    for form in [
        "card",
        "excerpt",
        "text",
        "full",
        "table",
        "list",
        "count",
        "value",
        "chart",
        "compare",
        "live value",
    ] {
        assert_eq!(
            links(&format!("![caption](message:codex:m \"{form}\")")).unwrap(),
            [link(LinkRole::Embeds, "message:codex:m", None, Some(form))],
        );
    }
    assert_eq!(
        links("![caption](message:codex:m \"thumbnail\")"),
        Err(TextError::Form {
            written: "thumbnail".into(),
        }),
    );
}

#[test]
fn query_values_and_session_citations_name_the_unbuilt_behavior() {
    for form in ["value", "live value"] {
        assert_eq!(
            links(&format!("![amount](query:report \"{form}\")")),
            Err(TextError::NotBuilt {
                what: "values kept in sentences",
            }),
        );
    }
    assert_eq!(
        links("A source.[^one]\n\n[^one]: \"words\" <session:codex:thread>"),
        Err(TextError::NotBuilt {
            what: "citing a session by quote",
        }),
    );
    assert_eq!(
        links("![report](query:report \"card\")").unwrap(),
        [link(LinkRole::Embeds, "query:report", None, Some("card"))],
    );
}

#[test]
fn a_message_cited_by_quote_names_the_unbuilt_behavior() {
    assert_eq!(
        links("A source.[^one]\n\n[^one]: \"words\" <message:codex:m>"),
        Err(TextError::NotBuilt {
            what: "citing a message by quote",
        }),
    );
    // Without a quote a message is cited as written.
    assert_eq!(
        links("A source.[^one]\n\n[^one]: <message:codex:m>").unwrap(),
        [link(LinkRole::Cites, "message:codex:m", None, None)],
    );
}

#[test]
fn an_unclosed_footnote_quote_is_refused_with_its_position() {
    // "One.[^one]" is 10 bytes, the blank line 2 and "[^one]: " 8, so the
    // quote after "see " is at byte 24.
    let text = "One.[^one]\n\n[^one]: see \"unfinished <span:message:codex:m#c=0-5>";
    assert_eq!(links(text), Err(TextError::UnclosedQuote { position: 24 }));
    let curly = "One.[^one]\n\n[^one]: see “unfinished <span:message:codex:m#c=0-5>";
    assert_eq!(links(curly), Err(TextError::UnclosedQuote { position: 24 }));
}

#[test]
fn malformed_known_references_report_the_written_destination() {
    let Err(TextError::Reference { written, .. }) = links("[bad](block:codex:m#bad)") else {
        panic!("a malformed block reference must refuse the text")
    };
    assert_eq!(written, "block:codex:m#bad");
    let Err(TextError::Reference {
        error: ReferenceError::SinexLocal { global, .. },
        ..
    }) = links("[event](ev:opaque)")
    else {
        panic!("a local Sinex reference must name its global form")
    };
    assert_eq!(global, "sinex:ev/opaque");
}
