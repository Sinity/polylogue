//! A signal outbox in the shape the host policy hands off, for tests of what
//! stores, recovers and delivers it.
use polylogue_model::{
    authored::ExecutionEligibility,
    signal::{Metadata, Outbox},
    user::Delivery as PreparedDelivery,
};
use shared_signals::notifications::{Notice, Outlets};
use shared_signals::{
    Attention, Author, Channel, Decision, Delivery, Fired, Occurrence, RuleVersion, Shipped,
    Trigger,
};

/// One delivery of the synthetic rule `rule` for `occurrence`, handed off at
/// `at`. An interrupting delivery is shown on the desktop; any other is a value
/// delivery run through its configured target.
pub fn outbox(rule: &str, occurrence: &str, at: i64, interrupts: bool) -> Outbox {
    let trigger = Trigger {
        rule: RuleVersion {
            product: "polylogue".into(),
            id: rule.into(),
            revision: 1,
        },
        occurrence: Occurrence {
            id: occurrence.into(),
            period: "one".into(),
        },
    };
    let id = trigger.delivery_id();
    let (channel, decision) = if interrupts {
        (Channel::Interrupt, Decision::Show { send_away: false })
    } else {
        (Channel::Value, Decision::Run)
    };
    let fired = Fired {
        id: id.clone(),
        trigger,
        rule_name: rule.into(),
        matched: at,
        at,
        basis: vec![],
        channel,
        change: None,
        value: (!interrupts).then_some(2.0),
        scope: None,
        to: vec![],
        author: Author::Operator,
        quiet_hours: None,
        rate_limit: None,
        current: false,
        shipped: Shipped::default(),
    };
    let notice = Notice {
        delivery_id: id,
        product: "polylogue".into(),
        icon: String::new(),
        desktop_entry: None,
        rule: rule.into(),
        occurrence: occurrence.into(),
        summary: occurrence.into(),
        body: String::new(),
        lines: vec![],
        reference: "repo:synthetic".into(),
        command_hint: String::new(),
        current: false,
        transient: false,
        critical: false,
        outlets: Outlets::of(&decision),
        actions: vec![],
        showing: 0,
    };
    Outbox {
        record: PreparedDelivery {
            recipient_kind: "operator".into(),
            recipient_id: None,
            sender: "signal".into(),
            event: "signal".into(),
            cwd: None,
            text: occurrence.into(),
            refs: vec![],
            left_out: vec![],
        },
        metadata: Metadata {
            delivery: Delivery {
                fired,
                at,
                decision,
                held: vec![],
                attention: Attention {
                    at,
                    ..Attention::default()
                },
            },
            notice,
            outlet: None,
            execution: ExecutionEligibility::Live,
        },
    }
}
