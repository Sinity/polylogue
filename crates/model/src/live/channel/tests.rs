use super::*;

fn probe(hosted: bool, hooks: bool, permission_hook: bool, headless_resume: bool) -> Probe {
    Probe {
        harness: "claude-code".into(),
        hosted,
        hooks,
        permission_hook,
        headless_resume,
    }
}

/// law: pl.live.channel-pure
/// The channel is a function of the act, the state and the probe; an
/// unconfirmed mechanism is refused naming the harness, never replaced by
/// another channel.
#[test]
fn the_channel_is_a_function_of_act_state_and_probe() {
    use LiveState::*;
    let observed = probe(false, true, true, true);
    let hosted = probe(true, true, true, true);
    let bare = probe(false, false, false, false);
    let ping = Act::Ping { live: false };
    let live_ping = Act::Ping { live: true };
    for state in [Working, Waiting, Idle, Unknown] {
        assert_eq!(channel(ping, state, &observed), Ok(Channel::Hook));
        assert_eq!(channel(ping, state, &hosted), Ok(Channel::Protocol));
        assert_eq!(channel(Act::Answer, state, &observed), Ok(Channel::Hook));
        assert_eq!(
            channel(Act::Interrupt, state, &observed).unwrap_err().reason,
            "unsupported"
        );
        assert_eq!(channel(Act::Interrupt, state, &hosted), Ok(Channel::Protocol));
        assert_eq!(
            channel(Act::Resume, state, &observed).unwrap_err().reason,
            "session_running"
        );
    }
    for state in [Ended, Stale] {
        assert_eq!(channel(ping, state, &observed), Ok(Channel::Resume));
        assert_eq!(channel(Act::Resume, state, &observed), Ok(Channel::Resume));
        assert_eq!(
            channel(live_ping, state, &observed).unwrap_err().reason,
            "session_not_live"
        );
        let refused = channel(ping, state, &bare).unwrap_err();
        assert_eq!(refused.reason, "unsupported");
        assert!(refused.detail.contains("claude-code"));
    }
    // The same inputs always give the same answer.
    assert_eq!(
        channel(ping, Working, &observed),
        channel(ping, Working, &observed)
    );
    assert_eq!(channel(Act::Answer, Working, &bare).unwrap_err().reason, "unsupported");
}
