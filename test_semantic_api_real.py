#!/usr/bin/env python3
"""Test semantic API methods on real conversations."""

from polylogue import Polylogue

# Test conversation IDs (one from each provider)
TEST_CONVOS = [
    "chatgpt:0012f391-9a6b-45c5-9653-972266abcb3c",  # 34 messages
    "claude:00caad11-9ac4-475b-8b74-6ab0f7991140",    # 4 messages
    "claude-code:003403ec-1c28-42f4-9a0b-cfd95157cb1d",  # 2 messages
    "gemini:1-IL7HyiiNlSJBaH8VHqjlg-YpDjh7hAT",      # 4 messages
]

def test_conversation(archive: Polylogue, conv_id: str):
    """Test all semantic API methods on a conversation."""
    print(f"\n{'='*70}")
    print(f"Testing: {conv_id}")
    print(f"{'='*70}")

    conv = archive.get_conversation(conv_id)
    if not conv:
        print(f"❌ Failed to load conversation {conv_id}")
        return

    print(f"✓ Loaded: {conv.title[:50]}")
    print(f"  Provider: {conv.provider}")
    print(f"  Total messages: {len(conv.messages)}")

    # Test iter_pairs()
    try:
        pairs = list(conv.iter_pairs())
        print(f"\n✓ iter_pairs(): {len(pairs)} user/assistant pairs")
        if pairs:
            first_pair = pairs[0]
            print(f"  First pair: user='{first_pair.user.text[:50]}...' assistant='{first_pair.assistant.text[:50]}...'")
    except Exception as e:
        print(f"❌ iter_pairs() failed: {e}")

    # Test iter_thinking()
    try:
        thinking = list(conv.iter_thinking())
        print(f"\n✓ iter_thinking(): {len(thinking)} thinking blocks")
        if thinking:
            print(f"  First thinking: '{thinking[0].text[:80]}...'")
    except Exception as e:
        print(f"❌ iter_thinking() failed: {e}")

    # Test iter_substantive()
    try:
        substantive = list(conv.iter_substantive())
        print(f"\n✓ iter_substantive(): {len(substantive)} substantive messages")
        substantive_ratio = len(substantive) / len(conv.messages) if conv.messages else 0
        print(f"  Substantive ratio: {substantive_ratio:.1%}")
    except Exception as e:
        print(f"❌ iter_substantive() failed: {e}")

    # Test without_noise()
    try:
        clean_conv = conv.without_noise()
        clean_msgs = clean_conv.messages
        print(f"\n✓ without_noise(): {len(clean_msgs)} messages (filtered {len(conv.messages) - len(clean_msgs)} noise)")
    except Exception as e:
        print(f"❌ without_noise() failed: {e}")

    # Test substantive_only()
    try:
        sub_conv = conv.substantive_only()
        print(f"\n✓ substantive_only(): {len(sub_conv.messages)} messages in filtered conversation")
    except Exception as e:
        print(f"❌ substantive_only() failed: {e}")

    # Test to_text()
    try:
        text = conv.to_text(include_role=True)
        print(f"\n✓ to_text(): {len(text)} characters")
        print(f"  Preview: '{text[:100]}...'")
    except Exception as e:
        print(f"❌ to_text() failed: {e}")

def main():
    archive = Polylogue()

    print("SEMANTIC API VALIDATION ON REAL CONVERSATIONS")
    print("=" * 70)

    for conv_id in TEST_CONVOS:
        test_conversation(archive, conv_id)

    print(f"\n{'='*70}")
    print("SEMANTIC API TESTING COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
