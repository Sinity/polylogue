"""Provider-neutral payloads and projections shared by public adapters."""

from polylogue.surfaces.authority import AuthorityBlock, AuthorityEnvelope, build_authority_envelope

__all__ = ["AuthorityBlock", "AuthorityEnvelope", "build_authority_envelope"]
