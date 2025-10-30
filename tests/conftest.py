import os
import sys
import types

class _SimpleEncoding:
    def __init__(self, name: str):
        self.name = name

    def encode(self, text: str):
        if not text:
            return []
        # Split on whitespace as a deterministic approximation.
        return text.split()


def _build_tiktoken_stub() -> types.ModuleType:
    module = types.ModuleType("tiktoken")

    def _get_encoding(name: str) -> _SimpleEncoding:
        return _SimpleEncoding(name)

    def _encoding_for_model(model: str) -> _SimpleEncoding:
        return _SimpleEncoding(model)

    module.get_encoding = _get_encoding  # type: ignore[attr-defined]
    module.encoding_for_model = _encoding_for_model  # type: ignore[attr-defined]
    core_module = types.ModuleType("tiktoken.core")
    core_module.Encoding = _SimpleEncoding  # type: ignore[attr-defined]
    module.core = core_module  # type: ignore[attr-defined]
    sys.modules.setdefault("tiktoken.core", core_module)
    return module


sys.modules.setdefault("tiktoken", _build_tiktoken_stub())

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
