#!/usr/bin/env python
"""Fuzz timestamp parsing functions.

Target: DoS via slow regex or parsing functions.

Security properties tested:
- No catastrophic backtracking (ReDoS)
- Bounded execution time for arbitrary inputs
- No crashes on malformed timestamps
- Safe handling of extreme values
"""

from __future__ import annotations

import os
import sys
import time

# Check if atheris is available
try:
    import atheris
    HAS_ATHERIS = True
except ImportError:
    HAS_ATHERIS = False

import pytest

# Maximum time for a single parse operation (seconds)
MAX_PARSE_TIME = 1.0


def fuzz_parse_timestamp(data: bytes) -> None:
    """Fuzz the parse_timestamp function with arbitrary inputs."""
    from polylogue.lib.timestamps import parse_timestamp

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    start_time = time.monotonic()
    try:
        # Parse the timestamp
        result = parse_timestamp(text)

        # Check execution time (ReDoS detection)
        elapsed = time.monotonic() - start_time
        if elapsed > MAX_PARSE_TIME:
            raise AssertionError(
                f"parse_timestamp took {elapsed:.2f}s (max {MAX_PARSE_TIME}s) for input: {text[:100]!r}"
            )

        # Result should be datetime or None
        if result is not None:
            assert hasattr(result, "year"), f"parse_timestamp returned non-datetime: {type(result)}"

    except (ValueError, TypeError, OverflowError, OSError):
        # These are acceptable rejections
        pass
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(f"Unexpected exception: {type(e).__name__}: {e}") from e


def fuzz_normalize_timestamp(data: bytes) -> None:
    """Fuzz the normalize_timestamp function from claude importer."""
    from polylogue.sources.parsers.claude import normalize_timestamp

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    start_time = time.monotonic()
    try:
        # Parse the timestamp
        result = normalize_timestamp(text)

        # Check execution time
        elapsed = time.monotonic() - start_time
        if elapsed > MAX_PARSE_TIME:
            raise AssertionError(
                f"normalize_timestamp took {elapsed:.2f}s (max {MAX_PARSE_TIME}s) for input: {text[:100]!r}"
            )

        # Result should be string or None
        if result is not None:
            assert isinstance(result, str), f"normalize_timestamp returned non-string: {type(result)}"

    except (ValueError, TypeError, OverflowError, OSError):
        pass
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(f"Unexpected exception: {type(e).__name__}: {e}") from e


def fuzz_format_timestamp(data: bytes) -> None:
    """Fuzz the format_timestamp function."""
    from polylogue.lib.timestamps import format_timestamp

    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return

    # Try to interpret as a number
    try:
        num = float(text)
    except ValueError:
        return

    start_time = time.monotonic()
    try:
        result = format_timestamp(num)

        elapsed = time.monotonic() - start_time
        if elapsed > MAX_PARSE_TIME:
            raise AssertionError(
                f"format_timestamp took {elapsed:.2f}s for input: {num}"
            )

        # Result should be string
        assert isinstance(result, str), f"format_timestamp returned non-string: {type(result)}"

    except (ValueError, TypeError, OverflowError, OSError):
        pass
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(f"Unexpected exception: {type(e).__name__}: {e}") from e


def fuzz_all_timestamps(data: bytes) -> None:
    """Combined fuzzer for all timestamp functions."""
    fuzz_parse_timestamp(data)
    fuzz_normalize_timestamp(data)
    fuzz_format_timestamp(data)


# =============================================================================
# Seed corpus
# =============================================================================

TIMESTAMP_CORPUS = [
    # Valid epochs
    b"0",
    b"1704067200",
    b"1704067200.123",
    b"-1",
    b"-1704067200",
    # ISO 8601
    b"2024-01-01T00:00:00Z",
    b"2024-01-01T00:00:00+00:00",
    b"2024-01-01T00:00:00-05:00",
    b"2024-01-01 00:00:00",
    b"2024-01-01",
    # Edge cases
    b"",
    b"   ",
    b"null",
    b"NaN",
    b"Infinity",
    b"-Infinity",
    b"not a timestamp",
    # Extreme values
    b"0.0",
    b"9999999999999",
    b"-9999999999999",
    b"1e20",
    b"-1e20",
    b"1e-20",
    # Milliseconds (common format)
    b"1704067200000",
    b"1704067200123",
    # Potential ReDoS patterns
    b"a" * 30 + b"!",
    b"0" * 50 + b"x",
    b"1" * 20 + b"." + b"1" * 20 + b"." + b"1" * 20,
    b"2024-01-01" + b"-01" * 50,
    # Malformed ISO
    b"2024-13-45T99:99:99Z",
    b"2024-01-01T00:00:00+99:99",
    b"2024-01-01T00:00:00.123456789Z",
    # Unicode
    b"\xc2\xa0",  # Non-breaking space
    b"2024\xe2\x80\x8b01\xe2\x80\x8b01",  # Zero-width spaces
    # Control characters
    b"2024-01-01\x00T00:00:00Z",
    b"\x001704067200",
    # Very long strings
    b"1" * 10000,
    b"2024-01-01T00:00:00+" + b"0" * 1000,
]


# =============================================================================
# Pytest-compatible test functions
# =============================================================================


class TestTimestampFuzz:
    """Pytest-compatible fuzz tests using seed corpus."""

    @pytest.mark.parametrize("data", TIMESTAMP_CORPUS)
    def test_parse_timestamp_corpus(self, data: bytes):
        """Run parse_timestamp fuzz with seed corpus."""
        fuzz_parse_timestamp(data)

    @pytest.mark.parametrize("data", TIMESTAMP_CORPUS)
    def test_normalize_timestamp_corpus(self, data: bytes):
        """Run normalize_timestamp fuzz with seed corpus."""
        fuzz_normalize_timestamp(data)

    @pytest.mark.parametrize("data", TIMESTAMP_CORPUS)
    def test_format_timestamp_corpus(self, data: bytes):
        """Run format_timestamp fuzz with seed corpus."""
        fuzz_format_timestamp(data)

    def test_parse_timestamp_random(self):
        """Run parse_timestamp with random bytes."""
        import random

        for _ in range(1000):
            length = random.randint(1, 200)
            data = bytes(random.randint(0, 255) for _ in range(length))
            fuzz_parse_timestamp(data)

    def test_parse_timestamp_numeric_strings(self):
        """Run parse_timestamp with random numeric strings."""
        import random

        for _ in range(1000):
            # Generate random numeric-like strings
            digits = "".join(str(random.randint(0, 9)) for _ in range(random.randint(1, 30)))
            if random.random() < 0.3:
                # Add decimal point
                pos = random.randint(0, len(digits))
                digits = digits[:pos] + "." + digits[pos:]
            if random.random() < 0.1:
                # Add negative sign
                digits = "-" + digits

            fuzz_parse_timestamp(digits.encode("utf-8"))

    def test_parse_timestamp_date_strings(self):
        """Run parse_timestamp with random date-like strings."""
        import random

        for _ in range(500):
            # Generate random date-like strings
            year = random.randint(-9999, 9999)
            month = random.randint(0, 99)
            day = random.randint(0, 99)
            hour = random.randint(0, 99)
            minute = random.randint(0, 99)
            second = random.randint(0, 99)

            sep = random.choice(["T", " ", "t", "_", "-"])
            tz = random.choice(["Z", "+00:00", "-05:00", "", "+99:99", "UTC"])

            date_str = f"{year:04d}-{month:02d}-{day:02d}{sep}{hour:02d}:{minute:02d}:{second:02d}{tz}"
            fuzz_parse_timestamp(date_str.encode("utf-8"))


# =============================================================================
# Standalone atheris fuzzer
# =============================================================================


def main():
    """Run atheris fuzzer with libFuzzer engine."""
    if not HAS_ATHERIS:
        print("atheris not installed, running pytest-compatible tests instead")
        pytest.main([__file__, "-v"])
        return

    iterations = int(os.environ.get("FUZZ_ITERATIONS", "10000"))
    print(f"Running atheris fuzzer for {iterations} iterations...")

    atheris.Setup(
        sys.argv + ["-max_total_time=300", f"-runs={iterations}"],
        fuzz_all_timestamps,
    )
    atheris.Fuzz()


if __name__ == "__main__":
    main()
