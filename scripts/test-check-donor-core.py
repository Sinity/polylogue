"""Pure provider-version controls for the donor upgrade driver."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('donor_driver', ROOT / 'scripts/check-donor-core.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class UpgradeVersionTests(unittest.TestCase):
    def test_each_retained_fixture_uses_its_provider_parser_version(self):
        manifest = json.loads((ROOT / 'crates/sources/tests/fixtures/donor/manifest.json').read_text())
        expected_providers = {
            'claude-normalization-main.jsonl': 'claude_code',
            'legacy-response-records.jsonl': 'codex',
            'tool-result-headers.jsonl': 'codex',
        }
        self.assertEqual({Path(path).name for path in manifest['files']}, set(expected_providers))
        for name, provider in expected_providers.items():
            with self.subTest(name=name):
                source = (ROOT / 'crates/sources/src' / provider / 'mod.rs').read_text()
                actual = int(re.search(r'pub const PARSER_VERSION: u32 = (\d+);', source).group(1))
                self.assertEqual(driver.expected_parser_version(name), actual)

    def test_unknown_input_does_not_inherit_a_provider_version(self):
        with self.assertRaises(KeyError):
            driver.expected_parser_version('unknown.jsonl')


if __name__ == '__main__':
    unittest.main()
