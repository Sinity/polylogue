import unittest
import measure


class CpuAttribution(unittest.TestCase):
    def data(self):
        parts = {name: {'calls': 1, 'seconds': 1.0, 'cpu_seconds': 1.0}
                 for name in measure.REQUIRED_PARTS}
        record = {'first_build_attribution': parts}
        profile = {'workers': {'polylogue-writer': {'cpu_seconds': 33.0}},
                   'pipeline_timings': {'rows': 90.0, 'parse': 80.0}}
        return profile, record

    def test_writer_reconciliation_excludes_other_thread_and_keeps_overlapping_walls(self):
        profile, record = self.data()
        result = measure.summarize(profile, [record])
        self.assertEqual(result['exclusive_writer_cpu_seconds'], 33.0)
        self.assertEqual(result['writer_cpu_coverage'], 1.0)
        self.assertEqual(result['writer_cpu_residual_seconds'], 0.0)
        self.assertEqual(result['overlapping_stage_seconds'], {'rows': 90.0, 'parse': 80.0})

    def test_missing_or_unknown_cpu_cannot_be_a_successful_profile(self):
        profile, record = self.data()
        record['first_build_attribution']['selected_blocks']['cpu_seconds'] = None
        with self.assertRaises(RuntimeError):
            measure.summarize(profile, [record])
        del record['first_build_attribution']['selected_blocks']
        with self.assertRaises(RuntimeError):
            measure.summarize(profile, [record])

    def test_no_profile_cannot_be_accepted(self):
        profile, _ = self.data()
        with self.assertRaises(RuntimeError):
            measure.summarize(profile, [])


if __name__ == '__main__':
    unittest.main()
