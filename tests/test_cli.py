"""The command line, exercised end to end.

Buffers are built by hand rather than taken from an extraction, so this test
carries no sample data.
"""
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hive_parser.__main__ import main  # noqa: E402
from test_hive_parser import (frame, v_dt18, v_int, v_obj, v_str)  # noqa: E402


class CliTest(unittest.TestCase):

    def setUp(self):
        handle = tempfile.NamedTemporaryFile(suffix='.hive', delete=False)
        handle.write(b''.join([
            frame('theme', v_str('system')),
            frame('theme', v_str('dark')),
            frame('count', v_int(7)),
            frame('when', v_dt18(1600000000000, True)),
            frame('rec', v_obj(34, [(0, v_str('title')), (1, v_int(2))])),
            frame('gone', v_str('bye')),
            frame('gone'),
        ]))
        handle.close()
        self.path = handle.name

    def tearDown(self):
        os.unlink(self.path)

    def run_cli(self, *argv):
        out = io.StringIO()
        stdout, sys.stdout = sys.stdout, out
        try:
            code = main(list(argv))
        finally:
            sys.stdout = stdout
        return code, out.getvalue()

    def test_dump_lists_every_frame_including_superseded_and_deleted(self):
        code, text = self.run_cli('dump', self.path)
        self.assertEqual(code, 0)
        self.assertIn('Frames      7 on disk', text)
        self.assertIn('Live keys   4', text)
        self.assertIn('Overwritten 1', text)
        self.assertIn('Removed     1', text)
        self.assertIn('Deletions   1 tombstone', text)
        self.assertIn('"system"', text)        # the superseded value is shown
        self.assertIn('"dark"', text)

    def test_live_shows_only_what_the_app_reads(self):
        code, text = self.run_cli('dump', '--live', self.path)
        self.assertEqual(code, 0)
        self.assertIn('"dark"', text)
        self.assertNotIn('"system"', text)     # superseded, so not live
        self.assertNotIn('bye', text)          # deleted

    def test_json_is_machine_readable_and_types_survive(self):
        code, text = self.run_cli('dump', '--json', self.path)
        self.assertEqual(code, 0)
        blob = json.loads(text)
        self.assertEqual(blob['frames_on_disk'], 7)
        self.assertEqual(blob['live_keys'], 4)
        when = blob['live']['when']
        self.assertEqual(when['_type'], 'DateTime')
        self.assertEqual(when['ms'], 1600000000000)
        self.assertTrue(when['zone_recorded'])
        rec = blob['live']['rec']
        self.assertEqual(rec['on_disk_type_id'], 34)
        self.assertEqual(rec['app_declared_type_id'], 2)
        deleted = [f for f in blob['frames'] if f['deleted']]
        self.assertEqual(len(deleted), 1)
        self.assertEqual(deleted[0]['key'], 'gone')

    def test_hex_accounts_for_every_byte_of_a_frame(self):
        code, text = self.run_cli('hex', self.path)
        self.assertEqual(code, 0)
        self.assertIn('frame length', text)
        self.assertIn('key type 1 = UTF-8 string', text)
        self.assertIn('key = "theme"', text)
        self.assertIn('value type 4 = string', text)
        self.assertIn('CRC32 verified', text)

    def test_hex_names_the_is_utc_byte_of_a_datetime(self):
        code, text = self.run_cli('hex', '--frame', '3', self.path)
        self.assertEqual(code, 0)
        self.assertIn('value type 18 = DateTime', text)
        self.assertIn('ms since epoch', text)
        self.assertIn('is_utc = True', text)

    def test_a_directory_reads_every_box_under_it(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('a.hive', 'b.hive'):
                with open(os.path.join(folder, name), 'wb') as handle:
                    handle.write(frame('k', v_str('v')))
            code, text = self.run_cli('dump', folder)
        self.assertEqual(code, 0)
        self.assertEqual(text.count('Frames      1 on disk'), 2)


if __name__ == '__main__':
    unittest.main()
