"""Pin the on-disk contract the Hive reader depends on.

A box is append-only, so writing a key appends a frame rather than editing the
existing one. A reader that collapses to a dict without honouring file order
reports a stale value as current, which looks exactly like a correct answer. A
deletion is a frame carrying the key and no value, so a reader that treats the
value as mandatory turns a deletion into a parse failure, and one that stores
None for it cannot tell a deleted key from a stored null.

Several traps are specific to this format.

A map is written as a field count and then that many *pairs*. A reader that
takes it for that many values returns half the map and leaves its cursor inside
the entry, so everything after it in the frame decodes as plausible nonsense.

An integer is stored as a float64. Unpacking eight bytes as a little-endian int
returns a believable and entirely wrong number.

A DateTime arrives from either of two adapters. Type 18 carries an explicit
is_utc flag; type 16 carries milliseconds and no zone at all, so reporting it as
UTC asserts something the record does not contain. Type 13 is not a DateTime in
any version: unassigned in hive, and intSetT in hive_ce.

A hiveList is a count, then the name of the box the keys live in, then those
keys. It is not a list of tagged values.

One value that does not parse must cost its own frame and no others, or a single
unreadable entry silently truncates the box.

Buffers are built by hand rather than taken from an extraction, so this test
carries no sample data.
"""
import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hive_parser  # noqa: E402
from hive_parser import (HiveDateTime, HiveListRef, HiveObject, HiveSet,  # noqa: E402
                         live_entries, read_frames, superseded_counts)


# --- builders ---------------------------------------------------------------

def frame(key, value=None, int_key=False):
    """One frame as Hive writes it. value=None makes a deletion."""
    if int_key:
        body = b'\x00' + struct.pack('<I', key)
    else:
        encoded = key.encode('utf-8')
        body = b'\x01' + bytes([len(encoded)]) + encoded
    if value is not None:
        body += value
    head = struct.pack('<I', 4 + len(body) + 4) + body
    return head + struct.pack('<I', zlib.crc32(head) & 0xFFFFFFFF)


def v_null():
    return b'\x00'


def v_int(n):
    return b'\x01' + struct.pack('<d', float(n))


def v_double(n):
    return b'\x02' + struct.pack('<d', n)


def v_bool(b):
    return b'\x03' + bytes([1 if b else 0])


def v_str(s):
    e = s.encode('utf-8')
    return b'\x04' + struct.pack('<I', len(e)) + e


def v_bytes(b):
    return b'\x05' + struct.pack('<I', len(b)) + b


def v_intlist(ns, tid=6):
    return bytes([tid]) + struct.pack('<I', len(ns)) + b''.join(
        struct.pack('<d', float(n)) for n in ns)


def v_strlist(ss, tid=9):
    out = bytes([tid]) + struct.pack('<I', len(ss))
    for s in ss:
        e = s.encode('utf-8')
        out += struct.pack('<I', len(e)) + e
    return out


def v_list(items, tid=10):
    return bytes([tid]) + struct.pack('<I', len(items)) + b''.join(items)


def v_map(pairs):
    out = b'\x0b' + struct.pack('<I', len(pairs))
    for k, val in pairs:
        out += k + val
    return out


def v_hivelist(box, keys):
    e = box.encode('utf-8')
    out = b'\x0c' + struct.pack('<I', len(keys)) + bytes([len(e)]) + e
    for k in keys:
        ke = k.encode('utf-8')
        out += b'\x01' + bytes([len(ke)]) + ke
    return out


def v_dt16(ms):
    return b'\x10' + struct.pack('<d', float(ms))


def v_bigint(n):
    s = str(n).encode('ascii')
    return b'\x11' + bytes([len(s)]) + s


def v_dt18(ms, is_utc):
    return b'\x12' + struct.pack('<d', float(ms)) + bytes([1 if is_utc else 0])


def v_obj(type_id, fields):
    out = bytes([type_id]) + bytes([len(fields)])
    for idx, val in fields:
        out += bytes([idx]) + val
    return out


def box(*frames):
    handle = tempfile.NamedTemporaryFile(suffix='.hive', delete=False)
    handle.write(b''.join(frames))
    handle.close()
    return handle.name


class HiveTest(unittest.TestCase):
    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    def box(self, *frames):
        p = box(*frames)
        self._paths.append(p)
        return p


# --- basics -----------------------------------------------------------------

class TestFrameBasics(HiveTest):

    def test_string_key_and_value_round_trip(self):
        frames = read_frames(self.box(frame('greeting', v_str('hello'))))
        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0]['crc_ok'])
        self.assertIsNone(frames[0]['error'])
        self.assertEqual(frames[0]['key'], 'greeting')
        self.assertEqual(frames[0]['value'], 'hello')

    def test_integer_keys_are_uint32(self):
        live, _ = hive_parser.hive_entries(
            self.box(frame(7, v_str('seven'), int_key=True)))
        self.assertEqual(live, {7: 'seven'})

    def test_integer_values_are_stored_as_float64(self):
        # Unpacking these eight bytes as an int gives a plausible wrong answer.
        live, _ = hive_parser.hive_entries(self.box(frame('n', v_int(123456789))))
        self.assertEqual(live['n'], 123456789)

    def test_a_bad_crc_stops_the_walk_by_default(self):
        good = frame('a', v_str('one'))
        bad = bytearray(frame('b', v_str('two')))
        bad[-1] ^= 0xFF
        frames = read_frames(self.box(good, bytes(bad), frame('c', v_str('three'))))
        self.assertEqual([f['key'] for f in frames], ['a', 'b'])
        self.assertFalse(frames[-1]['crc_ok'])

    def test_a_frame_that_failed_its_crc_is_not_live_data(self):
        # It is reported, so an examiner sees it, but its value is unverified
        # and must not be published as what the app would read.
        corrupt = bytearray(frame('b', v_str('two')))
        corrupt[-1] ^= 1
        frames = read_frames(self.box(frame('a', v_str('one')), bytes(corrupt)))
        self.assertEqual([f['key'] for f in frames], ['a', 'b'])
        self.assertFalse(frames[1]['crc_ok'])
        self.assertEqual(live_entries(frames), {'a': 'one'})

    def test_an_empty_box_reads_as_empty_not_as_an_error(self):
        p = self.box(b'')
        self.assertEqual(read_frames(p), [])
        self.assertEqual(hive_parser.hive_entries(p), ({}, {}))


# --- append-only history ----------------------------------------------------

class TestHistory(HiveTest):

    def test_read_frames_keeps_superseded_values(self):
        p = self.box(frame('k', v_str('first')), frame('k', v_str('second')),
                     frame('k', v_str('third')))
        self.assertEqual([f['value'] for f in read_frames(p)],
                         ['first', 'second', 'third'])

    def test_live_entries_collapses_to_the_last_write(self):
        p = self.box(frame('k', v_str('first')), frame('k', v_str('second')))
        self.assertEqual(live_entries(read_frames(p)), {'k': 'second'})

    def test_superseded_counts_reports_the_replaced_frames(self):
        p = self.box(frame('k', v_str('a')), frame('k', v_str('b')),
                     frame('k', v_str('c')), frame('j', v_str('x')))
        self.assertEqual(superseded_counts(read_frames(p)), {'k': 2})

    def test_a_deletion_is_a_frame_with_no_value(self):
        frames = read_frames(self.box(frame('k', v_str('live')), frame('k')))
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[1]['deleted'])
        self.assertIsNone(frames[1]['error'])
        self.assertEqual(live_entries(frames), {})

    def test_a_deletion_is_not_a_stored_null(self):
        frames = read_frames(self.box(frame('d'), frame('n', v_null())))
        deleted = {f['key']: f['deleted'] for f in frames}
        self.assertTrue(deleted['d'])
        self.assertFalse(deleted['n'])
        # the stored null survives into the live view, the deletion does not
        self.assertEqual(live_entries(frames), {'n': None})

    def test_a_key_rewritten_after_deletion_is_live_again(self):
        p = self.box(frame('k', v_str('one')), frame('k'), frame('k', v_str('two')))
        self.assertEqual(live_entries(read_frames(p)), {'k': 'two'})


# --- containers -------------------------------------------------------------

class TestBuiltInTypes(HiveTest):
    """One frame per built-in type id, so none of them is only assumed."""

    def test_every_built_in_type_round_trips(self):
        cases = [
            ('null', v_null(), None),
            ('int', v_int(-5), -5),
            ('double', v_double(1.5), 1.5),
            ('bool', v_bool(False), False),
            ('string', v_str('hi é \U0001f41d'), 'hi é \U0001f41d'),
            ('byteList', v_bytes(b'\x00\x01\xfe'), b'\x00\x01\xfe'),
            ('intList', v_intlist([1, 2, 3]), [1, 2, 3]),
            ('doubleList', bytes([7]) + struct.pack('<I', 2)
             + struct.pack('<d', 1.5) + struct.pack('<d', 2.5), [1.5, 2.5]),
            ('boolList', bytes([8]) + struct.pack('<I', 3) + bytes([1, 0, 1]),
             [True, False, True]),
            ('stringList', v_strlist(['a', 'b']), ['a', 'b']),
            ('list', v_list([v_int(1), v_str('x')]), [1, 'x']),
        ]
        for name, encoded, expected in cases:
            with self.subTest(name):
                frames = read_frames(self.box(frame('k', encoded)))
                self.assertIsNone(frames[0]['error'], name)
                self.assertEqual(frames[0]['value'], expected, name)

    def test_an_empty_container_reads_as_empty(self):
        for name, encoded in (('list', v_list([])),
                              ('stringList', v_strlist([])),
                              ('map', v_map([]))):
            with self.subTest(name):
                frames = read_frames(self.box(frame('k', encoded)))
                self.assertIsNone(frames[0]['error'])
                self.assertFalse(frames[0]['value'])


class TestContainers(HiveTest):

    def test_a_map_is_count_pairs_not_count_values(self):
        value = v_map([(v_str('a'), v_int(1)),
                       (v_str('b'), v_bool(True)),
                       (v_str('c'), v_null())])
        live, _ = hive_parser.hive_entries(self.box(frame('m', value)))
        self.assertEqual(live['m'], {'a': 1, 'b': True, 'c': None})

    def test_a_map_nested_in_a_list_reads_whole(self):
        inner = v_map([(v_str('t'), v_bool(True)), (v_str('f'), v_bool(False))])
        value = v_list([v_str('List'), v_int(5), inner])
        live, _ = hive_parser.hive_entries(self.box(frame('l', value)))
        self.assertEqual(live['l'], ['List', 5, {'t': True, 'f': False}])

    def test_a_map_that_consumed_too_little_would_leave_bytes_unread(self):
        # The guard that catches a half-read container.
        frames = read_frames(self.box(frame(
            'm', v_map([(v_str('a'), v_int(1))]) + b'\x00')))
        self.assertIsNotNone(frames[0]['error'])
        self.assertIn('unread', frames[0]['error'])

    def test_hive_list_carries_a_box_name_and_keys(self):
        value = v_hivelist('other_box', ['k1', 'k2'])
        live, _ = hive_parser.hive_entries(self.box(frame('ref', value)))
        ref = live['ref']
        self.assertIsInstance(ref, HiveListRef)
        self.assertEqual(ref.box_name, 'other_box')
        self.assertEqual(list(ref), ['k1', 'k2'])


# --- adapters ---------------------------------------------------------------

class TestAdapters(HiveTest):

    def test_type_18_carries_an_explicit_utc_flag(self):
        live, _ = hive_parser.hive_entries(
            self.box(frame('when', v_dt18(1600000000000, True))))
        when = live['when']
        self.assertIsInstance(when, HiveDateTime)
        self.assertEqual(int(when), 1600000000000)
        self.assertTrue(when.zone_recorded)
        self.assertIs(when.is_utc, True)

    def test_type_16_records_no_zone_at_all(self):
        live, _ = hive_parser.hive_entries(
            self.box(frame('when', v_dt16(1600000000000))))
        when = live['when']
        self.assertEqual(int(when), 1600000000000)
        self.assertFalse(when.zone_recorded)
        self.assertIsNone(when.is_utc)

    def test_a_datetime_is_still_an_int_for_arithmetic(self):
        live, _ = hive_parser.hive_entries(
            self.box(frame('when', v_dt18(1600000000000, False))))
        self.assertEqual(int(live['when']) // 1000, 1600000000)

    def test_type_17_is_an_arbitrary_precision_integer(self):
        big = 1234567890123456789012345678901234567890
        live, _ = hive_parser.hive_entries(self.box(frame('b', v_bigint(big))))
        self.assertEqual(live['b'], big)

    def test_type_13_is_an_int_set_not_a_datetime(self):
        live, _ = hive_parser.hive_entries(
            self.box(frame('s', v_intlist([3, 1, 2], tid=13))))
        self.assertIsInstance(live['s'], HiveSet)
        self.assertEqual(list(live['s']), [3, 1, 2])   # stored order kept

    def test_the_other_hive_ce_set_types_read(self):
        p = self.box(frame('d', v_intlist([1.5, 2.5], tid=14)),
                     frame('s', v_strlist(['a', 'b'], tid=15)),
                     frame('g', v_list([v_int(1), v_str('x')], tid=19)))
        live, _ = hive_parser.hive_entries(p)
        self.assertIsInstance(live['d'], HiveSet)
        self.assertIsInstance(live['s'], HiveSet)
        self.assertEqual(list(live['s']), ['a', 'b'])
        self.assertEqual(list(live['g']), [1, 'x'])


# --- developer types --------------------------------------------------------

class TestDeveloperTypes(HiveTest):

    def test_declared_ids_are_offset_by_32_on_disk(self):
        # @HiveType(typeId: 2) in the app's Dart source is written as 34.
        live, _ = hive_parser.hive_entries(self.box(frame(
            'r', v_obj(34, [(0, v_str('name')), (1, v_int(7))]))))
        obj = live['r']
        self.assertEqual(obj.type_id, 34)
        self.assertEqual(obj.declared_id, 2)
        self.assertEqual(obj[0], 'name')
        self.assertEqual(obj[1], 7)

    def test_a_developer_type_is_still_a_dict(self):
        # Callers tell a record from a scalar with isinstance(value, dict).
        live, _ = hive_parser.hive_entries(
            self.box(frame('r', v_obj(32, [(0, v_bool(True))]))))
        self.assertIsInstance(live['r'], dict)
        self.assertIsInstance(live['r'], HiveObject)
        self.assertEqual(live['r'].get(0), True)

    def test_a_developer_type_with_no_fields(self):
        live, _ = hive_parser.hive_entries(self.box(frame('r', v_obj(34, []))))
        self.assertEqual(dict(live['r']), {})
        self.assertEqual(live['r'].type_id, 34)

    def test_a_hand_written_adapter_falls_back_to_bytes(self):
        # A hand-written adapter calls the raw writers, so the payload carries
        # no field count, no indexes and no type bytes. Walking it anyway yields
        # plausible nonsense, so it must come back as bytes instead.
        payload = bytes([250]) + b'\xff' * 3        # a count nothing can satisfy
        live, _ = hive_parser.hive_entries(
            self.box(frame('r', bytes([36]) + payload)))
        raw = live['r']
        self.assertIsInstance(raw, hive_parser.HiveRaw)
        self.assertEqual(raw.type_id, 36)
        self.assertEqual(raw.payload, payload)

    def test_a_raw_payload_is_not_a_dict(self):
        # Callers tell a record from everything else with isinstance(v, dict),
        # so an undecoded blob must not pass that test.
        live, _ = hive_parser.hive_entries(
            self.box(frame('r', bytes([36]) + bytes([250]) + b'\xff' * 3)))
        self.assertNotIsInstance(live['r'], dict)

    def test_type_200_is_a_developer_type_not_a_colour(self):
        # registerAdapter adds 32 for any non-internal adapter and ids up to 223
        # are legal, so disk 200 is @HiveType(typeId: 168), not a Flutter Color.
        live, _ = hive_parser.hive_entries(
            self.box(frame('c', v_obj(200, [(0, v_int(42))]))))
        self.assertEqual(live['c'].type_id, 200)
        self.assertEqual(live['c'].declared_id, 168)
        self.assertEqual(live['c'][0], 42)

    def test_nested_developer_types(self):
        inner = v_obj(33, [(0, v_double(18.4)), (1, v_double(-66.1))])
        live, _ = hive_parser.hive_entries(self.box(frame(
            'p', v_obj(35, [(0, v_str('place')), (1, inner)]))))
        self.assertEqual(live['p'][0], 'place')
        self.assertEqual(live['p'][1][0], 18.4)


# --- damage -----------------------------------------------------------------

class TestDamage(HiveTest):

    def test_one_unreadable_value_costs_only_its_own_frame(self):
        bad = frame('b', b'\x11' + bytes([200]))     # BigInt, length past the end
        frames = read_frames(self.box(frame('a', v_str('one')), bad,
                                      frame('c', v_str('three'))))
        self.assertEqual([f['key'] for f in frames], ['a', 'b', 'c'])
        self.assertIsNotNone(frames[1]['error'])
        self.assertIsNone(frames[0]['error'])
        self.assertIsNone(frames[2]['error'])
        self.assertEqual(sorted(live_entries(frames)), ['a', 'c'])

    def test_a_truncated_final_frame_still_names_its_key(self):
        whole = frame('b', v_str('two'))
        frames = read_frames(self.box(frame('a', v_str('one')),
                                      whole[:len(whole) - 6]))
        truncated = [f for f in frames if f['truncated']]
        self.assertEqual(len(truncated), 1)
        self.assertEqual(truncated[0]['key'], 'b')
        self.assertFalse(truncated[0]['crc_ok'])

    def test_carving_is_off_by_default_and_opt_in(self):
        corrupt = bytearray(frame('a', v_str('one')))
        struct.pack_into('<I', corrupt, 0, 0)        # destroys frame one
        p = self.box(bytes(corrupt), frame('b', v_str('two')),
                     frame('c', v_str('three')))
        self.assertEqual(read_frames(p), [])
        carved = [f['key'] for f in read_frames(p, carve=True) if f['crc_ok']]
        self.assertEqual(carved, ['b', 'c'])

    def test_carving_finds_frames_in_a_fragment(self):
        p = self.box(b'\xAA' * 40, frame('a', v_str('one')),
                     frame('b', v_str('two')))
        keys = [f['key'] for f in read_frames(p, carve=True) if f['crc_ok']]
        self.assertEqual(keys, ['a', 'b'])

    def test_a_corrupt_frame_length_does_not_allocate_the_file(self):
        corrupt = bytearray(frame('a', v_str('one')))
        struct.pack_into('<I', corrupt, 0, 0xFFFFFFF)
        self.assertEqual(read_frames(self.box(bytes(corrupt))), [])


# --- encryption -------------------------------------------------------------

class TestEncryptedShape(HiveTest):

    @staticmethod
    def encrypted_box_bytes(keys):
        """Frames shaped like an encrypted box: plaintext keys, ciphertext
        values, and a checksum seeded with a CRC of the key so none matches."""
        out = []
        for i, key in enumerate(keys):
            body = (b'\x01' + bytes([len(key)]) + key.encode()
                    + bytes((i * 7 + n) % 256 for n in range(32)))
            head = struct.pack('<I', 4 + len(body) + 4) + body
            out.append(head + struct.pack('<I', 0xDEADBEEF))
        return b''.join(out)

    def test_an_encrypted_box_still_yields_its_whole_key_namespace(self):
        keys = ['access_token', 'account', 'issued']
        read = read_frames(self.box(self.encrypted_box_bytes(keys)))
        self.assertTrue(hive_parser.looks_encrypted(read))
        # the count is the point: stopping at the first bad CRC loses the rest
        self.assertEqual(len(read), len(keys))
        self.assertEqual([f['key'] for f in read], keys)

    def test_an_encrypted_box_invents_no_type_or_value(self):
        read = read_frames(self.box(self.encrypted_box_bytes(['a', 'b'])))
        self.assertTrue(all(f['value'] is None for f in read))
        self.assertTrue(all(f['type_id'] is None for f in read))
        self.assertTrue(all(f['encrypted'] for f in read))

    def test_one_corrupt_frame_does_not_read_as_encrypted(self):
        corrupt = bytearray(frame('a', v_str('one')))
        corrupt[-1] ^= 0xFF
        read = read_frames(self.box(bytes(corrupt), frame('b', v_str('two'))))
        self.assertFalse(hive_parser.looks_encrypted(read))

    def test_a_chain_that_does_not_reach_the_end_is_not_encrypted(self):
        blob = self.encrypted_box_bytes(['a', 'b'])
        read = read_frames(self.box(blob + b'\x00\x00\x00\x00trailing'))
        self.assertFalse(hive_parser.looks_encrypted(read))

    def test_a_clean_box_does_not_read_as_encrypted(self):
        read = read_frames(self.box(frame('a', v_str('one'))))
        self.assertFalse(hive_parser.looks_encrypted(read))


if __name__ == '__main__':
    unittest.main()
