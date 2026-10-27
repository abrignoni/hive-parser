"""Command line for hive-parser.

    python -m hive_parser dump BOX.hive          every frame in file order
    python -m hive_parser dump --live BOX.hive   only what the app reads back
    python -m hive_parser dump --json BOX.hive   machine readable
    python -m hive_parser hex BOX.hive           annotated byte walkthrough
"""

import argparse
import json
import os
import struct
import sys

from . import (HiveDateTime, HiveListRef, HiveObject, HiveSet, KEY_UTF8,
               __version__, live_entries, looks_encrypted, read_frames,
               superseded_counts, type_name)


def _short(value, width=58):
    if isinstance(value, (bytes, bytearray)):
        text = f'<{len(value)} bytes> ' + value[:10].hex(' ')
    elif isinstance(value, str):
        text = json.dumps(value)
    else:
        text = repr(value)
    text = text.replace('\n', '\\n')
    return text if len(text) <= width else text[:width - 1] + '…'


def _encode(value):
    if isinstance(value, HiveDateTime):
        return {'_type': 'DateTime', 'ms': int(value), 'is_utc': value.is_utc,
                'zone_recorded': value.zone_recorded}
    if isinstance(value, HiveObject):
        return {'_type': 'object', 'on_disk_type_id': value.type_id,
                'app_declared_type_id': value.declared_id,
                'fields': {str(k): _encode(v) for k, v in sorted(value.items())}}
    if isinstance(value, HiveListRef):
        return {'_type': 'hiveList', 'box': value.box_name,
                'keys': [_encode(v) for v in value]}
    if isinstance(value, HiveSet):
        return {'_type': 'set', 'items': [_encode(v) for v in value]}
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return {'_type': 'bytes', 'hex': value.hex()}
    return value


def _targets(path):
    if os.path.isdir(path):
        found = sorted(os.path.join(r, f) for r, _, fs in os.walk(path)
                       for f in fs if f.endswith('.hive'))
        if not found:
            sys.exit(f'no .hive files under {path}')
        return found
    return [path]


def _summary(path, frames, out):
    live = live_entries(frames)
    superseded = sum(superseded_counts(frames).values())
    deleted = sum(1 for f in frames if f['deleted'])
    bad = sum(1 for f in frames if not f['crc_ok'])
    errs = sum(1 for f in frames if f['error'])
    print(f'File        {path}', file=out)
    print(f'Frames      {len(frames)} on disk', file=out)
    print(f'Live keys   {len(live)}   (what the app itself would read back)', file=out)
    if superseded:
        print(f'Superseded  {superseded} frame(s) hold values that were later '
              f'overwritten', file=out)
    print(f'Deletions   {deleted}', file=out)
    print(f'CRC         {len(frames) - bad} valid, {bad} failed', file=out)
    if errs:
        print(f'Unreadable  {errs} value(s); the rest of the box still reads',
              file=out)
    if looks_encrypted(frames):
        print(file=out)
        print('  NOTE  every checksum failed while the keys read cleanly. Hive '
              'seeds the', file=out)
        print('        frame CRC with a CRC of the encryption key, so this is '
              'the signature', file=out)
        print('        of an encrypted box rather than a corrupt one. Only '
              'values are', file=out)
        print('        encrypted. Apps often keep the key in shared_prefs, '
              'NSUserDefaults', file=out)
        print('        or a config file beside the box.', file=out)
    print(file=out)


def cmd_dump(args, out=None):
    out = sys.stdout if out is None else out
    blobs = []
    for path in _targets(args.path):
        frames = read_frames(path, carve=args.carve)
        if args.json:
            blobs.append({
                'file': path,
                'frames_on_disk': len(frames),
                'live_keys': len(live_entries(frames)),
                'looks_encrypted': looks_encrypted(frames),
                'frames': [{
                    'index': i, 'offset': f['offset'], 'length': f['length'],
                    'key': f['key'], 'type_id': f['type_id'],
                    'type': type_name(f['type_id']), 'deleted': f['deleted'],
                    'crc_ok': f['crc_ok'], 'carved': f['carved'],
                    'truncated': f['truncated'], 'error': f['error'],
                    'value': _encode(f['value']),
                } for i, f in enumerate(frames)],
                'live': {str(k): _encode(v)
                         for k, v in live_entries(frames).items()},
            })
            continue
        _summary(path, frames, out)
        if args.live:
            print(f'{"key":<30} {"type":<22} value', file=out)
            print('-' * 100, file=out)
            for key, value in live_entries(frames).items():
                tid = next((f['type_id'] for f in reversed(frames)
                            if f['key'] == key and not f['deleted']), None)
                print(f'{str(key)[:29]:<30} {type_name(tid)[:21]:<22} '
                      f'{_short(value)}', file=out)
        else:
            print(f'{"#":>4} {"offset":>9} {"len":>6}  {"key":<26} '
                  f'{"type":<22} value', file=out)
            print('-' * 108, file=out)
            for i, f in enumerate(frames):
                flags = ''.join((
                    'C' if f['carved'] else '',
                    'T' if f['truncated'] else '',
                    '!' if not f['crc_ok'] else '',
                    'E' if f['error'] else ''))
                shown = ('deleted' if f['deleted']
                         else f['error'] or _short(f['value']))
                print(f'{i:>4} {f["offset"]:>9} {f["length"]:>6}{flags:<3}'
                      f'{str(f["key"])[:25]:<26} '
                      f'{type_name(f["type_id"])[:21]:<22} {shown}', file=out)
            if any(f['carved'] or f['truncated'] or not f['crc_ok'] or f['error']
                   for f in frames):
                print(file=out)
                print('  flags: C carved   T truncated at end of file   '
                      '! CRC failed   E value unreadable', file=out)
        print(file=out)
    if args.json:
        json.dump(blobs if len(blobs) > 1 else blobs[0], out, indent=1)
        print(file=out)
    return 0


def cmd_hex(args, out=None):
    out = sys.stdout if out is None else out
    path = _targets(args.path)[0]
    with open(path, 'rb') as handle:
        data = handle.read()
    frames = read_frames(path, carve=args.carve)
    if not frames:
        print(f'{path}: no frames', file=out)
        return 1
    frame = frames[min(args.frame, len(frames) - 1)]
    end = min(frame['offset'] + frame['length'], len(data))
    blob = data[frame['offset']:end]
    print(f'File  {path}', file=out)
    print(f'frame {args.frame} at offset {frame["offset"]}, '
          f'{frame["length"]} bytes declared, {len(blob)} present', file=out)
    print(file=out)
    for i in range(0, len(blob), 16):
        chunk = blob[i:i + 16]
        text = ''.join(chr(c) if 32 <= c < 127 else '.' for c in chunk)
        print(f'  {frame["offset"] + i:08x}  {chunk.hex(" "):<47}  |{text}|',
              file=out)
    print(file=out)
    pos = 0

    def row(count, label):
        nonlocal pos
        print(f'  {pos:>4}  {blob[pos:pos + count].hex(" "):<26}  {label}',
              file=out)
        pos += count

    row(4, f'frame length = {frame["length"]}')
    key_type = blob[pos]
    row(1, f'key type {key_type} = '
           f'{"UTF-8 string" if key_type == KEY_UTF8 else "uint32"}')
    if key_type == KEY_UTF8:
        key_len = blob[pos]
        row(1, f'key length = {key_len}')
        row(key_len, f'key = {json.dumps(frame["key"])}')
    else:
        row(4, f'key = {frame["key"]}')
    if frame['deleted']:
        print('        (no value follows: this frame is a deletion)', file=out)
    elif pos < len(blob) - 4:
        tid = blob[pos]
        row(1, f'value type {tid} = {type_name(tid)}')
        if tid in (4, 5, 9, 15) and pos + 4 <= len(blob) - 4:
            count = struct.unpack_from('<I', blob, pos)[0]
            row(4, f'count = {count}'
                   f'{" bytes of UTF-8" if tid == 4 else ""}')
        elif tid in (1, 2, 16, 18) and pos + 8 <= len(blob) - 4:
            number = struct.unpack_from('<d', blob, pos)[0]
            row(8, f'float64 = {number if tid == 2 else int(number)}'
                   f'{" ms since epoch" if tid in (16, 18) else ""}')
            if tid == 18 and pos < len(blob) - 4:
                row(1, f'is_utc = {bool(blob[pos])}')
        if pos < len(blob) - 4:
            left = len(blob) - 4 - pos
            print(f'  {pos:>4}  {blob[pos:pos + min(left, 8)].hex(" ") + (" ..." if left > 8 else ""):<26}  '
                  f'value payload ({left} bytes)', file=out)
            pos = len(blob) - 4
    row(4, 'CRC32 ' + ('verified' if frame['crc_ok'] else 'MISMATCH'))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='hive_parser',
        description='Read Flutter/Hive boxes, showing history rather than '
                    'hiding it.')
    parser.add_argument('--version', action='version',
                        version=f'hive-parser {__version__}')
    subs = parser.add_subparsers(dest='command', required=True)

    dump = subs.add_parser('dump', help='summary and frames')
    dump.add_argument('path', help='a .hive file, or a directory to search')
    dump.add_argument('--live', action='store_true',
                      help='only what the app itself would read back')
    dump.add_argument('--json', action='store_true', help='machine readable')
    dump.add_argument('--carve', action='store_true',
                      help='scan for CRC-valid frames past a damaged region')
    dump.set_defaults(func=cmd_dump)

    hexc = subs.add_parser('hex', help='annotated byte walkthrough of a frame')
    hexc.add_argument('path')
    hexc.add_argument('--frame', type=int, default=0)
    hexc.add_argument('--carve', action='store_true')
    hexc.set_defaults(func=cmd_hex)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
