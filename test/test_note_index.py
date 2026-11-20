"""Tests for the lazy core-dump note index (NoteIndex/ThreadNoteGroup)."""
from __future__ import annotations

import os
import struct
import unittest
from io import BytesIO

from elftools.common.exceptions import ELFNoteDecodeError, ELFParseError
from elftools.elf.elffile import ELFFile
from elftools.elf.notes import (
    NoteSourceKind,
    NoteTruncation,
)
from elftools.elf.sections import NoteSection
from elftools.elf.segments import NoteSegment

TESTFILES = os.path.join('test', 'testfiles_for_unittests')


#-------------------------------------------------------------------------------
# Synthetic core dump builder
#-------------------------------------------------------------------------------

PT_NOTE = 4
ET_CORE = 4
EM_X86_64 = 62

NT_PRSTATUS = 1
NT_FPREGSET = 2
NT_PRPSINFO = 3
NT_AUXV = 6

CORE = b'CORE\x00'


def make_note(name: bytes, ntype: int, desc: bytes) -> bytes:
    """Build one note, padded the way it appears on disk."""
    name_pad = (-len(name)) % 4
    desc_pad = (-len(desc)) % 4
    return (struct.pack('<III', len(name), len(desc), ntype)
            + name + b'\x00' * name_pad
            + desc + b'\x00' * desc_pad)


def prstatus(pid: int, descsz: int = 336) -> bytes:
    """An elf_prstatus descriptor with pr_pid at the ELF64 offset."""
    desc = bytearray(descsz)
    struct.pack_into('<i', desc, 32, pid)
    return bytes(desc)


def _pack_core(segments: list[list[bytes]]) -> bytes:
    """Build an ELF64 ET_CORE file with one PT_NOTE segment per note list."""
    ehdr_size = 64
    phdr_size = 56
    phnum = len(segments)
    offsets = []
    cursor = ehdr_size + phnum * phdr_size
    payloads = [b''.join(notes) for notes in segments]
    for payload in payloads:
        cursor = (cursor + 3) & ~3
        offsets.append(cursor)
        cursor += len(payload)

    e_ident = b'\x7fELF' + bytes([2, 1, 1]) + b'\x00' * 9
    ehdr = e_ident + struct.pack(
        '<HHIQQQIHHHHHH', ET_CORE, EM_X86_64, 1, 0, ehdr_size, 0, 0,
        ehdr_size, phdr_size, phnum, 0, 0, 0)
    phdrs = b''.join(
        struct.pack('<IIQQQQQQ', PT_NOTE, 4, off, 0, 0,
                    len(payload), len(payload), 4)
        for off, payload in zip(offsets, payloads))

    out = bytearray(ehdr + phdrs)
    for offset, payload in zip(offsets, payloads):
        out += b'\x00' * (offset - len(out))
        out += payload
    return bytes(out)


class CountingStream(BytesIO):
    """BytesIO that records how many bytes were read by the parser."""
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.bytes_read = 0
        self.max_read = 0

    def read(self, size: int = -1) -> bytes:
        data = super().read(size)
        self.bytes_read += len(data)
        self.max_read = max(self.max_read, len(data))
        return data


#-------------------------------------------------------------------------------
# Existing files: regression and stability
#-------------------------------------------------------------------------------

class TestNoteIndexOnCoreFiles(unittest.TestCase):
    CORE_FILES = ('core_linux64.elf', 'core_linux32.elf',
                  'core_linux32_qemu_mips.elf')

    def _open(self, name):
        return ELFFile.load_from_path(os.path.join(TESTFILES, name))

    def test_index_matches_legacy_per_segment_enumeration(self):
        """Requirement: single-segment count and order are unchanged."""
        for name in self.CORE_FILES:
            with self.subTest(name=name), self._open(name) as elf:
                legacy = []
                for segment in elf.iter_segments():
                    if isinstance(segment, NoteSegment):
                        legacy.extend(segment.iter_notes())
                records = list(elf.note_index().iter_records())
                self.assertEqual(len(records), len(legacy))
                for record, note in zip(records, legacy):
                    decoded = record.read()
                    self.assertEqual(decoded['n_name'], note['n_name'])
                    self.assertEqual(decoded['n_type'], note['n_type'])
                    self.assertEqual(decoded['n_descdata'], note['n_descdata'])
                    self.assertEqual(decoded['n_offset'], note['n_offset'])
                    self.assertEqual(decoded['n_size'], note['n_size'])
                    self.assertEqual(decoded['n_namesz'], note['n_namesz'])
                    self.assertEqual(decoded['n_descsz'], note['n_descsz'])
                    self.assertEqual(decoded['n_sequence'], len(
                        [x for x in records if x.sequence <= record.sequence]) - 1)

    def test_provenance_is_global_and_stable(self):
        for name in self.CORE_FILES:
            with self.subTest(name=name):
                first = self._snapshot(name)
                second = self._snapshot(name)
                self.assertEqual(first, second)
                sequences = [s[0] for s in first]
                self.assertEqual(sequences, list(range(len(first))))

    def _snapshot(self, name):
        with self._open(name) as elf:
            return [
                (r.sequence, r.source.kind.value, r.source.index,
                 r.index_in_source, r.offset, r.name, r.n_type)
                for r in elf.note_index().iter_records()]

    def test_process_note_fields_unchanged(self):
        """Process-level notes keep the exact legacy parsing result."""
        with self._open('core_linux64.elf') as elf:
            legacy = {}
            for segment in elf.iter_segments():
                if isinstance(segment, NoteSegment):
                    for note in segment.iter_notes():
                        legacy[note['n_offset']] = note
            for record in elf.note_index().iter_records():
                note = legacy[record.offset]
                decoded = record.read()
                self.assertEqual(repr(decoded['n_desc']),
                                 repr(note['n_desc']))
                for key in ('n_name', 'n_type', 'n_descdata', 'n_offset',
                            'n_size', 'n_namesz', 'n_descsz'):
                    self.assertEqual(decoded[key], note[key], key)

    def test_regression_note_sections_still_enumerable(self):
        """Requirement: existing per-section enumeration keeps working."""
        with ELFFile.load_from_path(
                os.path.join(TESTFILES, 'note_tc3xxx_blinky.elf')) as elf:
            section = next(s for s in elf.iter_sections()
                           if isinstance(s, NoteSection))
            self.assertEqual(len(list(section.iter_notes())), 522)

    def test_regression_section_and_segment_do_not_duplicate(self):
        # This file has one .note section and a PT_NOTE segment covering it.
        with ELFFile.load_from_path(
                os.path.join(TESTFILES,
                             'note_after_gnu_property', 'main.elf')) as elf:
            index = elf.note_index()
            self.assertEqual([s.kind for s in index.sources],
                             [NoteSourceKind.SECTION])
            self.assertEqual(len(list(index.iter_records())), 2)
            legacy = next(s for s in elf.iter_sections()
                          if isinstance(s, NoteSection)).iter_notes()
            self.assertEqual([n.n_type for n in legacy],
                             [r.n_type for r in index.iter_records()])


#-------------------------------------------------------------------------------
# Multi-segment, multi-thread synthetic dump
#-------------------------------------------------------------------------------

def multi_thread_dump() -> bytes:
    segment0 = [
        make_note(CORE, NT_PRSTATUS, prstatus(101)),
        make_note(CORE, NT_FPREGSET, b'fpregs-101'),
        make_note(b'VENDOR\x00', 0xdead, b'\x01\x02\x03\x04'),
        make_note(CORE, NT_AUXV, b'auxv-process'),
        make_note(CORE, NT_PRSTATUS, prstatus(102)),
        make_note(CORE, NT_FPREGSET, b'fpregs-102'),
    ]
    segment1 = [
        make_note(CORE, NT_PRSTATUS, prstatus(103)),
        make_note(CORE, NT_FPREGSET, b'fpregs-103'),
    ]
    return _pack_core([segment0, segment1])


class TestMultiSegmentNotes(unittest.TestCase):
    def test_global_order_and_provenance(self):
        with ELFFile(BytesIO(multi_thread_dump())) as elf:
            index = elf.note_index()
            self.assertEqual(len(index.sources), 2)
            self.assertEqual([s.index for s in index.sources], [0, 1])
            self.assertEqual([s.type for s in index.sources],
                             ['PT_NOTE', 'PT_NOTE'])
            records = list(index.iter_records())
            self.assertEqual(len(records), 8)
            self.assertEqual([r.sequence for r in records], list(range(8)))
            self.assertEqual([r.source.index for r in records],
                             [0, 0, 0, 0, 0, 0, 1, 1])
            self.assertEqual([r.index_in_source for r in records],
                             [0, 1, 2, 3, 4, 5, 0, 1])
            # Later segment follows every note of the earlier one.
            self.assertTrue(all(r.source.index == 0 for r in records[:6]))
            self.assertEqual(index.truncations, ())

    def test_thread_groups_match_thread_count(self):
        with ELFFile(BytesIO(multi_thread_dump())) as elf:
            groups = elf.note_index().thread_groups()
            self.assertEqual(len(groups), 3)
            self.assertEqual([g.pid for g in groups], [101, 102, 103])
            for group in groups:
                self.assertTrue(group.is_complete)
                self.assertTrue(group.has_registers)
                self.assertTrue(group.register_view_available)
                self.assertIsNotNone(group.primary_register)
                self.assertEqual(group.primary_register.n_type, 'NT_FPREGSET')

    def test_process_and_thread_notes_separated(self):
        with ELFFile(BytesIO(multi_thread_dump())) as elf:
            index = elf.note_index()
            process = list(index.iter_process_records())
            self.assertEqual([r.n_type for r in process],
                             [0xdead, 'NT_AUXV'])
            thread_records = [r for r in index.iter_records()
                              if r.is_thread_note]
            self.assertEqual(len(thread_records), 6)

    def test_unknown_owner_and_type_preserved(self):
        with ELFFile(BytesIO(multi_thread_dump())) as elf:
            record = next(r for r in elf.note_index().iter_records()
                          if r.name == 'VENDOR')
            self.assertEqual(record.type_raw, 0xdead)
            self.assertEqual(record.n_type, 0xdead)
            self.assertEqual(record.source.index, 0)
            self.assertEqual(record.index_in_source, 2)
            decoded = record.read()
            self.assertEqual(decoded['n_name'], 'VENDOR')
            self.assertEqual(decoded['n_desc'], b'\x01\x02\x03\x04')
            self.assertEqual(decoded['n_source_kind'], 'segment')
            self.assertEqual(decoded['n_source_index'], 0)
            self.assertEqual(decoded['n_offset'], record.offset)

    def test_reparse_yields_identical_grouping(self):
        with ELFFile(BytesIO(multi_thread_dump())) as elf:
            first = [(g.key, [r.sequence for r in g.records])
                     for g in elf.note_index().iter_thread_groups()]
        with ELFFile(BytesIO(multi_thread_dump())) as elf:
            second = [(g.key, [r.sequence for r in g.records])
                      for g in elf.note_index().iter_thread_groups()]
        self.assertEqual(first, second)


#-------------------------------------------------------------------------------
# Thread grouping edge cases
#-------------------------------------------------------------------------------

class TestThreadGroupingEdgeCases(unittest.TestCase):
    def test_missing_status_note_keeps_registers(self):
        data = _pack_core([[
            make_note(CORE, NT_FPREGSET, b'orphan-registers'),
            make_note(CORE, NT_PRSTATUS, prstatus(200)),
            make_note(CORE, NT_FPREGSET, b'registers-200'),
        ]])
        with ELFFile(BytesIO(data)) as elf:
            groups = elf.note_index().thread_groups()
            self.assertEqual(len(groups), 2)
            orphan = groups[0]
            self.assertFalse(orphan.is_complete)
            self.assertTrue(orphan.has_registers)
            self.assertFalse(orphan.register_view_available)
            self.assertIsNone(orphan.primary_status)
            self.assertEqual(orphan.primary_register.read()['n_desc'],
                             b'orphan-registers')
            self.assertTrue(groups[1].is_complete)
            self.assertEqual(groups[1].pid, 200)

    def test_duplicate_notes_keep_primary_and_duplicates(self):
        data = _pack_core([[
            make_note(CORE, NT_PRSTATUS, prstatus(300)),
            make_note(CORE, NT_FPREGSET, b'registers-300-first'),
            make_note(CORE, NT_PRSTATUS, prstatus(300)),
            make_note(CORE, NT_FPREGSET, b'registers-300-second'),
        ]])
        with ELFFile(BytesIO(data)) as elf:
            groups = elf.note_index().thread_groups()
            self.assertEqual(len(groups), 1)
            group = groups[0]
            self.assertEqual(group.pid, 300)
            self.assertEqual(len(group.status_records), 2)
            self.assertEqual(len(group.register_records), 2)
            self.assertEqual(len(group.duplicate_status_records), 1)
            self.assertEqual(len(group.duplicate_register_records), 1)
            self.assertIsNot(group.primary_status,
                              group.duplicate_status_records[0])
            self.assertEqual(group.primary_register.read()['n_desc'],
                             b'registers-300-first')
            self.assertEqual(
                group.duplicate_register_records[0].read()['n_desc'],
                b'registers-300-second')
            # The primary record is the first one in capture order.
            self.assertLess(group.primary_register.sequence,
                            group.duplicate_register_records[0].sequence)

    def test_group_without_registers_is_complete_but_has_no_view(self):
        data = _pack_core([[
            make_note(CORE, NT_PRSTATUS, prstatus(400)),
        ]])
        with ELFFile(BytesIO(data)) as elf:
            (group,) = elf.note_index().thread_groups()
            self.assertTrue(group.is_complete)
            self.assertFalse(group.has_registers)
            self.assertFalse(group.register_view_available)
            self.assertIsNone(group.primary_register)


#-------------------------------------------------------------------------------
# Truncation
#-------------------------------------------------------------------------------

def truncation_dump() -> bytes:
    return _pack_core([[
        make_note(CORE, NT_PRSTATUS, prstatus(501)),
        make_note(CORE, NT_FPREGSET, b'registers-501'),
        make_note(CORE, NT_PRPSINFO, bytes(136)),
    ]])


class TestTruncation(unittest.TestCase):
    def _scan(self, data):
        with ELFFile(BytesIO(data)) as elf:
            index = elf.note_index()
            records = list(index.iter_records())
            return records, index.truncations

    def _note_start(self, data, index):
        with ELFFile(BytesIO(data)) as elf:
            return list(elf.note_index().iter_records())[index].offset

    def test_header_truncation(self):
        data = truncation_dump()
        second = self._note_start(data, 1)
        records, truncations = self._scan(data[:second + 6])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].n_type, 'NT_PRSTATUS')
        (info,) = truncations
        self.assertEqual(info.truncation, NoteTruncation.HEADER)
        self.assertEqual(info.offset, second)
        self.assertEqual(info.consumed, second - truncations[0].source.offset)
        self.assertLess(info.available, info.expected)
        self.assertEqual(info.note_index, 1)

    def test_name_truncation(self):
        data = truncation_dump()
        second = self._note_start(data, 1)
        # Complete header, but only part of the 8-byte on-disk owner name.
        records, truncations = self._scan(data[:second + 12 + 3])
        self.assertEqual(len(records), 1)
        (info,) = truncations
        self.assertEqual(info.truncation, NoteTruncation.NAME)
        self.assertEqual(info.offset, second)
        self.assertEqual(info.namesz, 5)
        self.assertEqual(info.descsz, len(b'registers-501'))
        self.assertEqual(info.expected, 12 + 8)

    def test_descriptor_truncation(self):
        data = truncation_dump()
        second = self._note_start(data, 1)
        # Complete header and name, but only part of the descriptor.
        records, truncations = self._scan(data[:second + 12 + 8 + 10])
        self.assertEqual(len(records), 1)
        (info,) = truncations
        self.assertEqual(info.truncation, NoteTruncation.DESCRIPTOR)
        self.assertEqual(info.offset, second)
        self.assertEqual(info.expected, 12 + 8 + 16)
        self.assertEqual(info.available, 12 + 8 + 10)

    def test_three_truncations_are_distinct(self):
        data = truncation_dump()
        second = self._note_start(data, 1)
        statuses = {
            self._scan(data[:second + 6])[1][0].truncation,
            self._scan(data[:second + 12 + 3])[1][0].truncation,
            self._scan(data[:second + 12 + 8 + 10])[1][0].truncation,
        }
        self.assertEqual(statuses, {
            NoteTruncation.HEADER, NoteTruncation.NAME,
            NoteTruncation.DESCRIPTOR})

    def test_truncation_never_raises_and_preserves_notes(self):
        data = truncation_dump()
        second = self._note_start(data, 1)
        for cut in (second + 0, second + 6, second + 12 + 3,
                    second + 12 + 8 + 10):
            with self.subTest(cut=cut - second), ELFFile(BytesIO(data[:cut])) as elf:
                    index = elf.note_index()
                    records = list(index.iter_records())
                    self.assertEqual(len(records), 1)
                    self.assertFalse(index.is_complete)
                    self.assertNotEqual(index.truncation, None)
                    # The already parsed note is fully usable.
                    self.assertEqual(records[0].read()['n_type'],
                                     'NT_PRSTATUS')

    def test_declared_descriptor_past_segment_does_not_eat_later_segment(self):
        # Segment 0's last note claims a descriptor far beyond the segment.
        segment0 = [
            make_note(CORE, NT_PRSTATUS, prstatus(600)),
            struct.pack('<III', 5, 1 << 20, NT_FPREGSET) + CORE + b'abcd',
        ]
        segment1 = [
            make_note(CORE, NT_PRSTATUS, prstatus(601)),
            make_note(CORE, NT_FPREGSET, b'registers-601'),
        ]
        with ELFFile(BytesIO(_pack_core([segment0, segment1]))) as elf:
            index = elf.note_index()
            records = list(index.iter_records())
            self.assertEqual([r.source.index for r in records], [0, 1, 1])
            self.assertEqual([r.n_type for r in records],
                             ['NT_PRSTATUS', 'NT_PRSTATUS', 'NT_FPREGSET'])
            (info,) = index.truncations
            self.assertEqual(info.truncation, NoteTruncation.DESCRIPTOR)
            self.assertEqual(info.source.index, 0)
            groups = index.thread_groups()
            self.assertEqual([g.pid for g in groups], [600, 601])
            self.assertEqual(groups[1].primary_register.read()['n_desc'],
                             b'registers-601')

    def test_truncation_observable_without_draining_iterator(self):
        data = truncation_dump()
        second = self._note_start(data, 1)
        with ELFFile(BytesIO(data[:second + 12 + 8 + 10])) as elf:
            index = elf.note_index()
            # No call to iter_records() before consulting the status.
            self.assertFalse(index.is_complete)
            self.assertEqual(index.truncation, NoteTruncation.DESCRIPTOR)
            (info,) = index.truncations
            self.assertEqual(info.truncation, NoteTruncation.DESCRIPTOR)
            self.assertEqual(info.consumed, second - info.source.offset)
            self.assertEqual(len(index.records()), 1)

    def test_trailing_segment_padding_is_not_truncation(self):
        """Padding inside a complete source is not reported as truncation."""
        with ELFFile.load_from_path(
                os.path.join(TESTFILES, 'note_with_segment_padding',
                             'main.elf')) as elf:
            index = elf.note_index()
            self.assertEqual(len(list(index.iter_records())), 1)
            self.assertEqual(index.truncations, ())
            self.assertTrue(index.is_complete)

    def test_legacy_iter_notes_still_silently_stops(self):
        """Legacy API keeps its old behavior: no exception, no status."""
        data = truncation_dump()
        second = self._note_start(data, 1)
        with ELFFile(BytesIO(data[:second + 12 + 8 + 10])) as elf:
            segment = next(s for s in elf.iter_segments()
                           if isinstance(s, NoteSegment))
            self.assertEqual(len(list(segment.iter_notes())), 1)


#-------------------------------------------------------------------------------
# Decode failures and laziness
#-------------------------------------------------------------------------------

def malformed_nt_file_dump() -> bytes:
    # NT_FILE declares a huge number of mappings in a tiny descriptor.
    bad_desc = struct.pack('<QQ', 1 << 40, 4096)
    return _pack_core([[
        make_note(CORE, 0x46494c45, bad_desc),
        make_note(CORE, NT_PRSTATUS, prstatus(700)),
    ]])


class TestDecodeFailure(unittest.TestCase):
    def test_decode_failure_isolated_to_its_note(self):
        with ELFFile(BytesIO(malformed_nt_file_dump())) as elf:
            index = elf.note_index()
            records = list(index.iter_records())
            self.assertEqual(len(records), 2)
            bad, good = records
            decoded = bad.read()
            self.assertIsNotNone(bad.decode_error)
            self.assertEqual(decoded['n_desc'], decoded['n_descdata'])
            self.assertEqual(decoded['n_descdata'], bad.raw_descriptor())
            # Later notes and groups are unaffected.
            self.assertEqual(good.read()['n_type'], 'NT_PRSTATUS')
            self.assertEqual(index.thread_groups()[0].pid, 700)
            errors = [r for r in index.iter_decode_errors()]
            self.assertEqual([r.offset for r in errors], [bad.offset])

    def test_legacy_iter_notes_still_raises_on_bad_descriptor(self):
        with ELFFile(BytesIO(malformed_nt_file_dump())) as elf:
            segment = next(s for s in elf.iter_segments()
                           if isinstance(s, NoteSegment))
            with self.assertRaises(ELFParseError):
                list(segment.iter_notes())

    def test_strict_read_raises_note_decode_error(self):
        with ELFFile(BytesIO(malformed_nt_file_dump())) as elf:
            bad = next(iter(elf.note_index().iter_records()))
            with self.assertRaises(ELFNoteDecodeError) as caught:
                bad.read(strict=True)
            self.assertIsInstance(caught.exception, ELFParseError)


class TestLaziness(unittest.TestCase):
    def test_index_does_not_read_descriptors(self):
        huge = bytes(1 << 20)
        data = _pack_core([[
            make_note(CORE, NT_PRSTATUS, prstatus(800)),
            make_note(CORE, NT_FPREGSET, huge),
            make_note(CORE, NT_AUXV, b'auxv'),
        ]])
        stream = CountingStream(data)
        with ELFFile(stream) as elf:
            stream.bytes_read = 0
            stream.max_read = 0
            index = elf.note_index()
            records = list(index.iter_records())
            self.assertEqual(len(records), 3)
            self.assertEqual(len(index.thread_groups()), 1)
            # Only headers and owner names were read, not 1 MiB of registers.
            self.assertLess(stream.bytes_read, 4096)
            self.assertLess(stream.max_read, 4096)

            # Registers are read on demand, one group at a time.
            group = index.thread_groups()[0]
            stream.bytes_read = 0
            notes = list(group.iter_register_notes())
            self.assertEqual(len(notes), 1)
            self.assertGreaterEqual(stream.bytes_read, len(huge))

    def test_records_report_descriptor_offsets(self):
        with ELFFile(BytesIO(multi_thread_dump())) as elf:
            for record in elf.note_index().iter_records():
                self.assertEqual(
                    record.descriptor_offset,
                    record.offset + 12 + record.name_size)
                self.assertEqual(record.descriptor_size, record.descsz)
                self.assertEqual(len(record.raw_descriptor()), record.descsz)


#-------------------------------------------------------------------------------
# A real kernel-generated multithreaded core (notes only, reduced for size)
#-------------------------------------------------------------------------------

class TestRealMultithreadedCore(unittest.TestCase):
    """Fixture extracted from a real 5-thread Linux core dump.

    Only the PT_NOTE segment and a minimal ELF header were kept, so the file
    is small while the note layout is exactly what the kernel produced.
    """
    PIDS = (1301631, 1301632, 1301633, 1301634, 1301635)

    def _open(self):
        return ELFFile.load_from_path(
            os.path.join(TESTFILES, 'core_multithread_notes.elf'))

    def test_thread_group_count_matches_thread_count(self):
        with self._open() as elf:
            groups = elf.note_index().thread_groups()
            self.assertEqual(len(groups), len(self.PIDS))
            self.assertEqual(tuple(g.pid for g in groups), self.PIDS)

    def test_every_group_yields_registers(self):
        with self._open() as elf:
            for group in elf.note_index().thread_groups():
                self.assertTrue(group.is_complete)
                self.assertTrue(group.register_view_available)
                status = group.primary_status.read()
                registers = list(group.iter_register_notes())
                self.assertEqual(len(registers), 1)
                self.assertEqual(len(status['n_descdata']), 336)
                self.assertEqual(len(registers[0]['n_descdata']), 512)
                # Registers belong to the status note of the same thread.
                self.assertEqual(group.primary_status.read_pid(), group.pid)

    def test_process_notes_are_separate(self):
        with self._open() as elf:
            index = elf.note_index()
            process_types = [r.n_type for r in index.iter_process_records()]
            self.assertIn('NT_PRPSINFO', process_types)
            self.assertIn('NT_AUXV', process_types)
            self.assertIn('NT_FILE', process_types)
            self.assertNotIn('NT_PRSTATUS', process_types)
            self.assertNotIn('NT_FPREGSET', process_types)

    def test_provenance_marks_index_within_source(self):
        with self._open() as elf:
            records = list(elf.note_index().iter_records())
            self.assertEqual(len(records), 20)
            for record in records:
                self.assertEqual(record.source.kind, NoteSourceKind.SEGMENT)
                self.assertEqual(record.descriptor_offset,
                                 record.offset + 12 + record.name_size)
            # The status/register pairs are adjacent in the first thread.
            first = [r for r in records if r.index_in_source in (0, 5)]
            self.assertEqual([r.n_type for r in first],
                             ['NT_PRSTATUS', 'NT_FPREGSET'])


if __name__ == '__main__':
    unittest.main()
