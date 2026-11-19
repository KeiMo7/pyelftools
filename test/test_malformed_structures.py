"""Rejection semantics for malformed and truncated ELF structures.

These tests treat the ELF input as untrusted: every rejection must be one of
the library's own exceptions, must carry structural location information and
must happen before the offending bytes are read or allocated.
"""
import io
import os
import struct
import unittest
import zlib
from unittest.mock import Mock

from elftools.common.exceptions import (
    ELFCompressionError,
    ELFError,
    ELFParseError,
    ELFUnparsableFileError,
)
from elftools.construct import Container
from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile
from elftools.elf.report import StructureRejection
from elftools.elf.sections import (
    Section,
    StabSection,
    StringTableSection,
    SymbolTableIndexSection,
    SymbolTableSection,
)
from elftools.elf.segments import Segment
from elftools.elf.structs import ELFStructs

TEST_DIR = os.path.join('test', 'testfiles_for_unittests')


def stream(data):
    return io.BytesIO(data)


def minimal_elf(elfclass=64, little=True, *, phnum=0, phoff=0, phentsize=None,
                shnum=0, shoff=0, shentsize=None, body=b'', ident_version=1,
                shstrndx=0):
    """Build a minimal, otherwise well-formed ELF header followed by |body|."""
    ident = bytearray(16)
    ident[0:4] = b'\x7fELF'
    ident[4] = 2 if elfclass == 64 else 1
    ident[5] = 1 if little else 2
    ident[6] = ident_version
    endian = '<' if little else '>'
    if phentsize is None:
        phentsize = 56 if elfclass == 64 else 32
    if shentsize is None:
        shentsize = 64 if elfclass == 64 else 40
    if elfclass == 64:
        header = struct.pack(
            endian + 'HHIQQQIHHHHHH', 2, 62, 1, 0, phoff, shoff, 0, 64,
            phentsize, phnum, shentsize, shnum, shstrndx)
    else:
        header = struct.pack(
            endian + 'HHIIIIIHHHHHH', 2, 3, 1, 0, phoff, shoff, 0, 52,
            phentsize, phnum, shentsize, shnum, shstrndx)
    return bytes(ident) + header + body


def make_structs(elfclass=64, little=True):
    structs = ELFStructs(elfclass=elfclass, little_endian=little)
    structs.create_basic_structs()
    structs.create_advanced_structs()
    return structs


def standalone_section(data, *, offset, size, kind='SHT_PROGBITS',
                       compressed=False, elfclass=64, little=True):
    structs = make_structs(elfclass, little)
    f = data if hasattr(data, 'seek') else io.BytesIO(data)
    elf = _FakeElf(f, structs)
    return Section(Container(
        sh_type=kind, sh_offset=offset, sh_size=size, sh_addralign=1,
        sh_flags=SH_FLAGS.SHF_COMPRESSED if compressed else 0), '', elf)


class _FakeElf:
    def __init__(self, stream, structs):
        self.stream = stream
        self.structs = structs
        stream.seek(0, io.SEEK_END)
        self.stream_len = stream.tell()


class TestWholeFileRejection(unittest.TestCase):
    """A file truncated in the header is unparsable as a whole."""

    def test_truncation_before_version_byte(self):
        full = minimal_elf()
        for length in (1, 2, 3, 4, 5, 6):
            with self.subTest(length=length):
                with self.assertRaises(ELFError) as caught:
                    ELFFile(stream(full[:length]))
                # 1..3 fail the magic check; 4..6 are identification
                # truncations reported by the library itself.
                if length >= 4:
                    self.assertIsInstance(caught.exception, ELFUnparsableFileError)
                    self.assertEqual(caught.exception.scope, 'file')

    def test_truncation_in_header_body(self):
        full = minimal_elf(elfclass=32)
        # 32-bit header is 52 bytes; every prefix inside it fails as a file.
        for length in range(7, len(full)):
            with self.subTest(length=length):
                with self.assertRaises(ELFUnparsableFileError) as caught:
                    ELFFile(stream(full[:length]))
                self.assertEqual(caught.exception.structure, 'ELF header')
                self.assertIsInstance(caught.exception, ELFParseError)

    def test_truncation_is_distinguishable_from_structural_rejection(self):
        with self.assertRaises(ELFUnparsableFileError) as whole:
            ELFFile(stream(minimal_elf()[:20]))
        self.assertEqual(whole.exception.scope, 'file')

        with stream(minimal_elf(shnum=1, shoff=10_000, body=b'x' * 32)) as f:
            elf = ELFFile(f)
        with self.assertRaises(ELFParseError) as single:
            elf.get_section(0)
        self.assertNotIsInstance(single.exception, ELFUnparsableFileError)
        self.assertEqual(single.exception.scope, 'structure')

    def test_invalid_identification_values(self):
        full = bytearray(minimal_elf())
        for index, value in ((4, 0x07), (5, 0x07)):
            with self.subTest(index=index):
                bad = bytearray(full)
                bad[index] = value
                with self.assertRaises(ELFUnparsableFileError):
                    ELFFile(stream(bytes(bad)))

    def test_whole_file_rejection_is_a_parse_error(self):
        # Callers can catch the base class alone.
        with self.assertRaises(ELFParseError):
            ELFFile(stream(b'\x7fELF'))


class TestSectionTableCapacity(unittest.TestCase):
    """Table-level checks happen before any entry is read."""

    def test_table_entirely_outside_file(self):
        data = minimal_elf(shnum=5, shoff=1000, body=b'')
        elf = ELFFile(stream(data))
        self.assertEqual(elf.section_table_capacity()[1], 0)
        with self.assertRaises(ELFParseError) as caught:
            elf.get_section(0)
        error = caught.exception
        self.assertEqual(error.structure, 'section header table')
        self.assertEqual(error.first_invalid_index, 0)
        self.assertEqual(error.available_count, 0)
        self.assertEqual(error.declared_count, 5)

    def test_capacity_is_checked_before_reading(self):
        data = minimal_elf(shnum=3, shoff=64 + 1000, body=b'')
        with stream(data) as raw:
            reader = Mock(wraps=raw)
            elf = ELFFile(reader)
            elf.stream = reader
            reader.read.reset_mock()
            with self.assertRaises(ELFParseError):
                elf.get_section(0)
            reader.read.assert_not_called()

    def test_declared_count_larger_than_capacity_keeps_prefix(self):
        # One whole 64-byte entry exists, but five are declared.
        body = b'\0' * 64
        data = minimal_elf(shnum=5, shoff=64, body=body)
        elf = ELFFile(stream(data))
        declared, available = elf.section_table_capacity()
        self.assertEqual((declared, available), (5, 1))

        with self.assertRaises(ELFParseError) as caught:
            elf.get_section(1)
        self.assertEqual(caught.exception.first_invalid_index, 1)

        scan = elf.iter_sections_lenient()
        self.assertEqual(len(scan), 1)
        self.assertEqual(scan.truncated_at, 1)
        # The tail beyond the capacity is reported once, at its first index,
        # rather than once per absurd declared index.
        self.assertEqual([r.index for r in scan.rejections], [1])
        self.assertTrue(all(isinstance(r.error, ELFParseError)
                            for r in scan.rejections))
        self.assertEqual(scan.rejections[0].error.first_invalid_index, 1)
        # Single-item and whole-table paths agree.
        self.assertEqual(elf.classify_section(1)[1].reason,
                         scan.rejections[0].reason)
        for beyond in (2, 3, 4, 1000):
            self.assertEqual(elf.classify_section(beyond)[1].error.index, beyond)

    def test_declared_count_equal_to_capacity(self):
        data = minimal_elf(shnum=1, shoff=64, body=b'\0' * 64)
        elf = ELFFile(stream(data))
        self.assertEqual(elf.section_table_capacity(), (1, 1))
        scan = elf.iter_sections_lenient()
        self.assertIsNone(scan.truncated_at)
        self.assertEqual(len(scan), 1)

    def test_entry_size_too_small(self):
        data = minimal_elf(shnum=1, shoff=64, shentsize=4, body=b'\0' * 64)
        elf = ELFFile(stream(data))
        with self.assertRaises(ELFParseError) as caught:
            elf.section_table_capacity()
        self.assertEqual(caught.exception.structure, 'section header table')
        with self.assertRaises(ELFParseError):
            elf.get_section(0)

    def test_extended_numbering_absurd_count(self):
        # e_shnum == 0 defers the count to entry 0's sh_size.
        entry0 = struct.pack('<IIQQQQIIQQ', 0, 0, 0, 0, 0, 2**40, 0, 0, 0, 0)
        data = minimal_elf(shnum=0, shoff=64, body=entry0)
        elf = ELFFile(stream(data))
        declared, available = elf.section_table_capacity()
        self.assertEqual(declared, 2**40)
        self.assertEqual(available, 1)
        scan = elf.iter_sections_lenient()
        self.assertEqual(len(scan), 1)
        self.assertEqual(scan.truncated_at, 1)

    def test_absurd_count_does_not_amplify(self):
        # A 16-bit declared count against a one-entry table must not produce
        # one rejection object per declared index.
        data = minimal_elf(shnum=0xffff, shoff=64, body=b'\0' * 64)
        elf = ELFFile(stream(data))
        scan = elf.iter_sections_lenient()
        self.assertEqual(len(scan), 1)
        self.assertEqual(len(scan.rejections), 1)
        self.assertEqual(scan.truncated_at, 1)

    def test_extended_numbering_entry_zero_truncated(self):
        data = minimal_elf(shnum=0, shoff=64, body=b'\0' * 10)
        elf = ELFFile(stream(data))
        with self.assertRaises(ELFParseError) as caught:
            elf.section_table_capacity()
        self.assertEqual(caught.exception.structure, 'section header table')
        self.assertEqual(caught.exception.index, 0)


class TestEntryTruncation(unittest.TestCase):
    """An entry only half inside the file is a damaged item, not the table."""

    def _elf_with_half_entry(self):
        # Header is 64 bytes; place the table so its first entry has only its
        # first 20 bytes inside the file.
        body = b'\0' * 20
        return minimal_elf(elfclass=32, shnum=2, shoff=52, body=body)

    def test_half_entry_is_a_damaged_item(self):
        data = self._elf_with_half_entry()
        elf = ELFFile(stream(data))
        _declared, available = elf.section_table_capacity()
        # The 20 remaining bytes are not a whole 40-byte entry.
        self.assertEqual(available, 0)
        with self.assertRaises(ELFParseError) as caught:
            elf.get_section(0)
        error = caught.exception
        self.assertEqual(error.index, 0)
        self.assertEqual(error.expected, 40)
        self.assertEqual(error.actual, 20)

    def test_half_last_entry_among_whole_ones(self):
        # Two whole entries plus 12 bytes of a third: the third is a damaged
        # item whose shortfall is reported in bytes.
        entry_size = 40
        body = b'\0' * (entry_size * 2 + 12)
        data = minimal_elf(elfclass=32, shnum=3, shoff=52, body=body)
        elf = ELFFile(stream(data))
        with self.assertRaises(ELFParseError) as caught:
            elf.get_section(2)
        self.assertEqual(caught.exception.index, 2)
        self.assertEqual(caught.exception.expected, entry_size)
        self.assertEqual(caught.exception.actual, 12)

    def test_other_entries_still_enumerable(self):
        # Three whole entries, then a partial fourth.
        entry_size = 40
        body = b'\0' * (entry_size * 3 + 12)
        data = minimal_elf(elfclass=32, shnum=4, shoff=52, body=body)
        elf = ELFFile(stream(data))
        declared, available = elf.section_table_capacity()
        self.assertEqual((declared, available), (4, 3))
        scan = elf.iter_sections_lenient()
        self.assertEqual(len(scan), 3)
        self.assertEqual([r.index for r in scan.rejections], [3])
        self.assertEqual(scan.truncated_at, 3)

    def test_strict_iterator_raises_at_damaged_entry(self):
        entry_size = 40
        body = b'\0' * (entry_size * 3 + 12)
        data = minimal_elf(elfclass=32, shnum=4, shoff=52, body=body)
        elf = ELFFile(stream(data))
        seen = []
        with self.assertRaises(ELFParseError):
            for section in elf.iter_sections():
                seen.append(section)  # noqa: PERF402 - observe the prefix
        self.assertEqual(len(seen), 3)


class TestSectionRangeRejection(unittest.TestCase):
    """Section offsets and sizes outside the file are rejected before reading."""

    def test_forward_out_of_bounds(self):
        section = standalone_section(b'abcdef', offset=10, size=4)
        with self.assertRaises(ELFParseError) as caught:
            section.data()
        self.assertEqual(caught.exception.expected, 4)

    def test_addition_wraparound(self):
        section = standalone_section(b'abcdef', offset=2**64 - 4, size=8)
        with self.assertRaises(ELFParseError):
            section.data()

    def test_negative_values(self):
        for offset, size in ((-1, 1), (0, -1), (-1, -1)):
            with self.subTest(offset=offset, size=size):
                section = standalone_section(b'abcdef', offset=offset, size=size)
                with self.assertRaises(ELFParseError):
                    section.data()

    def test_rejected_before_read_for_section_and_segment(self):
        for maker, kwargs in (
                (standalone_section, {'offset': 100, 'size': 4}),
                (standalone_segment, {'offset': 100, 'size': 4}),
                (standalone_segment, {'offset': 0, 'size': 2**32 - 1})):
            with self.subTest(kwargs=kwargs), stream(b'abcdef') as raw:
                reader = Mock(wraps=raw)
                obj = maker(reader, **kwargs)
                with self.assertRaises(ELFParseError):
                    obj.data()
                reader.read.assert_not_called()

    def test_negative_reads_do_not_move_stream_inside_file(self):
        with stream(b'abcdef') as raw:
            reader = Mock(wraps=raw)
            section = standalone_section(reader, offset=-1, size=1)
            with self.assertRaises(ELFParseError):
                section.data()
            sought = [c.args[0] for c in reader.seek.call_args_list
                      if len(c.args) == 1]
            self.assertTrue(all(not (0 <= s < 6) for s in sought))


def standalone_segment(stream, *, offset, size):
    return Segment(Container(p_offset=offset, p_filesz=size, p_memsz=1 << 32),
                   stream)


class TestSymbolIndexRejection(unittest.TestCase):
    """A symbol index outside the table never parses neighbouring bytes."""

    def _symtab(self, symbol_count=2, *, symbol_size=None):
        structs = make_structs()
        if symbol_size is None:
            symbol_size = structs.Elf_Sym.sizeof()
        strtab_data = b'\0name\0'
        # Lay out [strtab][symtab] in one buffer.
        strtab_offset = 0
        symtab_offset = len(strtab_data)
        data = strtab_data + b'\0' * (symbol_count * symbol_size)
        stream_obj = io.BytesIO(data)
        elf = _FakeElf(stream_obj, structs)
        strtab = StringTableSection(Container(
            sh_type='SHT_STRTAB', sh_offset=strtab_offset,
            sh_size=len(strtab_data), sh_addralign=1, sh_flags=0), '.strtab', elf)
        symtab = SymbolTableSection(Container(
            sh_type='SHT_SYMTAB', sh_offset=symtab_offset,
            sh_size=symbol_count * symbol_size, sh_addralign=1, sh_flags=0,
            sh_entsize=symbol_size, sh_link=1), '.symtab', elf, strtab)
        return symtab

    def test_index_beyond_count_rejected(self):
        symtab = self._symtab(symbol_count=2)
        for bad in (2, 3, 100, -1):
            with self.subTest(index=bad):
                with self.assertRaises(ELFParseError) as caught:
                    symtab.get_symbol(bad)
                self.assertEqual(caught.exception.structure, 'symbol table')
                self.assertEqual(caught.exception.index, bad)

    def test_index_rejected_before_reading(self):
        symtab = self._symtab(symbol_count=2)
        symtab.elffile.stream = Mock(wraps=symtab.elffile.stream)
        with self.assertRaises(ELFParseError):
            symtab.get_symbol(2)
        symtab.elffile.stream.read.assert_not_called()

    def test_valid_indices_still_work(self):
        symtab = self._symtab(symbol_count=2)
        self.assertEqual(symtab.get_symbol(0).name, '')
        self.assertEqual(symtab.get_symbol(1).name, '')

    def test_lenient_scan_matches_single_lookup(self):
        symtab = self._symtab(symbol_count=2)
        scan = symtab.iter_symbols_lenient()
        self.assertEqual(len(scan), 2)
        self.assertEqual(scan.rejections, ())


class TestStringOffsetRejection(unittest.TestCase):
    """Name lookups cannot read beyond the string table."""

    def _strtab(self, data):
        structs = make_structs()
        f = io.BytesIO(data)
        elf = _FakeElf(f, structs)
        return StringTableSection(Container(
            sh_type='SHT_STRTAB', sh_offset=0, sh_size=len(data),
            sh_addralign=1, sh_flags=0), '.strtab', elf)

    def test_offset_outside_table_rejected(self):
        table = self._strtab(b'a\0b\0')
        for offset in (4, 5, 1000):
            with self.subTest(offset=offset):
                with self.assertRaises(ELFParseError) as caught:
                    table.get_string(offset)
                self.assertEqual(caught.exception.structure, 'string table')

    def test_negative_offset_rejected(self):
        table = self._strtab(b'a\0b\0')
        with self.assertRaises(ELFParseError):
            table.get_string(-1)

    def test_missing_name_is_none_not_empty(self):
        table = self._strtab(b'a\0b\0')
        self.assertEqual(table.get_string_lenient(0), 'a')
        self.assertEqual(table.get_string_lenient(2), 'b')
        # A real empty name at an in-range offset.
        self.assertEqual(table.get_string_lenient(1), '')
        # An offset outside the table is unavailable, not empty.
        self.assertIsNone(table.get_string_lenient(1000))
        self.assertIsNone(table.get_string_lenient(-1))

    def test_unterminated_string_cannot_borrow_neighbouring_bytes(self):
        # The string table ends without a NUL; the next bytes in the stream
        # belong to something else and must not be appended to the name.
        structs = make_structs()
        f = io.BytesIO(b'ab' + b'LEAK')
        elf = _FakeElf(f, structs)
        table = StringTableSection(Container(
            sh_type='SHT_STRTAB', sh_offset=0, sh_size=2,
            sh_addralign=1, sh_flags=0), '.strtab', elf)
        with self.assertRaises(ELFParseError):
            table.get_string(0)
        self.assertIsNone(table.get_string_lenient(0))

    def test_no_bytes_echoed_in_message(self):
        table = self._strtab(b'\xde\xad\xbe\xef\0')
        with self.assertRaises(ELFParseError) as caught:
            table.get_string(50)
        message = str(caught.exception)
        self.assertNotIn('\\xde', message)
        self.assertNotIn('dead', message.lower())


class TestCompressedSectionRejection(unittest.TestCase):
    """Truncated header, size mismatch and corrupt stream all converge."""

    def _compressed(self, elfclass, payload, *, sh_size=None, ch_size=None):
        structs = make_structs(elfclass)
        header = structs.Elf_Chdr.build(Container(
            ch_type='ELFCOMPRESS_ZLIB', ch_reserved=0,
            ch_size=len(payload) if ch_size is None else ch_size,
            ch_addralign=1))
        stored = header + zlib.compress(payload)
        total = len(stored) if sh_size is None else sh_size
        return standalone_section(
            stored, offset=0, size=total, compressed=True, elfclass=elfclass)

    def test_truncated_header(self):
        # The header is read in the constructor, so the rejection happens
        # there: before data() would read anything.
        for elfclass in (32, 64):
            header_size = make_structs(elfclass).Elf_Chdr.sizeof()
            with self.subTest(elfclass=elfclass):
                with self.assertRaises(ELFCompressionError) as caught:
                    standalone_section(
                        bytes(header_size), offset=0, size=header_size - 1,
                        compressed=True, elfclass=elfclass)
                self.assertEqual(caught.exception.expected, header_size)
                self.assertEqual(caught.exception.actual, header_size - 1)

    def test_declared_size_mismatch(self):
        header = make_structs(64).Elf_Chdr.build(Container(
            ch_type='ELFCOMPRESS_ZLIB', ch_reserved=0, ch_size=99,
            ch_addralign=1))
        stored = header + zlib.compress(b'hello')
        section = standalone_section(stored, offset=0, size=len(stored),
                                     compressed=True)
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        self.assertEqual(caught.exception.expected, 99)
        self.assertEqual(caught.exception.actual, 5)

    def test_corrupt_stream(self):
        section = self._compressed(64, b'hello world')
        # Flip a byte inside the deflate stream.
        data = bytearray(section.elffile.stream.getvalue())
        data[-1] ^= 0xFF
        section.elffile.stream = io.BytesIO(bytes(data))
        section.stream = section.elffile.stream
        with self.assertRaises(ELFCompressionError):
            section.data()

    def test_truncated_stream_rejected(self):
        payload = b'hello world' * 100
        structs = make_structs(64)
        header = structs.Elf_Chdr.build(Container(
            ch_type='ELFCOMPRESS_ZLIB', ch_reserved=0,
            ch_size=len(payload), ch_addralign=1))
        stored = header + zlib.compress(payload)
        # Cut the deflate stream short; the declared uncompressed size stays.
        truncated = stored[:len(stored) // 2]
        section = standalone_section(
            truncated, offset=0, size=len(truncated), compressed=True)
        with self.assertRaises(ELFCompressionError):
            section.data()

    def test_all_three_are_one_category(self):
        for error in (ELFCompressionError,):
            self.assertTrue(issubclass(error, ELFParseError))
            self.assertTrue(issubclass(error, ELFError))

    def test_unknown_compression_type_unchanged(self):
        section = standalone_section(
            bytes(64), offset=0, size=64, compressed=True)
        # ch_type 0 is unknown; the error is still the library's own.
        with self.assertRaises(ELFCompressionError):
            section.data()


class TestUndeclaredSizeAllocation(unittest.TestCase):
    """A huge declared size is rejected before allocation."""

    def test_nobits_huge_size_not_allocated(self):
        section = standalone_section(
            b'', offset=0, size=2**64 - 1, kind='SHT_NOBITS')
        with self.assertRaises(ELFParseError):
            section.data()

    def test_nobits_over_max_limit(self):
        section = standalone_section(
            b'', offset=0, size=Section.MAX_MATERIALIZED_SIZE + 1,
            kind='SHT_NOBITS')
        with self.assertRaises(ELFParseError) as caught:
            section.data()
        self.assertEqual(caught.exception.structure, 'NOBITS section')

    def test_nobits_within_limit_still_zeros(self):
        section = standalone_section(b'', offset=0, size=16, kind='SHT_NOBITS')
        self.assertEqual(section.data(), bytes(16))

    def test_decompression_bomb_is_bounded(self):
        # A highly compressible payload whose declared size lies: the output
        # is capped, so a tiny sample cannot inflate without limit.
        payload = b'\0' * (Section.MAX_MATERIALIZED_SIZE + 1)
        with self.assertRaises(ELFCompressionError):
            self._bomb_section(payload).data()

    def _bomb_section(self, payload):
        structs = make_structs(64)
        header = structs.Elf_Chdr.build(Container(
            ch_type='ELFCOMPRESS_ZLIB', ch_reserved=0,
            ch_size=len(payload), ch_addralign=1))
        stored = header + zlib.compress(payload, 9)
        return standalone_section(stored, offset=0, size=len(stored),
                                  compressed=True)


class TestProgramHeaderParity(unittest.TestCase):
    """The program header side follows the same judgement as sections."""

    def test_table_outside_file(self):
        data = minimal_elf(phnum=4, phoff=5000, body=b'')
        elf = ELFFile(stream(data))
        self.assertEqual(elf.segment_table_capacity()[1], 0)
        with self.assertRaises(ELFParseError) as caught:
            elf.get_segment(0)
        self.assertEqual(caught.exception.structure, 'program header table')
        self.assertEqual(caught.exception.first_invalid_index, 0)

    def test_partial_prefix_is_enumerable(self):
        entry_size = 56
        body = b'\0' * (entry_size * 2 + 10)
        data = minimal_elf(phnum=3, phoff=64, body=body)
        elf = ELFFile(stream(data))
        declared, available = elf.segment_table_capacity()
        self.assertEqual((declared, available), (3, 2))
        scan = elf.iter_segments_lenient()
        self.assertEqual(len(scan), 2)
        self.assertEqual([r.index for r in scan.rejections], [2])
        self.assertEqual(scan.truncated_at, 2)

    def test_single_and_table_paths_agree(self):
        data = minimal_elf(phnum=3, phoff=64, body=b'\0' * 70)
        elf = ELFFile(stream(data))
        scan = elf.iter_segments_lenient()
        for rejection in scan.rejections:
            single = elf.classify_segment(rejection.index)[1]
            self.assertEqual(single.reason, rejection.reason)

    def test_truncated_entry_rejected_before_read(self):
        data = minimal_elf(phnum=1, phoff=64, body=b'\0' * 10)
        with stream(data) as raw:
            reader = Mock(wraps=raw)
            elf = ELFFile(reader)
            elf.stream = reader
            reader.read.reset_mock()
            with self.assertRaises(ELFParseError):
                elf.get_segment(0)
            reader.read.assert_not_called()


class TestDeterminism(unittest.TestCase):
    """Repeated lookups on one instance give the identical verdict."""

    def test_repeated_section_lookup_is_stable(self):
        data = minimal_elf(shnum=5, shoff=64, body=b'\0' * 100)
        elf = ELFFile(stream(data))
        first = [elf.section_table_capacity() for _ in range(3)]
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[1], first[2])

        verdicts = []
        for _ in range(3):
            try:
                elf.get_section(2)
            except ELFParseError as e:
                verdicts.append((type(e), str(e)))
        self.assertEqual(len(set(verdicts)), 1)

    def test_lenient_scan_matches_repeated_single(self):
        data = minimal_elf(shnum=4, shoff=64, body=b'\0' * 80)
        elf = ELFFile(stream(data))
        scan = elf.iter_sections_lenient()
        again = elf.iter_sections_lenient()
        self.assertEqual([r.reason for r in scan.rejections],
                         [r.reason for r in again.rejections])


class TestRejectionDoesNotEchoBytes(unittest.TestCase):
    """Error messages carry counts and indices, never file content."""

    def test_table_and_section_messages_have_no_content(self):
        data = minimal_elf(shnum=5, shoff=64, body=b'\xde\xad\xbe\xef' * 8)
        elf = ELFFile(stream(data))
        with self.assertRaises(ELFParseError) as caught:
            elf.get_section(0)
        message = str(caught.exception)
        self.assertNotIn('\xde', message)
        self.assertNotIn('dead', message.lower())

    def test_message_contains_location_and_counts(self):
        section = standalone_section(b'abcdef', offset=10, size=4)
        with self.assertRaises(ELFParseError) as caught:
            section.data()
        error = caught.exception
        self.assertIn('offset 10', str(error))
        self.assertEqual(error.expected, 4)
        self.assertEqual(error.actual, 0)


class TestRejectionPathDoesNotFailTwice(unittest.TestCase):
    """Building a rejection never reads the stream again."""

    def test_truncated_entry_error_built_without_reading(self):
        data = minimal_elf(elfclass=32, shnum=2, shoff=52, body=b'\0' * 20)
        with stream(data) as raw:
            reader = Mock(wraps=raw)
            elf = ELFFile(reader)
            elf.stream = reader
            reader.read.reset_mock()
            with self.assertRaises(ELFParseError) as caught:
                elf.get_section(0)
            reader.read.assert_not_called()
            # The message itself is fully formed.
            self.assertIn('section header table', str(caught.exception))


class TestPublicRejectionApi(unittest.TestCase):
    """The result types expose usable partial results to callers."""

    def test_table_scan_iterates_only_usable_entries(self):
        data = minimal_elf(shnum=4, shoff=64, body=b'\0' * 80)
        elf = ELFFile(stream(data))
        scan = elf.iter_sections_lenient()
        self.assertEqual(len(scan), len(list(scan)))
        self.assertEqual([s.header for s in scan],
                         [s.header for s in elf.iter_sections_lenient()])

    def test_rejections_expose_structure_and_reason(self):
        data = minimal_elf(shnum=4, shoff=64, body=b'\0' * 80)
        elf = ELFFile(stream(data))
        for rejection in elf.iter_sections_lenient().rejections:
            self.assertIsInstance(rejection, StructureRejection)
            self.assertEqual(rejection.structure, 'section header table')
            self.assertIsInstance(rejection.reason, str)
            self.assertIsInstance(rejection.error, ELFParseError)

    def test_compression_error_is_a_parse_error(self):
        self.assertTrue(issubclass(ELFCompressionError, ELFParseError))

    def test_scope_distinguishes_whole_file(self):
        self.assertEqual(ELFUnparsableFileError.scope.__get__(
            ELFUnparsableFileError('')), 'file')
        self.assertEqual(ELFParseError('').scope, 'structure')


class TestExtendedSegmentCount(unittest.TestCase):
    """PN_XNUM resolves the segment count through section header 0."""

    def test_pn_xnum_count_and_capacity(self):
        structs = make_structs(64)
        entry0 = structs.Elf_Shdr.build(Container(
            sh_name=0, sh_type='SHT_NULL', sh_flags=0, sh_addr=0, sh_offset=0,
            sh_size=0, sh_link=0, sh_info=3, sh_addralign=0, sh_entsize=0))
        body = b'\0' * 56 + entry0
        data = minimal_elf(phnum=0xffff, phoff=64, shnum=1, shoff=64 + 56,
                           body=body)
        elf = ELFFile(stream(data))
        self.assertEqual(elf.num_segments(), 3)
        self.assertEqual(elf.segment_table_capacity(), (3, 2))
        scan = elf.iter_segments_lenient()
        self.assertEqual(len(scan), 2)
        self.assertEqual(scan.truncated_at, 2)
        # Both num_segments() and the scalar path agree on the verdict.
        with self.assertRaises(ELFParseError) as caught:
            elf.get_segment(2)
        self.assertEqual(caught.exception.index, 2)


class TestNoSecondExceptionWhileReporting(unittest.TestCase):
    """The strict and lenient paths report a truncated table identically."""

    def test_lenient_reports_truncated_table_once(self):
        data = minimal_elf(shnum=2**16 - 1, shoff=64, body=b'\0' * 64)
        elf = ELFFile(stream(data))
        scan = elf.iter_sections_lenient()
        self.assertEqual(len(scan), 1)
        self.assertEqual(scan.truncated_at, 1)
        self.assertEqual(len(scan.rejections), 1)
        self.assertEqual(scan.rejections[0].index, 1)
        # The strict path raises for the same first unavailable index.
        with self.assertRaises(ELFParseError) as caught:
            elf.get_section(scan.rejections[0].index)
        self.assertEqual(caught.exception.first_invalid_index, 1)


class TestStabSection(unittest.TestCase):
    """Stab entries are bounded by their section like any table."""

    def test_truncated_stab_entry_rejected(self):
        structs = make_structs(32)
        entry_size = structs.Elf_Stabs.sizeof()
        data = b'\0' * (entry_size + entry_size // 2)
        f = io.BytesIO(data)
        elf = _FakeElf(f, structs)
        section = StabSection(Container(
            sh_type='SHT_PROGBITS', sh_offset=0, sh_size=len(data),
            sh_addralign=1, sh_flags=0), '.stab', elf)
        # The whole entry is parsed; the half-entry tail must not be.
        seen = 0
        with self.assertRaises(ELFParseError) as caught:
            for _stab in section.iter_stabs():
                seen += 1
        self.assertEqual(seen, 1)
        self.assertEqual(caught.exception.structure, 'stab section')
        self.assertEqual(caught.exception.actual, entry_size // 2)

    def test_stab_section_past_eof_rejected(self):
        structs = make_structs(32)
        f = io.BytesIO(b'\0' * 10)
        elf = _FakeElf(f, structs)
        section = StabSection(Container(
            sh_type='SHT_PROGBITS', sh_offset=0, sh_size=100,
            sh_addralign=1, sh_flags=0), '.stab', elf)
        with self.assertRaises(ELFParseError):
            list(section.iter_stabs())


class TestSymbolTableIndexSection(unittest.TestCase):
    """The SHNDX companion table rejects indices outside its own entries."""

    def test_index_outside_table_rejected(self):
        structs = make_structs(64)
        data = b'\0' * 8
        f = io.BytesIO(data)
        elf = _FakeElf(f, structs)
        section = SymbolTableIndexSection(Container(
            sh_type='SHT_SYMTAB_SHNDX', sh_offset=0, sh_size=8,
            sh_entsize=4, sh_addralign=1, sh_flags=0), '.symtab_shndx', elf,
            symboltable=Container())
        self.assertEqual(section.get_section_index(1), 0)
        for bad in (2, 100, -1):
            with self.subTest(index=bad):
                with self.assertRaises(ELFParseError) as caught:
                    section.get_section_index(bad)
                self.assertEqual(caught.exception.structure,
                                 'symbol table index section')
                self.assertEqual(caught.exception.index, bad)


class TestDynamicSegmentSymbolBounds(unittest.TestCase):
    """A dynamic segment does not read symbols past its recovered count."""

    def test_index_outside_recovered_count_rejected(self):
        from elftools.elf.dynamic import DynamicSegment
        elf = Mock()
        elf.structs = make_structs(64)
        elf.little_endian = True
        raw = io.BytesIO(b'\0' * 64)
        stream = Mock(wraps=raw)
        segment = DynamicSegment.__new__(DynamicSegment)
        segment.header = Container(p_type='PT_DYNAMIC', p_offset=0,
                                   p_filesz=0, p_vaddr=0, p_memsz=0)
        segment.stream = stream
        segment.elffile = elf
        segment.get_table_offset = lambda tag: (0x1000, 0)
        segment.num_symbols = lambda: 2
        segment._get_stringtable = lambda: StringTableSection.__new__(
            StringTableSection)
        with self.assertRaises(ELFParseError) as caught:
            segment.get_symbol(2)
        self.assertEqual(caught.exception.structure, 'symbol table')
        self.assertEqual(caught.exception.index, 2)
        stream.read.assert_not_called()


class TestFileBackedStream(unittest.TestCase):
    """A real file stream takes the same rejection paths as BytesIO."""

    def test_prefix_truncation_on_disk_stream(self):
        import tempfile
        with open(os.path.join(TEST_DIR, 'trailing_null_dies.elf'), 'rb') as f:
            data = f.read()
        with tempfile.TemporaryFile() as stream:
            for length in range(len(data) + 1):
                stream.seek(0)
                stream.truncate(0)
                stream.write(data[:length])
                stream.seek(0)
                with self.subTest(length=length):
                    try:
                        elf = ELFFile(stream)
                        list(elf.iter_sections())
                        list(elf.iter_segments())
                    except ELFError:
                        pass


class TestValidInputsUnchanged(unittest.TestCase):
    """Well-formed corpus files keep their values."""

    def test_section_names_symbols_and_data(self):
        path = os.path.join(TEST_DIR, 'simple_gcc.elf.arm')
        with open(path, 'rb') as f:
            elf = ELFFile(f)
            names = [s.name for s in elf.iter_sections()]
            self.assertEqual(len(names), elf.num_sections())
            self.assertEqual(names[elf.get_section_index('.data')], '.data')
            sec = elf.get_section_by_name('.data')
            self.assertTrue(sec.data())
            symtab = elf.get_section_by_name('.symtab')
            if symtab is not None:
                self.assertEqual(len(list(symtab.iter_symbols())),
                                 symtab.num_symbols())

    def test_no_raw_exceptions_for_any_prefix(self):
        # Byte-by-byte truncation of a legal sample: no non-library exception
        # may escape from any container-level entry point.
        path = os.path.join(TEST_DIR, 'obj_stabs.o.elf')
        with open(path, 'rb') as f:
            data = f.read()
        for length in range(len(data) + 1):
            with self.subTest(length=length):
                try:
                    self._exercise(data[:length])
                except ELFError:
                    pass


    def _exercise(self, data):
        try:
            elf = ELFFile(io.BytesIO(data))
        except ELFUnparsableFileError:
            return
        dict(elf.header)
        sections = list(elf.iter_sections())
        segments = list(elf.iter_segments())
        sections += list(elf.iter_sections_lenient())
        segments += list(elf.iter_segments_lenient())
        for section in sections:
            section.header.__dict__.copy()
            section.data()
            if hasattr(section, 'iter_symbols'):
                for symbol in section.iter_symbols():
                    _ = symbol.name
        for segment in segments:
            segment.header.__dict__.copy()
            segment.data()


if __name__ == '__main__':
    unittest.main()
