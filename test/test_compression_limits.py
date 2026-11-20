"""Bounded, explainable decompression of compressed ELF sections.

These tests build synthetic ELF files in memory so that every length
contradiction can be produced exactly, and check that each failure carries a
distinct machine-readable category instead of returning truncated data.
"""
import struct
import tracemalloc
import unittest
import zlib
from io import BytesIO

from elftools.common.compression import (
    DEFAULT_MAX_DECOMPRESSED_SIZE,
    CompressionFailure,
    decompress_zlib_section,
)
from elftools.common.exceptions import ELFCompressionError
from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile

SHT_PROGBITS = 1
SHT_STRTAB = 3

EM_X86_64 = 62
ET_REL = 1


def chdr(declared_size, ch_type=1, addralign=1):
    """ELF64 compression header (ELFCOMPRESS_ZLIB = 1 by default)."""
    return struct.pack('<IIQQ', ch_type, 0, declared_size, addralign)


def build_elf(sections):
    """Build a minimal ELF64 little-endian relocatable object.

    ``sections`` is a list of (name, sh_flags, payload_bytes).  A .shstrtab is
    appended automatically.
    """
    names = [name for name, _, _ in sections] + ['.shstrtab']
    shstrtab = b'\0'
    name_offsets = {}
    for name in names:
        name_offsets[name] = len(shstrtab)
        shstrtab += name.encode() + b'\0'

    ehdr_size = 64
    body = bytearray()
    placed = []
    offset = ehdr_size
    for name, flags, payload in sections:
        placed.append((name, flags, offset, len(payload)))
        body += payload
        offset += len(payload)
    shstrtab_offset = offset
    body += shstrtab
    offset += len(shstrtab)

    # Align the section header table to 8 bytes.
    if offset % 8:
        body += bytes(8 - offset % 8)
        offset += 8 - offset % 8
    shoff = offset

    shdrs = [struct.pack('<IIQQQQIIQQ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for name, flags, sec_off, sec_size in placed:
        shdrs.append(struct.pack(
            '<IIQQQQIIQQ', name_offsets[name], SHT_PROGBITS, flags, 0,
            sec_off, sec_size, 0, 0, 1, 0))
    shdrs.append(struct.pack(
        '<IIQQQQIIQQ', name_offsets['.shstrtab'], SHT_STRTAB, 0, 0,
        shstrtab_offset, len(shstrtab), 0, 0, 1, 0))
    shnum = len(shdrs)
    shstrndx = shnum - 1

    ident = b'\x7fELF' + bytes([2, 1, 1, 0]) + bytes(8)
    ehdr = ident + struct.pack(
        '<HHIQQQIHHHHHH', ET_REL, EM_X86_64, 1, 0, 0, shoff, 0,
        ehdr_size, 0, 0, 64, shnum, shstrndx)
    return bytes(ehdr) + bytes(body) + b''.join(shdrs)


def compressed_section(name, payload, declared_size=None, suffix=b'',
                       ch_type=1):
    """Build one SHF_COMPRESSED section wrapping ``payload`` with zlib."""
    if declared_size is None:
        declared_size = len(payload)
    return (name, SH_FLAGS.SHF_COMPRESSED,
            chdr(declared_size, ch_type) + zlib.compress(payload) + suffix)


def open_elf(data):
    return ELFFile(BytesIO(data))


class CategoryAssertions(unittest.TestCase):
    def assert_category(self, error, category):
        self.assertIsInstance(error, ELFCompressionError)
        self.assertEqual(error.category, category)
        # str-valued enum, so textual comparisons also work.
        self.assertEqual(error.category, category.value)


class CompressionCategoryTest(CategoryAssertions):
    def test_declared_smaller_than_actual(self):
        payload = b'abcdefghij' * 100
        data = build_elf([compressed_section('.debug_info', payload,
                                             declared_size=len(payload) - 7)])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error, CompressionFailure.DATA_LONGER_THAN_DECLARED)
        self.assertGreater(error.decompressed_size, error.declared_size)
        self.assertEqual(error.extra_bytes,
                         error.decompressed_size - error.declared_size)
        # No truncated success: nothing is returned.
        self.assertNotIsInstance(error, bytes)

    def test_declared_larger_than_actual(self):
        payload = b'abcdefghij' * 100
        data = build_elf([compressed_section('.debug_info', payload,
                                             declared_size=len(payload) + 13)])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error,
                             CompressionFailure.DATA_SHORTER_THAN_DECLARED)
        self.assertEqual(error.declared_size, len(payload) + 13)
        self.assertEqual(error.decompressed_size, len(payload))
        self.assertEqual(error.missing_bytes, 13)

    def test_trailing_data_is_distinct(self):
        payload = b'x' * 4096
        data = build_elf([compressed_section(
            '.debug_info', payload, suffix=b'\0' * 32)])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error, CompressionFailure.TRAILING_DATA)
        self.assertEqual(error.trailing_bytes, 32)
        self.assertEqual(error.decompressed_size, len(payload))
        self.assertNotEqual(error.category,
                            CompressionFailure.DATA_LONGER_THAN_DECLARED)
        self.assertNotEqual(error.category,
                            CompressionFailure.DATA_SHORTER_THAN_DECLARED)

    def test_second_concatenated_stream_counts_as_trailing(self):
        payload = b'first' * 50
        suffix = zlib.compress(b'second' * 50)
        data = build_elf([compressed_section('.debug_info', payload,
                                             suffix=suffix)])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        self.assert_category(caught.exception,
                             CompressionFailure.TRAILING_DATA)
        self.assertEqual(caught.exception.trailing_bytes, len(suffix))

    def test_absurd_declared_size_rejected_before_reading(self):
        payload = b'tiny'
        data = build_elf([compressed_section('.debug_info', payload,
                                             declared_size=1 << 40)])
        section = open_elf(data).get_section_by_name('.debug_info')
        # The section is a few hundred bytes; 2**40 is ~10**10 times larger.
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error,
                             CompressionFailure.DECLARED_SIZE_TOO_LARGE)
        self.assertEqual(error.declared_size, 1 << 40)
        self.assertEqual(error.max_output_size, DEFAULT_MAX_DECOMPRESSED_SIZE)
        self.assertEqual(error.decompressed_size, 0)

    def test_absurd_declared_size_does_not_read(self):
        class NoReadStream(BytesIO):
            def read(self, *args):
                raise AssertionError('must not read before rejecting size')

        stream = NoReadStream(bytes(4096))
        with self.assertRaises(ELFCompressionError) as caught:
            decompress_zlib_section(
                stream, 0, 64, 1 << 40, section_label='section X')
        self.assert_category(caught.exception,
                             CompressionFailure.DECLARED_SIZE_TOO_LARGE)

    def test_output_limit_exceeded_despite_small_declaration(self):
        # ~64 MiB of zeros compresses to ~64 KiB, but only 1 KiB is declared.
        payload = bytes(64 * 1024 * 1024)
        limit = 4 * 1024 * 1024
        stored = zlib.compress(payload, 9)
        data = build_elf([compressed_section('.debug_info', payload,
                                             declared_size=1024)])
        elf = open_elf(data)
        elf.decompression_limit = limit
        section = elf.get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error, CompressionFailure.OUTPUT_LIMIT_EXCEEDED)
        self.assertEqual(error.max_output_size, limit)
        self.assertEqual(error.declared_size, 1024)
        self.assertGreater(error.decompressed_size, limit)
        self.assertGreater(len(stored), 0)

    def test_output_limit_peak_memory_is_bounded(self):
        payload = bytes(32 * 1024 * 1024)
        limit = 2 * 1024 * 1024
        data = build_elf([compressed_section('.debug_info', payload,
                                             declared_size=1024)])
        elf = open_elf(data)
        elf.decompression_limit = limit
        section = elf.get_section_by_name('.debug_info')
        tracemalloc.start()
        try:
            with self.assertRaises(ELFCompressionError):
                section.data()
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The whole 32 MiB is never materialised: peak stays near the limit.
        self.assertLess(peak, limit + 4 * 1024 * 1024)

    def test_unsupported_algorithm_reports_id_and_location(self):
        payload = b'some bytes'
        data = build_elf([compressed_section('.debug_info', payload,
                                             ch_type=0x7ffffffe)])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error,
                             CompressionFailure.UNSUPPORTED_ALGORITHM)
        self.assertEqual(error.algorithm, 0x7ffffffe)
        self.assertIn('.debug_info', error.section)
        self.assertIn('0x7ffffffe', str(error))
        # Must not fall back to raw compressed bytes.
        self.assertNotIn(b'\x78\x9c', data[data.find(payload) - 64:])

    def test_corrupt_stream_middle_reports_progress(self):
        payload = bytes(range(256)) * 200
        stored = bytearray(zlib.compress(payload, 1))
        # Corrupt a byte well inside the deflate stream.
        stored[len(stored) // 2] ^= 0xff
        data = build_elf([('.debug_info', SH_FLAGS.SHF_COMPRESSED,
                           chdr(len(payload)) + bytes(stored))])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error, CompressionFailure.CORRUPT_STREAM)
        self.assertGreaterEqual(error.decompressed_size, 0)
        self.assertIsNotNone(getattr(error, 'compressed_offset', None))

    def test_zero_declared_size_with_compressed_bytes(self):
        payload = b'not empty'
        data = build_elf([compressed_section('.debug_info', payload,
                                             declared_size=0)])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error, CompressionFailure.EMPTY_DECLARED_SIZE)
        self.assertEqual(error.declared_size, 0)

    def test_zero_declared_size_and_no_bytes_succeeds(self):
        data = build_elf([('.debug_info', SH_FLAGS.SHF_COMPRESSED,
                           chdr(0))])
        section = open_elf(data).get_section_by_name('.debug_info')
        self.assertEqual(section.data(), b'')

    def test_one_failing_section_does_not_block_others(self):
        good_payload = b'good section contents' * 64
        data = build_elf([
            compressed_section('.debug_one', good_payload),
            compressed_section('.debug_two', b'bad', declared_size=100),
            compressed_section('.debug_three', good_payload[::-1]),
        ])
        elf = open_elf(data)
        self.assertEqual(elf.get_section_by_name('.debug_one').data(),
                         good_payload)
        with self.assertRaises(ELFCompressionError):
            elf.get_section_by_name('.debug_two').data()
        self.assertEqual(elf.get_section_by_name('.debug_three').data(),
                         good_payload[::-1])
        # And the good ones remain readable afterwards.
        self.assertEqual(elf.get_section_by_name('.debug_one').data(),
                         good_payload)

    def test_truncated_stream_at_declared_length_is_failure(self):
        payload = b'abcdefghij' * 100
        stored = zlib.compress(payload)
        # Keep only the prefix that decompresses to exactly declared_size:
        # dropping the trailing adler32 leaves the output complete but the
        # stream without an end marker.
        truncated = stored[:-4]
        data = build_elf([('.debug_info', SH_FLAGS.SHF_COMPRESSED,
                           chdr(len(payload)) + truncated)])
        section = open_elf(data).get_section_by_name('.debug_info')
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        error = caught.exception
        self.assert_category(error, CompressionFailure.INCOMPLETE_STREAM)
        self.assertEqual(error.declared_size, error.decompressed_size)

    def test_valid_compressed_matches_uncompressed(self):
        payload = bytes(range(256)) * 40
        compressed = build_elf([compressed_section('.debug_info', payload)])
        plain = build_elf([('.debug_info', 0, payload)])
        self.assertEqual(
            open_elf(compressed).get_section_by_name('.debug_info').data(),
            open_elf(plain).get_section_by_name('.debug_info').data())
        self.assertEqual(
            open_elf(compressed).get_section_by_name('.debug_info').data(),
            payload)

    def test_limit_is_configurable_per_section(self):
        payload = b'y' * 1024
        data = build_elf([compressed_section('.debug_info', payload)])
        elf = open_elf(data)
        section = elf.get_section_by_name('.debug_info')
        self.assertEqual(section.data(), payload)
        elf.decompression_limit = 1023
        with self.assertRaises(ELFCompressionError) as caught:
            section.data()
        self.assert_category(caught.exception,
                             CompressionFailure.DECLARED_SIZE_TOO_LARGE)
        self.assertEqual(caught.exception.max_output_size, 1023)

    def test_chunk_sizes_are_bounded(self):
        from elftools.common import compression
        self.assertLessEqual(compression.DECOMPRESS_INPUT_CHUNK_SIZE,
                             1024 * 1024)
        self.assertLessEqual(compression.DECOMPRESS_OUTPUT_CHUNK_SIZE,
                             1024 * 1024)
        # Compression ratio this high is impossible with zlib; the declared
        # size never becomes an allocation size.
        self.assertLess(compression.DECOMPRESS_OUTPUT_CHUNK_SIZE,
                        DEFAULT_MAX_DECOMPRESSED_SIZE)

    def test_header_too_small_has_category(self):
        # A SHF_COMPRESSED section whose sh_size cannot hold an Elf_Chdr.
        data = build_elf([('.debug_info', SH_FLAGS.SHF_COMPRESSED, b'\x01\x00')])
        with self.assertRaises(ELFCompressionError) as caught:
            open_elf(data).get_section_by_name('.debug_info')
        self.assert_category(caught.exception,
                             CompressionFailure.INCOMPLETE_STREAM)


class GnuZdebugTest(CategoryAssertions):
    """GNU-style .zdebug_* sections share the same categorised machinery."""

    @staticmethod
    def gnu_section(name, payload, declared_size=None, suffix=b''):
        if declared_size is None:
            declared_size = len(payload)
        body = (b'ZLIB' + declared_size.to_bytes(8, 'big')
                + zlib.compress(payload) + suffix)
        return (name, 0, body)

    @staticmethod
    def dwarf_elf(gnu_payload):
        # include a valid .debug_abbrev so DWARFInfo can be constructed with
        # only .zdebug_info failing.
        data = build_elf([
            ('.zdebug_info', 0, gnu_payload),
            ('.debug_abbrev', 0, b'\x01\x11\x01\x00\x00\x00'),
        ])
        return ELFFile(BytesIO(data))

    def test_gnu_valid_section_decompresses(self):
        payload = b'gnu dwarf bytes' * 100
        body = (b'ZLIB' + len(payload).to_bytes(8, 'big')
                + zlib.compress(payload))
        elf = self.dwarf_elf(body)
        dwarf = elf.get_dwarf_info()
        sec = dwarf.debug_info_sec
        sec.stream.seek(0)
        self.assertEqual(sec.stream.read(), payload)

    def test_gnu_trailing_data_rejected(self):
        payload = b'gnu dwarf bytes' * 100
        body = (b'ZLIB' + len(payload).to_bytes(8, 'big')
                + zlib.compress(payload) + b'\0' * 16)
        elf = self.dwarf_elf(body)
        with self.assertRaises(ELFCompressionError) as caught:
            elf.get_dwarf_info()
        self.assert_category(caught.exception,
                             CompressionFailure.TRAILING_DATA)

    def test_gnu_short_data_rejected(self):
        payload = b'gnu dwarf bytes' * 10
        body = (b'ZLIB' + (len(payload) + 5).to_bytes(8, 'big')
                + zlib.compress(payload))
        elf = self.dwarf_elf(body)
        with self.assertRaises(ELFCompressionError) as caught:
            elf.get_dwarf_info()
        self.assert_category(caught.exception,
                             CompressionFailure.DATA_SHORTER_THAN_DECLARED)
        self.assertEqual(caught.exception.missing_bytes, 5)

    def test_gnu_absurd_size_rejected(self):
        payload = b'x'
        body = (b'ZLIB' + (1 << 48).to_bytes(8, 'big')
                + zlib.compress(payload))
        elf = self.dwarf_elf(body)
        with self.assertRaises(ELFCompressionError) as caught:
            elf.get_dwarf_info()
        self.assert_category(caught.exception,
                             CompressionFailure.DECLARED_SIZE_TOO_LARGE)


class LegacyFixtureRegressionTest(unittest.TestCase):
    """The pre-existing valid fixtures must decompress byte-identically."""

    def test_valid_fixtures_match_zlib(self):
        import os
        for name, section_name in (
                ('compressed_32.o', '.debug_info'),
                ('compressed_32.o', '.debug_abbrev'),
                ('compressed_64.o', '.debug_info'),
                ('compressed_64.o', '.debug_abbrev'),
                ('compressed_64.o', '.debug_str')):
            path = os.path.join('test', 'testfiles_for_unittests', name)
            with (self.subTest(fixture=name, section=section_name),
                  open(path, 'rb') as f):
                raw = f.read()
                elf = ELFFile(BytesIO(raw))
                section = elf.get_section_by_name(section_name)
                if section is None or not section.compressed:
                    continue
                hdr_size = elf.structs.Elf_Chdr.sizeof()
                start = section['sh_offset'] + hdr_size
                expected = zlib.decompress(
                    raw[start:section['sh_offset'] + section['sh_size']])
                self.assertEqual(section.data(), expected)
                self.assertEqual(len(section.data()), section.data_size)

    def test_gnu_zdebug_sections_still_decompress(self):
        import os
        path = os.path.join('test', 'testfiles_for_readelf',
                            'zdebug_mixed.so.elf')
        if not os.path.exists(path):
            self.skipTest('missing zdebug fixture')
        with open(path, 'rb') as f:
            elf = ELFFile(f)
            if not elf.has_section('.zdebug_info'):
                self.skipTest('fixture has no .zdebug_info')
            dwarf = elf.get_dwarf_info()
            self.assertIsNotNone(dwarf.debug_info_sec)


if __name__ == '__main__':
    unittest.main()
