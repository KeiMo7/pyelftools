#-------------------------------------------------------------------------------
# elftools tests
#
# Acceptance tests for bounded-memory symbol table access.
#
# This code is in the public domain
#-------------------------------------------------------------------------------
"""Behavioural tests for symbol tables with millions of entries.

The fixtures are synthetic ELF64 images written by test/support/synthetic_elf,
so the tests do not need a compiler and run in a second or two. Memory is
observed with tracemalloc, which counts Python allocations only - exactly the
growth the old "materialize the whole table" design produced.
"""
from __future__ import annotations

import io
import os
import sys
import threading
import tracemalloc
import unittest
from typing import Self

from elftools.common.exceptions import (
    ELFError,
    ELFMemoryLimitError,
    ELFParseError,
    ELFSymbolError,
)
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from support.synthetic_elf import (
    CountingReader,
    SymDef,
    build_elf64_with_symtab,
    make_symbols,
    sym_name,
)

MILLION = 1_000_000


def expected_name(index: int) -> str:
    """The name `make_symbols` gives to the entry at table index `index`.

    Entry 0 is the ELF null symbol; entry `index` carries sym_name(index).
    """
    return '' if index == 0 else sym_name(index)


def open_symtab(image: bytes) -> tuple[ELFFile, SymbolTableSection, CountingReader]:
    reader = CountingReader(image)
    elf = ELFFile(reader)
    symtab = elf.get_section_by_name('.symtab')
    assert isinstance(symtab, SymbolTableSection)
    return elf, symtab, reader


class MemoryProbe:
    """Context manager reporting the peak Python allocation inside it."""
    def __enter__(self) -> Self:
        tracemalloc.start()
        tracemalloc.reset_peak()
        self._start, _ = tracemalloc.get_traced_memory()
        return self

    def __exit__(self, *exc: object) -> None:
        _cur, self.peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    @property
    def growth(self) -> int:
        return self.peak - self._start


class TestStreamingIteration(unittest.TestCase):
    """Order and peak memory of a single sequential pass."""

    def test_iteration_peak_memory_is_bounded(self):
        symbols = make_symbols(MILLION)
        image = build_elf64_with_symtab(symbols)
        _elf, symtab, _ = open_symtab(image)
        self.assertEqual(symtab.num_symbols(), MILLION)

        count = 0
        last_name = None
        with MemoryProbe() as probe:
            for symbol in symtab.iter_symbols():
                count += 1
                last_name = symbol.name
        # The table is ~24 MB and the string table ~29 MB; a streaming pass
        # must stay far below either of them.
        self.assertEqual(count, MILLION)
        self.assertEqual(last_name, sym_name(MILLION - 1))
        self.assertLess(probe.growth, 2 * 1024 * 1024,
                        f'peak grew by {probe.growth} bytes')
        self.assertEqual(symtab.cache_stats()['cached_symbols'], 0)

    def test_iteration_does_not_build_a_list(self):
        image = build_elf64_with_symtab(make_symbols(10_000))
        _, symtab, _ = open_symtab(image)
        iterator = symtab.iter_symbols()
        # A fresh iterator holds no symbol objects at all.
        for _ in range(3):
            next(iterator)
        self.assertEqual(symtab.cache_stats()['cached_symbols'], 0)

    def test_index_range_only_parses_that_range(self):
        image = build_elf64_with_symtab(make_symbols(100_000))
        _, symtab, reader = open_symtab(image)
        symtab.iter_symbols(50_000, 50_010).__iter__()
        reader.reset_counts()
        names = [s.name for s in symtab.iter_symbols(50_000, 50_010)]
        self.assertEqual(names, [expected_name(i) for i in range(50_000, 50_010)])
        # Ten entries, each a single 24-byte read of its own.
        # Each entry costs one 24-byte read plus one small chunked read of
        # its name from the string table.
        self.assertLessEqual(reader.read_count, 20)
        self.assertLessEqual(reader.bytes_read, 10 * (24 + 64))

    def test_total_and_order_unchanged(self):
        image = build_elf64_with_symtab(make_symbols(5000, duplicate_every=97))
        _, symtab, _ = open_symtab(image)
        expected = [s.name for s in make_symbols(5000, duplicate_every=97)]
        self.assertEqual([s.name for s in symtab.iter_symbols()], expected)
        self.assertEqual(
            [s['st_value'] for s in symtab.iter_symbols()],
            [s.value for s in make_symbols(5000, duplicate_every=97)])
        self.assertEqual(symtab.num_symbols(), len(expected))


class TestNameLookup(unittest.TestCase):
    """Only matching entries and referenced strings are parsed."""

    def test_only_matching_entries_are_parsed(self):
        # One match in the middle of a million-entry table.
        image = build_elf64_with_symtab(make_symbols(MILLION))
        _, symtab, reader = open_symtab(image)
        parsed = []
        original = symtab._parse_symbol

        def counting(index: int):
            parsed.append(index)
            return original(index)

        symtab._parse_symbol = counting
        reader.reset_counts()
        found = symtab.get_symbol_by_name(sym_name(500_000))
        self.assertIsNotNone(found)
        self.assertEqual([s.name for s in found], [sym_name(500_000)])
        # Exactly one entry was fully parsed; the scan touched st_name only.
        self.assertEqual(parsed, [500_000])
        # The whole string table is scanned in chunks (~29 MB) but entries
        # are read only in small batches, never all at once.
        # ~123 batches of entries plus ~440 chunks of the string table.
        self.assertLessEqual(reader.read_count, 700)

    def test_nonexistent_name_parses_no_entries(self):
        image = build_elf64_with_symtab(make_symbols(200_000))
        _, symtab, _ = open_symtab(image)
        parsed = []
        original = symtab._parse_symbol
        symtab._parse_symbol = lambda i: (parsed.append(i), original(i))[1]
        self.assertIsNone(symtab.get_symbol_by_name('no_such_symbol_here'))
        self.assertEqual(parsed, [])

    def test_duplicate_names_return_every_entry_in_order(self):
        symbols = make_symbols(1000, duplicate_every=100)
        image = build_elf64_with_symtab(symbols)
        _, symtab, _ = open_symtab(image)
        expected = [i for i, s in enumerate(symbols) if s.name == 'dup_symbol']
        found = symtab.get_symbol_by_name('dup_symbol')
        self.assertIsNotNone(found)
        self.assertEqual(len(found), len(expected))
        self.assertEqual([s['st_value'] for s in found],
                         [i * 16 for i in expected])
        self.assertEqual([s.name for s in found], ['dup_symbol'] * len(expected))

    def test_repeated_query_is_served_from_cache(self):
        image = build_elf64_with_symtab(make_symbols(100_000))
        _, symtab, reader = open_symtab(image)
        target = sym_name(54_321)
        first = symtab.get_symbol_by_name(target)
        self.assertIsNotNone(first)
        reader.reset_counts()
        parsed = []
        original = symtab._parse_symbol
        symtab._parse_symbol = lambda i: (parsed.append(i), original(i))[1]
        second = symtab.get_symbol_by_name(target)
        self.assertEqual([s.name for s in second], [target])
        self.assertEqual(parsed, [], 'second query re-parsed entries')
        self.assertEqual(reader.read_count, 0, 'second query touched the file')
        # Results are equal in every visible field.
        self.assertEqual(first[0].entry, second[0].entry)
        self.assertIs(first[0], second[0])

    def test_cache_is_bounded_and_evicts_correctly(self):
        image = build_elf64_with_symtab(make_symbols(50_000))
        _, symtab, _ = open_symtab(image)
        symtab.max_cached_symbols = 8
        symtab.max_cached_name_lookups = 4
        for i in range(100):
            self.assertEqual(symtab.get_symbol(i).name, expected_name(i))
        self.assertEqual(symtab.cache_stats()['cached_symbols'], 8)
        for i in range(1, 21):
            self.assertEqual(symtab.get_symbol_by_name(sym_name(i))[0].name,
                             expected_name(i))
        self.assertEqual(symtab.cache_stats()['cached_name_lookups'], 4)
        # Evicted names still answer correctly, by reparsing.
        self.assertEqual(symtab.get_symbol_by_name(sym_name(1))[0].name,
                         sym_name(1))

    def test_empty_name_query_matches_null_symbol_only(self):
        image = build_elf64_with_symtab(make_symbols(1000))
        _, symtab, reader = open_symtab(image)
        reader.reset_counts()
        found = symtab.get_symbol_by_name('')
        self.assertIsNotNone(found)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['st_value'], 0)
        # An empty name short-circuits: the table is scanned, not the strings.
        self.assertEqual(found[0].name, '')


class TestStringTable(unittest.TestCase):
    """A string table far larger than the symbol table is never loaded whole."""

    def test_huge_string_table_not_read_whole(self):
        # .strtab is far larger than .symtab: a small table of used names
        # followed by 64 MB of zeros. Only the used part is ever relevant.
        symbols = [SymDef('')] + [
            SymDef(f'used_{i}', value=i, info=0x12) for i in range(100)]
        image = build_elf64_with_symtab(symbols, strtab_pad=64 * 1024 * 1024)
        _, symtab, reader = open_symtab(image)
        self.assertGreater(symtab.stringtable.data_size, 64 * 1024 * 1024)

        reader.reset_counts()
        with MemoryProbe() as probe:
            found = symtab.get_symbol_by_name('used_42')
        self.assertIsNotNone(found)
        self.assertEqual(found[0].name, 'used_42')
        # The scan streams the table; nothing as large as it is held at once.
        self.assertLess(probe.growth, 1024 * 1024,
                        f'peak grew by {probe.growth} bytes')
        self.assertLessEqual(reader.bytes_read,
                             symtab.stringtable.data_size + 1024 * 1024)


class TestIndependentReaders(unittest.TestCase):
    """Several readers of one file object do not disturb each other."""

    def test_interleaved_iterators(self):
        image = build_elf64_with_symtab(make_symbols(2000))
        _elf, symtab, _ = open_symtab(image)
        it1 = symtab.iter_symbols()
        it2 = symtab.iter_symbols(1000, 1010)
        it3 = symtab.iter_symbols()
        names1, names2, names3 = [], [], []
        for _ in range(10):
            names1.append(next(it1).name)
            names2.append(next(it2).name)
            names3.append(next(it3).name)
        self.assertEqual(names1, [expected_name(i) for i in range(10)])
        self.assertEqual(names2, [expected_name(i) for i in range(1000, 1010)])
        self.assertEqual(names3, [expected_name(i) for i in range(10)])

    def test_concurrent_lookups(self):
        image = build_elf64_with_symtab(make_symbols(20_000))
        _, symtab, _ = open_symtab(image)
        errors = []

        def worker(offset: int) -> None:
            try:
                for i in range(offset + 1, 20_000, 997):
                    found = symtab.get_symbol_by_name(sym_name(i))
                    if found is None or found[0].name != sym_name(i):
                        errors.append((i, found))
                    symbol = symtab.get_symbol(i)
                    if symbol.name != sym_name(i) or symbol['st_value'] != i * 16:
                        errors.append((i, symbol))
            except Exception as e:  # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_concurrent_readers_never_see_partial_entries(self):
        image = build_elf64_with_symtab(make_symbols(50_000))
        _, symtab, _ = open_symtab(image)
        stop = threading.Event()
        bad = []

        def scan() -> None:
            while not stop.is_set():
                for symbol in symtab.iter_symbols(1, 5000):
                    if not symbol.name.startswith('sym_') or \
                            symbol['st_info']['type'] != 'STT_FUNC':
                        bad.append(symbol)
                        return

        threads = [threading.Thread(target=scan) for _ in range(3)]
        for t in threads:
            t.start()
        for _ in range(20):
            symtab.get_symbol_by_name(sym_name(30_000))
            symtab.get_symbol(1234)
        stop.set()
        for t in threads:
            t.join()
        self.assertEqual(bad, [])


class TestCleanup(unittest.TestCase):
    """Early exit does not leak memory into the next analysis."""

    def test_clear_cache_after_early_exit(self):
        image = build_elf64_with_symtab(make_symbols(100_000))
        _, symtab, _ = open_symtab(image)
        for _ in symtab.iter_symbols(0, 10_000):
            pass
        symtab.get_symbol_by_name(sym_name(5))
        self.assertGreater(symtab.cache_stats()['cached_symbols'], 0)
        symtab.clear_cache()
        stats = symtab.cache_stats()
        self.assertEqual(stats['cached_symbols'], 0)
        self.assertEqual(stats['cached_name_lookups'], 0)

    def test_peak_does_not_accumulate_across_runs(self):
        image = build_elf64_with_symtab(make_symbols(200_000))
        peaks = []
        for _ in range(3):
            _, symtab, _ = open_symtab(image)
            with MemoryProbe() as probe:
                for symbol in symtab.iter_symbols():
                    if symbol['st_value'] > 100:
                        break
                symtab.clear_cache()
            peaks.append(probe.growth)
        # No run may cost dramatically more than the first one.
        self.assertLess(max(peaks), 2 * 1024 * 1024)
        self.assertLess(max(peaks[1:]), 2 * peaks[0] + 512 * 1024)


class TestErrorReporting(unittest.TestCase):
    """Corruption is reported against the exact entry index."""

    def _corrupt_symbol_name(self, image: bytes, index: int) -> bytes:
        elf = ELFFile(io.BytesIO(image))
        symtab = elf.get_section_by_name('.symtab')
        assert isinstance(symtab, SymbolTableSection)
        offset = symtab['sh_offset'] + index * symtab['sh_entsize']
        data = bytearray(image)
        # st_name is the first four bytes of the entry.
        data[offset:offset + 4] = (0xFFFFFF00).to_bytes(4, 'little')
        return bytes(data)

    def test_out_of_range_name_offset_names_entry(self):
        image = self._corrupt_symbol_name(
            build_elf64_with_symtab(make_symbols(1000)), 777)
        _, symtab, _ = open_symtab(image)
        with self.assertRaises(ELFSymbolError) as caught:
            symtab.get_symbol(777)
        self.assertEqual(caught.exception.index, 777)
        self.assertIn('777', str(caught.exception))
        self.assertIsInstance(caught.exception, ELFParseError)

    def test_out_of_range_name_offset_in_iteration(self):
        image = self._corrupt_symbol_name(
            build_elf64_with_symtab(make_symbols(1000)), 123)
        _, symtab, _ = open_symtab(image)
        with self.assertRaises(ELFSymbolError) as caught:
            list(symtab.iter_symbols())
        self.assertEqual(caught.exception.index, 123)

    def test_truncated_table_entry_names_index(self):
        image = build_elf64_with_symtab(make_symbols(100))
        elf = ELFFile(io.BytesIO(image))
        symtab = elf.get_section_by_name('.symtab')
        assert isinstance(symtab, SymbolTableSection)
        # Claim one more entry than the file holds.
        data = bytearray(image)
        shoff = elf['e_shoff']
        shdr_offset = shoff + elf._section_name_map['.symtab'] * elf['e_shentsize']
        size_field = shdr_offset + 32  # sh_size in Elf64_Shdr
        data[size_field:size_field + 8] = (
            (symtab.num_symbols() + 1) * symtab['sh_entsize']).to_bytes(8, 'little')
        _, broken, _ = open_symtab(bytes(data))
        with self.assertRaises(ELFSymbolError) as caught:
            list(broken.iter_symbols())
        self.assertEqual(caught.exception.index, 100)

    def test_memory_limit_names_the_data_category(self):
        image = build_elf64_with_symtab(make_symbols(1000, duplicate_every=2))
        _, symtab, _ = open_symtab(image)
        symtab.max_name_candidates = 10
        with self.assertRaises(ELFMemoryLimitError) as caught:
            symtab.get_symbol_by_name('dup_symbol')
        self.assertEqual(caught.exception.category, 'symbol-names')
        self.assertIn('10', str(caught.exception))
        # The limit refuses rather than silently truncating the result.
        self.assertIsInstance(caught.exception, ELFError)


class TestDebugSections(unittest.TestCase):
    """Presence of debug sections does not change symbol behavior."""

    def test_same_results_with_debug_sections(self):
        symbols = make_symbols(50_000, duplicate_every=101)
        plain = build_elf64_with_symtab(symbols)
        debug = build_elf64_with_symtab(symbols, debug_sections=True)
        _, plain_tab, _ = open_symtab(plain)
        _, debug_tab, _ = open_symtab(debug)
        self.assertTrue(debug_tab.elffile.has_dwarf_info())
        self.assertEqual(plain_tab.num_symbols(), debug_tab.num_symbols())
        self.assertEqual(
            [(s.name, s['st_value']) for s in plain_tab.iter_symbols()],
            [(s.name, s['st_value']) for s in debug_tab.iter_symbols()])
        self.assertEqual(
            [s.name for s in plain_tab.get_symbol_by_name('dup_symbol')],
            [s.name for s in debug_tab.get_symbol_by_name('dup_symbol')])


class TestCompatibility(unittest.TestCase):
    """The old call style still works and yields the same fields."""

    def test_small_file_behavior_unchanged(self):
        path = os.path.join('test', 'testfiles_for_unittests',
                            'simple_gcc.elf.arm')
        with open(path, 'rb') as f:
            elf = ELFFile(f)
            symtab = elf.get_section_by_name('.symtab')
            assert isinstance(symtab, SymbolTableSection)
            mains = symtab.get_symbol_by_name('main')
            self.assertEqual(len(mains), 1)
            self.assertEqual(mains[0].name, 'main')
            self.assertEqual(mains[0]['st_value'], 0x8068)
            self.assertEqual(mains[0]['st_size'], 0x28)
            self.assertEqual(mains[0]['st_info']['bind'], 'STB_GLOBAL')
            self.assertEqual(mains[0]['st_info']['type'], 'STT_FUNC')
            arm_markers = symtab.get_symbol_by_name('$a')
            self.assertEqual(len(arm_markers), 7)
            self.assertIsNone(symtab.get_symbol_by_name('non-existent'))
            self.assertEqual(symtab.num_symbols(), 50)
            # Field order inside an entry is unchanged.
            self.assertEqual(
                list(symtab.get_symbol(1).entry.keys()),
                ['st_name', 'st_value', 'st_size', 'st_info', 'st_other',
                 'st_shndx'])


if __name__ == '__main__':
    unittest.main()
