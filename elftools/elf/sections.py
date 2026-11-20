#-------------------------------------------------------------------------------
# elftools: elf/sections.py
#
# ELF sections
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import annotations

import struct
import zlib
from collections import OrderedDict
from threading import RLock
from typing import IO, TYPE_CHECKING, Any, Literal, cast, overload
from weakref import WeakKeyDictionary

from elftools.construct.lib.container import Container

from ..common.exceptions import (
    ELFCompressionError,
    ELFMemoryLimitError,
    ELFParseError,
    ELFSymbolError,
)
from ..common.utils import (
    check_file_range,
    elf_assert,
    parse_cstring_from_stream,
    stream_read,
    struct_parse,
)
from ..construct.core import ConstructError
from .constants import SH_FLAGS
from .notes import iter_notes

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .elffile import ELFFile
    from .structs import ELFStructs


_stream_locks: WeakKeyDictionary[Any, RLock] = WeakKeyDictionary()


def _shared_stream_lock(owner: Any) -> RLock:
    """ Return a per-file reentrant lock guarding seeks on its stream.

        Symbol access reads and parses under this lock, so readers sharing one
        file object can interleave without one seeing the other's cursor or a
        half-read entry. The lock is stored on the owner itself (so lookups
        are a plain attribute read), and is reentrant because parsing may nest.
    """
    lock = getattr(owner, '_elftools_stream_lock', None)
    if lock is not None:
        return lock
    lock = RLock()
    try:
        owner._elftools_stream_lock = lock
    except (AttributeError, TypeError):
        # The owner does not accept attributes; fall back to a weak map.
        key = owner if owner is not None else object()
        try:
            lock = _stream_locks.get(key)
            if lock is None:
                lock = RLock()
                _stream_locks[key] = lock
        except TypeError:
            lock = RLock()
    return lock


class Section:
    """ Base class for ELF sections. Also used for all sections types that have
        no special functionality.

        Allows dictionary-like access to the section header. For example:
         > sec = Section(...)
         > sec['sh_type']  # section type
    """
    def __init__(self, header: Container, name: str, elffile: ELFFile) -> None:
        self.header = header
        self.name = name
        self.elffile = elffile
        self.stream: IO[bytes] = self.elffile.stream
        self.structs: ELFStructs = self.elffile.structs
        self._compressed: int = header['sh_flags'] & SH_FLAGS.SHF_COMPRESSED

        if self.compressed:
            # Read the compression header now to know about the size/alignment
            # of the decompressed data.
            check_file_range(self['sh_offset'], self['sh_size'], self.elffile.stream_len)
            if self['sh_size'] < self.structs.Elf_Chdr.sizeof():
                raise ELFCompressionError('Compressed section is too small for its header')
            header = struct_parse(self.structs.Elf_Chdr,
                                  self.stream,
                                  stream_pos=self['sh_offset'])
            self._compression_type: str | int = header['ch_type']
            self._decompressed_size: int = header['ch_size']
            self._decompressed_align: int = header['ch_addralign']
        else:
            self._decompressed_size = header['sh_size']
            self._decompressed_align = header['sh_addralign']

    @property
    def compressed(self) -> int:
        """ Is this section compressed?
        """
        return self._compressed

    @property
    def data_size(self) -> int:
        """ Return the logical size for this section's data.

        This can be different from the .sh_size header field when the section
        is compressed.
        """
        return self._decompressed_size

    @property
    def data_alignment(self) -> int:
        """ Return the logical alignment for this section's data.

        This can be different from the .sh_addralign header field when the
        section is compressed.
        """
        return self._decompressed_align

    def data(self) -> bytes:
        """ The section data from the file.

        Note that data is decompressed if the stored section data is
        compressed.
        """
        # If this section is NOBITS, there is no data. provide a dummy answer
        if self.header['sh_type'] == 'SHT_NOBITS':
            try:
                return b'\0'*self.data_size
            except OverflowError as e:
                raise ELFParseError(f'NOBITS section size {self.data_size} is too large') from e

        # Section zero can store extended numbering in sh_size, not file data.
        if self.header['sh_type'] == 'SHT_NULL':
            return b''

        # If this section is compressed, deflate it
        if self.compressed:
            c_type = self._compression_type
            if c_type == 'ELFCOMPRESS_ZLIB':
                # Read the data to decompress starting right after the
                # compression header until the end of the section.
                hdr_size: int = self.structs.Elf_Chdr.sizeof()
                compressed = stream_read(
                    self.stream, self['sh_offset'] + hdr_size, self['sh_size'] - hdr_size)

                decomp = zlib.decompressobj()
                result = decomp.decompress(compressed, self.data_size)
            elif isinstance(c_type, int):
                raise ELFCompressionError(
                    f'Unknown compression type: {c_type:#0x}'
                )
            else:
                raise ELFCompressionError(
                    f'Unknown compression type: {c_type!r}'
                )

            if len(result) != self._decompressed_size:
                raise ELFCompressionError(
                    f'Decompressed data is {len(result)} bytes long, should be {self._decompressed_size} bytes'
                    ' long'
                )
        else:
            result = stream_read(self.stream, self['sh_offset'], self._decompressed_size)

        return result

    def is_null(self) -> bool:
        """ Is this a null section?
        """
        return False

    @overload
    def __getitem__(self, name: Literal["sh_addr", "sh_entsize", "sh_flags", "sh_offset", "sh_size"]) -> int: ...
    @overload
    def __getitem__(self, name: Literal["sh_type"]) -> str | int: ...
    @overload
    def __getitem__(self, name: str) -> Any: ...
    def __getitem__(self, name: str) -> Any:
        """ Implement dict-like access to header entries
        """
        return self.header[name]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Section) and self.header == other.header

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.name!r} "
                f"addr={self['sh_addr']:#x} size={self['sh_size']:#x} "
                f"type={self['sh_type']!r}>")


class NullSection(Section):
    """ ELF NULL section
    """
    def is_null(self) -> bool:
        return True


class StringTableSection(Section):
    """ ELF string table section.
    """
    # When the table is compressed it has to be inflated before it can be
    # scanned. Refuse to inflate an unreasonably large table rather than
    # silently materializing it.
    max_decompressed_scan_size: int = 256 * 1024 * 1024

    def get_string(self, offset: int) -> str:
        """ Get the string stored at the given offset in this string table.
        """
        table_offset = self['sh_offset']
        with _shared_stream_lock(self.elffile):
            s = parse_cstring_from_stream(self.stream, table_offset + offset)
        return s.decode('utf-8', errors='replace') if s else ''

    def _iter_scan_chunks(self, chunk_size: int) -> Iterator[bytes]:
        """ Yield the logical (decompressed, if needed) table bytes in order,
            in chunks no larger than chunk_size.

            Each chunk is read at an explicit offset under the shared stream
            lock, so concurrent readers of the same file cannot disturb each
            other.
        """
        size = self.data_size
        if self.compressed:
            if size > self.max_decompressed_scan_size:
                raise ELFMemoryLimitError(
                    f'string table {self.name!r} would need {size} bytes to '
                    f'decompress for scanning, limit is '
                    f'{self.max_decompressed_scan_size}',
                    category='symbol-names')
            data = self.data()
            for off in range(0, len(data), chunk_size):
                yield data[off:off + chunk_size]
            return

        table_offset = self['sh_offset']
        for off in range(0, size, chunk_size):
            n = min(chunk_size, size - off)
            with _shared_stream_lock(self.elffile):
                chunk = stream_read(self.stream, table_offset + off, n)
            # Yield after releasing the lock so a scan in one thread does not
            # starve readers in another for its whole duration.
            yield chunk


class _BoundedSymbolAccess:
    """ Bounded-memory access to a symbol table.

        This mixin implements iteration, random access and name lookup without
        ever materializing the whole table. The memory held by an instance is
        bounded by configuration, never by the number of symbols in the file:

          - ``iter_symbols`` streams entries and holds a single entry at a
            time (plus a bounded read buffer),
          - ``get_symbol`` keeps recently used entries in an LRU cache whose
            size is ``max_cached_symbols``,
          - ``get_symbol_by_name`` scans only the ``st_name`` fields of the
            table (never parsing entries) and keeps recent query results in an
            LRU cache whose size is ``max_cached_name_lookups``. Only the
            entries whose name matches are parsed, and only the strings they
            reference are read.

        Subclasses provide the table geometry and string-table access through
        the ``_symbol_*`` hooks. A table whose lookups would need more memory
        than allowed raises :class:`ELFMemoryLimitError` naming the data
        category; results are never silently truncated.

        Requested entries and query results are cached in a bounded LRU, so a
        caller can call :meth:`clear_cache` to drop them as soon as a traversal
        ends early.
    """

    #: How many recently used entries are kept parsed in memory.
    max_cached_symbols: int = 4096
    #: How many recent name queries keep their index lists in memory.
    max_cached_name_lookups: int = 1024
    #: How many matching entries one name query may collect. Exceeding it
    #: raises ELFMemoryLimitError instead of returning a truncated answer.
    max_name_candidates: int = 1 << 18
    #: How many symbol entries are read at once while scanning for a name.
    #: Together with the string-table chunk size this caps the read buffer.
    scan_batch_entries: int = 8192
    #: How many bytes of the string table are read at once while scanning.
    scan_chunk_bytes: int = 1 << 16

    def _init_symbol_caches(self) -> None:
        self._symbol_cache_lock = RLock()
        self._symbol_cache: OrderedDict[int, Symbol] = OrderedDict()
        self._name_index_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    # -- Hooks provided by the concrete table type --------------------------

    def num_symbols(self) -> int:
        """ Number of symbols in the table
        """
        raise NotImplementedError

    def _symbol_table_bounds(self) -> tuple[int, int]:
        """ Return the file offset at which the table starts and its size.
        """
        raise NotImplementedError

    def _symbol_entry_size(self) -> int:
        """ The on-disk stride of one symbol entry.
        """
        raise NotImplementedError

    def _symbol_struct(self) -> Any:
        """ The construct that parses one symbol entry.
        """
        raise NotImplementedError

    def _symbol_structs(self) -> ELFStructs:
        """ The ELFStructs of the owning file (for endianness).
        """
        raise NotImplementedError

    def _symbol_stream(self) -> IO[bytes]:
        """ The stream backing the table.
        """
        raise NotImplementedError

    def _symbol_stream_owner(self) -> Any:
        """ Object whose per-file stream lock guards this table.
        """
        raise NotImplementedError

    def _symbol_stringtable(self) -> Any:
        """ The associated string table (get_string and bounded chunk scan).
        """
        raise NotImplementedError

    # -- Raw reads ----------------------------------------------------------

    def _read_at(self, offset: int, size: int, index: int) -> bytes:
        """ Read `size` bytes at `offset` in one locked operation.
        """
        stream = self._symbol_stream()
        with _shared_stream_lock(self._symbol_stream_owner()):
            try:
                stream.seek(offset)
            except (OverflowError, ValueError) as e:
                raise ELFSymbolError(
                    f'invalid seek offset {offset}', index) from e
            data = stream.read(size)
        if len(data) != size:
            raise ELFSymbolError(
                f'entry is truncated: expected {size} bytes at offset '
                f'{offset}, found {len(data)}', index)
        return data

    def _symbol_fast_decoder(self) -> Any:
        """ A fast decoder for the entry struct, or None if unavailable.
        """
        return getattr(self._symbol_structs(), 'Elf_Sym_fast', None)

    def _symbol_struct_size(self) -> int:
        """ sizeof() of the entry struct, computed once per table.
        """
        size = getattr(self, '_symbol_struct_size_cache', None)
        if size is None:
            size = self._symbol_struct().sizeof()
            self._symbol_struct_size_cache = size
        return size

    def _parse_symbol(self, index: int) -> Symbol:
        """ Parse one entry; failures name the offending index.
        """
        struct = self._symbol_struct()
        full_size = self._symbol_struct_size()
        entsize = self._symbol_entry_size()
        size = min(entsize, full_size) if entsize else full_size
        offset, table_size = self._symbol_table_bounds()
        entry_offset = offset + index * entsize
        if offset < 0 or entry_offset < 0 or entry_offset + size > offset + table_size:
            raise ELFSymbolError(
                f'entry at index {index} extends past the end of the table',
                index)
        data = self._read_at(entry_offset, size, index)
        decoder = self._symbol_fast_decoder()
        entry = decoder.decode(data) if decoder is not None else None
        if entry is None:
            try:
                entry = struct.parse(data)
            except ConstructError as e:
                raise ELFSymbolError(f'corrupt entry: {e}', index) from e
        name = self._resolve_name(entry['st_name'], index)
        return Symbol(entry, name)

    def _resolve_name(self, st_name: int, index: int) -> str:
        stringtable = self._symbol_stringtable()
        size = getattr(stringtable, 'data_size', None)
        if size is not None and st_name >= size:
            raise ELFSymbolError(
                f'name offset {st_name} is outside string table '
                f'{getattr(stringtable, "name", "?")!r} of size {size}', index)
        try:
            return stringtable.get_string(st_name)
        except ELFSymbolError:
            raise
        except Exception as e:
            raise ELFSymbolError(
                f'cannot read name at string table offset {st_name}: {e}',
                index) from e

    # -- Caches -------------------------------------------------------------

    def _cache_get(self, cache: OrderedDict[Any, Any], key: Any) -> Any:
        with self._symbol_cache_lock:
            value = cache.get(key)
            if value is not None:
                cache.move_to_end(key)
            return value

    def _cache_put(self, cache: OrderedDict[Any, Any], key: Any, value: Any,
                   limit: int) -> None:
        if limit <= 0:
            return
        with self._symbol_cache_lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > limit:
                cache.popitem(last=False)

    def _cached_symbol(self, index: int) -> Symbol:
        cached = self._cache_get(self._symbol_cache, index)
        if cached is not None:
            return cached
        symbol = self._parse_symbol(index)
        self._cache_put(self._symbol_cache, index, symbol,
                        self.max_cached_symbols)
        return symbol

    def clear_cache(self) -> None:
        """ Release every cached symbol and name lookup.

            Use this when a traversal is abandoned early so the memory does not
            carry over into the next analysis. The caches are also released
            when the owning section is discarded.
        """
        with self._symbol_cache_lock:
            self._symbol_cache.clear()
            self._name_index_cache.clear()

    def cache_stats(self) -> dict[str, int]:
        """ Report current cache occupancy together with the limits.
        """
        with self._symbol_cache_lock:
            return {
                'cached_symbols': len(self._symbol_cache),
                'max_cached_symbols': self.max_cached_symbols,
                'cached_name_lookups': len(self._name_index_cache),
                'max_cached_name_lookups': self.max_cached_name_lookups,
            }

    # -- Name lookup helpers ------------------------------------------------

    def _matching_string_offsets(self, name: str) -> set[int] | None:
        """ All string-table offsets that hold exactly `name`.

            The table is scanned in bounded chunks and never read as a whole.
            Any position is accepted, not just the start of a string, because
            linkers may merge suffixes and point ``st_name`` into the middle of
            another name.

            Returns None when an exact answer cannot be derived from the bytes
            (empty or undecodable names); the caller then resolves names entry
            by entry, which is still bounded.
        """
        try:
            needle = name.encode('utf-8')
        except UnicodeEncodeError:
            return None
        if not needle or '�' in name:
            # Undecodable bytes in the table become U+FFFD on decode, and the
            # empty string is reachable from any NUL byte; neither can be
            # matched byte-exactly.
            return None

        strtab = self._symbol_stringtable()
        strtab_size = getattr(strtab, 'data_size', None)
        if strtab_size is None:
            return None
        chunks = getattr(strtab, '_iter_scan_chunks', None)
        if chunks is None:
            return None

        pattern = needle + b'\x00'
        # Keep the last len(pattern) - 1 bytes so a match spanning a chunk
        # boundary is still recognized.
        overlap = len(pattern) - 1
        found: set[int] = set()
        tail = b''
        consumed = 0
        try:
            for chunk in chunks(self.scan_chunk_bytes):
                buf = tail + chunk
                buf_start = consumed - len(tail)
                pos = buf.find(pattern)
                while pos >= 0:
                    found.add(buf_start + pos)
                    if len(found) > self.max_name_candidates:
                        raise ELFMemoryLimitError(
                            f'name {name!r} occurs at more than '
                            f'{self.max_name_candidates} string table offsets',
                            category='symbol-names')
                    pos = buf.find(pattern, pos + 1)
                tail = buf[-overlap:] if overlap else b''
                consumed += len(chunk)
        except (ELFMemoryLimitError, ELFSymbolError):
            raise
        except Exception:  # noqa: BLE001 - fall back to per-entry resolution
            return None
        # An offset at or past the end of the table cannot be referenced.
        return {o for o in found if o < strtab_size}

    def _iter_st_names(self) -> Iterator[tuple[int, int]]:
        """ Yield (index, st_name) for every entry in bounded batches.

            Only the first four bytes of each entry are unpacked, so scanning
            the table never builds entry objects.
        """
        count = self.num_symbols()
        entsize = self._symbol_entry_size()
        if entsize < 4:
            raise ELFParseError(
                f'symbol entry size {entsize} is too small to hold st_name')
        fmt = '<I' if self._symbol_structs().little_endian else '>I'
        batch = max(1, self.scan_batch_entries)
        offset, table_size = self._symbol_table_bounds()
        for first in range(0, count, batch):
            n = min(batch, count - first)
            nbytes = n * entsize
            start = offset + first * entsize
            if start + nbytes > offset + table_size:
                # Keep whichever entries are fully inside the table; a
                # truncated entry is reported when it is reached.
                n = max(0, (offset + table_size - start) // entsize)
                nbytes = n * entsize
            if n == 0:
                break
            raw = self._read_at(start, nbytes, first)
            unpack = struct.unpack_from
            for k in range(n):
                yield first + k, unpack(fmt, raw, k * entsize)[0]

    def _indices_by_name(self, name: str) -> tuple[int, ...]:
        """ Indices of every entry whose name is `name`, in table order.
        """
        count = self.num_symbols()
        offsets = self._matching_string_offsets(name)
        result: list[int] = []
        if offsets is not None and not offsets:
            return ()
        for i, st_name in self._iter_st_names():
            if offsets is not None and st_name not in offsets:
                continue
            if self._resolve_name(st_name, i) == name:
                result.append(i)
                if len(result) > self.max_name_candidates:
                    raise ELFMemoryLimitError(
                        f'name {name!r} matches more than '
                        f'{self.max_name_candidates} symbols',
                        category='symbol-names')
            if offsets is None and i + 1 >= count:
                break
        return tuple(result)

    # -- Public API ---------------------------------------------------------

    def get_symbol(self, n: int) -> Symbol:
        """ Get the symbol at index #n from the table (Symbol object).

            Recently requested entries are kept in a bounded LRU cache, so a
            repeated request does not parse the entry again until it is
            evicted.
        """
        if not 0 <= n < self.num_symbols():
            raise IndexError(n)
        return self._cached_symbol(n)

    def get_symbol_by_name(self, name: str) -> list[Symbol] | None:
        """ Get a symbol(s) by name. Return None if no symbol by the given name
            exists.

            The table is scanned through its st_name fields; only entries whose
            name matches are parsed and only the strings those entries
            reference are read. The resulting index list is kept in a bounded
            LRU cache, making a repeated query reparse nothing that is cached.
        """
        indices = self._cache_get(self._name_index_cache, name)
        if indices is None:
            indices = self._indices_by_name(name)
            self._cache_put(self._name_index_cache, name, indices,
                            self.max_cached_name_lookups)
        if not indices:
            return None
        return [self._cached_symbol(i) for i in indices]

    def iter_symbols(self, start: int = 0, stop: int | None = None) -> Iterator[Symbol]:
        """ Yield symbols lazily, one entry at a time.

            With the optional range, only entries ``[start, stop)`` are parsed;
            nothing outside the range is read. The iterator keeps no stream
            cursor of its own, so several iterators over the same file (or the
            same section) can be interleaved without disturbing each other.

            Iteration does not populate the LRU caches, so walking a table of
            millions of entries has constant memory cost.
        """
        count = self.num_symbols()
        if stop is None:
            stop = count
        start = max(0, start)
        stop = min(stop, count)
        for i in range(start, stop):
            yield self._parse_symbol(i)


class SymbolTableIndexSection(Section):
    """ A section containing the section header table indices corresponding
        to symbols in the linked symbol table. This section has to exist if the
        symbol table contains an entry with a section header index set to
        SHN_XINDEX (0xffff). The format of the section is described at
        https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.sheader.html
    """
    def __init__(
        self,
        header: Container,
        name: str,
        elffile: ELFFile,
        symboltable: Container,
    ) -> None:
        super().__init__(header, name, elffile)
        self.symboltable = symboltable

    def get_section_index(self, n: int) -> int:
        """ Get the section header table index for the symbol with index #n.
            The section contains an array of Elf32_word values with one entry
            for every symbol in the associated symbol table.
        """
        return struct_parse(self.elffile.structs.Elf_word(''), self.stream,
                            self['sh_offset'] + n * self['sh_entsize'])


class SymbolTableSection(_BoundedSymbolAccess, Section):
    """ ELF symbol table section. Has an associated StringTableSection that's
        passed in the constructor.

        Access is bounded-memory: entries are parsed on demand, iteration
        streams, and name lookups only parse the entries they return. The
        bounds are per-instance class attributes; set them before creating the
        section or on the class to suit the workload.
    """
    def __init__(
        self,
        header: Container,
        name: str,
        elffile: ELFFile,
        stringtable: StringTableSection,
    ) -> None:
        super().__init__(header, name, elffile)
        self.stringtable = stringtable
        elf_assert(self['sh_entsize'] > 0,
                f'Expected entry size of section {name!r} to be > 0')
        elf_assert(self['sh_size'] % self['sh_entsize'] == 0,
                f'Expected section size to be a multiple of entry size in section {name!r}')
        self._init_symbol_caches()

    def num_symbols(self) -> int:
        """ Number of symbols in the table
        """
        return self['sh_size'] // self['sh_entsize']

    def _symbol_table_bounds(self) -> tuple[int, int]:
        return self['sh_offset'], self['sh_size']

    def _symbol_entry_size(self) -> int:
        return int(self['sh_entsize'])

    def _symbol_struct(self) -> Any:
        return self.structs.Elf_Sym

    def _symbol_structs(self) -> ELFStructs:
        return self.structs

    def _symbol_stream(self) -> IO[bytes]:
        return self.stream

    def _symbol_stream_owner(self) -> Any:
        return self.elffile

    def _symbol_stringtable(self) -> StringTableSection:
        return self.stringtable


class Symbol:
    """ Symbol object - representing a single symbol entry from a symbol table
        section.

        Similarly to Section objects, allows dictionary-like access to the
        symbol entry.
    """
    def __init__(self, entry: Container, name: str) -> None:
        self.entry = entry
        self.name = name

    def __getitem__(self, name: str) -> Any:
        """ Implement dict-like access to entries
        """
        return self.entry[name]

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.name!r} "
                f"value={self['st_value']:#x} size={self['st_size']:#x} "
                f"type={self['st_info']['type']!r}>")


class SUNWSyminfoTableSection(Section):
    """ ELF .SUNW Syminfo table section.
        Has an associated SymbolTableSection that's passed in the constructor.
    """
    def __init__(
        self,
        header: Container,
        name: str,
        elffile: ELFFile,
        symboltable: SymbolTableSection,
    ) -> None:
        super().__init__(header, name, elffile)
        self.symboltable = symboltable

    def num_symbols(self) -> int:
        """ Number of symbols in the table
        """
        return self['sh_size'] // self['sh_entsize'] - 1

    def get_symbol(self, n: int) -> Symbol:
        """ Get the symbol at index #n from the table (Symbol object).
            It begins at 1 and not 0 since the first entry is used to
            store the current version of the syminfo table.
        """
        # Grab the symbol's entry from the stream
        entry_offset = self['sh_offset'] + n * self['sh_entsize']
        entry = struct_parse(
            self.structs.Elf_Sunw_Syminfo,
            self.stream,
            stream_pos=entry_offset)
        # Find the symbol name in the associated symbol table
        name = self.symboltable.get_symbol(n).name
        return Symbol(entry, name)

    def iter_symbols(self) -> Iterator[Symbol]:
        """ Yield all the symbols in the table
        """
        for i in range(1, self.num_symbols() + 1):
            yield self.get_symbol(i)


class NoteSection(Section):
    """ ELF NOTE section. Knows how to parse notes.
    """
    def iter_notes(self) -> Iterator[Container]:
        """ Yield all the notes in the section.  Each result is a dictionary-
            like object with "n_name", "n_type", and "n_desc" fields, amongst
            others.
        """
        return iter_notes(self.elffile, self['sh_offset'], self['sh_size'])


class StabSection(Section):
    """ ELF stab section.
    """
    def iter_stabs(self) -> Iterator[Container]:
        """ Yield all stab entries.  Result type is ELFStructs.Elf_Stabs.
        """
        offset: int = self['sh_offset']
        size: int = self['sh_size']
        end = offset + size
        while offset < end:
            stabs = struct_parse(
                self.structs.Elf_Stabs,
                self.stream,
                stream_pos=offset)
            stabs['n_offset'] = offset
            offset += self.structs.Elf_Stabs.sizeof()
            self.stream.seek(offset)
            yield stabs

class Attribute:
    """ Attribute object - representing a build attribute of ELF files.
    """
    if TYPE_CHECKING:
        value: Any

    def __init__(self, structs: ELFStructs, stream: IO[bytes]) -> None:
        self._tag = self._parse(structs, stream)
        # Section/symbol indices for TAG_SECTION/TAG_SYMBOL; otherwise empty.
        self.extra: list[int] = []

    @classmethod
    def _parse(cls, structs: ELFStructs, stream: IO[bytes]) -> Container:
        raise NotImplementedError

    @property
    def tag(self) -> str:
        # ARM and RISC-V attribute enums reject unknown values rather than
        # passing them through, so these tags are always symbolic names.
        return cast(str, self._tag['tag'])

    def __repr__(self) -> str:
        s = f'<{self.__class__.__name__} ({self.tag}): {self.value!r}>'
        s += f' {self.extra}' if self.extra else ''
        return s


class AttributesSubsubsection(Section):
    """ Subsubsection of an ELF attribute section's subsection.
    """
    attribute: type[Attribute]

    def __init__(self, stream: IO[bytes], structs: ELFStructs, offset: int) -> None:
        self.stream = stream
        self.offset = offset
        self.structs = structs

        self.header: Attribute = self.attribute(self.structs, self.stream)  # type: ignore[assignment]

        self.attr_start = self.stream.tell()

    def iter_attributes(self, tag: str | None = None) -> Iterator[Attribute]:
        """ Yield all attributes (limit to |tag| if specified).
        """
        for attribute in self._make_attributes():
            if tag is None or attribute.tag == tag:
                yield attribute

    @property
    def num_attributes(self) -> int:
        """ Number of attributes in the subsubsection.
        """
        return sum(1 for _ in self.iter_attributes()) + 1

    @property
    def attributes(self) -> list[Attribute]:
        """ List of all attributes in the subsubsection.
        """
        return [self.header, *(self.iter_attributes())]

    def _make_attributes(self) -> Iterator[Attribute]:
        """ Create all attributes for this subsubsection except the first one
            which is the header.
        """
        end = self.offset + self.header.value

        self.stream.seek(self.attr_start)

        while self.stream.tell() != end:
            yield self.attribute(self.structs, self.stream)

    def __repr__(self) -> str:
        s = "<%s (%s): %d bytes>"
        return s % (self.__class__.__name__,
                    self.header.tag[4:], self.header.value)


class AttributesSubsection(Section):
    """ Subsection of an ELF attributes section.
    """
    subsubsection = AttributesSubsubsection

    def __init__(self, stream: IO[bytes], structs: ELFStructs, offset: int) -> None:
        self.stream = stream
        self.offset = offset
        self.structs = structs

        self.header: Container = struct_parse(structs.Elf_Attr_Subsection_Header, self.stream, self.offset)

        self.subsubsec_start = self.stream.tell()

    def iter_subsubsections(self, scope: str | None = None) -> Iterator[AttributesSubsubsection]:
        """ Yield all subsubsections (limit to |scope| if specified).
        """
        for subsubsec in self._make_subsubsections():
            if scope is None or subsubsec.header.tag == scope:
                yield subsubsec

    @property
    def num_subsubsections(self) -> int:
        """ Number of subsubsections in the subsection.
        """
        return sum(1 for _ in self.iter_subsubsections())

    @property
    def subsubsections(self) -> list[AttributesSubsubsection]:
        """ List of all subsubsections in the subsection.
        """
        return list(self.iter_subsubsections())

    def _make_subsubsections(self) -> Iterator[AttributesSubsubsection]:
        """ Create all subsubsections for this subsection.
        """
        end = self.offset + self['length']

        self.stream.seek(self.subsubsec_start)

        while self.stream.tell() != end:
            subsubsec = self.subsubsection(self.stream,
                                           self.structs,
                                           self.stream.tell())
            self.stream.seek(self.subsubsec_start + subsubsec.header.value)
            yield subsubsec

    def __getitem__(self, name: str) -> Any:
        """ Implement dict-like access to header entries.
        """
        return self.header[name]

    def __repr__(self) -> str:
        s = "<%s (%s): %d bytes>"
        return s  % (self.__class__.__name__,
                     self.header['vendor_name'], self.header['length'])


class AttributesSection(Section):
    """ ELF attributes section.
    """
    subsection = AttributesSubsection

    def __init__(self, header: Container, name: str, elffile: ELFFile) -> None:
        super().__init__(header, name, elffile)

        fv: int = struct_parse(self.structs.Elf_byte('format_version'),
                          self.stream,
                          self['sh_offset'])

        elf_assert(chr(fv) == 'A',
                   f"Unknown attributes version {chr(fv)}, expecting 'A'.")

        self.subsec_start = self.stream.tell()

    def iter_subsections(self, vendor_name: str | None = None) -> Iterator[AttributesSubsection]:
        """ Yield all subsections (limit to |vendor_name| if specified).
        """
        for subsec in self._make_subsections():
            if vendor_name is None or subsec['vendor_name'] == vendor_name:
                yield subsec

    @property
    def num_subsections(self) -> int:
        """ Number of subsections in the section.
        """
        return sum(1 for _ in self.iter_subsections())

    @property
    def subsections(self) -> list[AttributesSubsection]:
        """ List of all subsections in the section.
        """
        return list(self.iter_subsections())

    def _make_subsections(self) -> Iterator[AttributesSubsection]:
        """ Create all subsections for this section.
        """
        end = self['sh_offset'] + self.data_size

        self.stream.seek(self.subsec_start)

        while self.stream.tell() != end:
            subsec = self.subsection(self.stream,
                                     self.structs,
                                     self.stream.tell())
            self.stream.seek(self.subsec_start + subsec['length'])
            yield subsec


class ARMAttribute(Attribute):
    """ ARM attribute object - representing a build attribute of ARM ELF files.

        vendor_name holds the vendor for TAG_COMPATIBILITY, or None for other
        tags.
    """

    @classmethod
    def _parse(cls, structs: ELFStructs, stream: IO[bytes]) -> Container:
        return struct_parse(structs.Elf_Arm_Attribute_Tag, stream)

    def __init__(self, structs: ELFStructs, stream: IO[bytes]) -> None:
        super().__init__(structs, stream)
        self.vendor_name: str | None = None

        if self.tag in ('TAG_FILE', 'TAG_SECTION', 'TAG_SYMBOL'):
            self.value = struct_parse(structs.Elf_word('value'), stream)

            if self.tag != 'TAG_FILE':
                s_number: int = struct_parse(structs.Elf_uleb128('s_number'), stream)

                while s_number != 0:
                    self.extra.append(s_number)
                    s_number = struct_parse(structs.Elf_uleb128('s_number'),
                                            stream)

        elif self.tag in ('TAG_CPU_RAW_NAME', 'TAG_CPU_NAME', 'TAG_CONFORMANCE'):
            self.value = struct_parse(structs.Elf_ntbs('value',
                                                       encoding='utf-8'),
                                      stream)

        elif self.tag == 'TAG_COMPATIBILITY':
            self.value = struct_parse(structs.Elf_uleb128('value'), stream)
            self.vendor_name = struct_parse(
                structs.Elf_ntbs('vendor_name', encoding='utf-8'), stream)

        elif self.tag == 'TAG_ALSO_COMPATIBLE_WITH':
            self.value = ARMAttribute(structs, stream)

            if type(self.value.value) is not str:
                nul: int = struct_parse(structs.Elf_byte('nul'), stream)
                elf_assert(nul == 0,
                           f"Invalid terminating byte {nul!r}, expecting NUL.")

        else:
            self.value = struct_parse(structs.Elf_uleb128('value'), stream)

    def __repr__(self) -> str:
        s = super().__repr__()
        return s + f' {self.vendor_name}' if self.vendor_name is not None else s


class ARMAttributesSubsubsection(AttributesSubsubsection):
    """ Subsubsection of an ELF .ARM.attributes section's subsection.
    """
    attribute = ARMAttribute


class ARMAttributesSubsection(AttributesSubsection):
    """ Subsection of an ELF .ARM.attributes section.
    """
    subsubsection = ARMAttributesSubsubsection


class ARMAttributesSection(AttributesSection):
    """ ELF .ARM.attributes section.
    """
    subsection = ARMAttributesSubsection


class RISCVAttribute(Attribute):
    """ Attribute of an ELF .riscv.attributes section.
    """

    @classmethod
    def _parse(cls, structs: ELFStructs, stream: IO[bytes]) -> Container:
        return struct_parse(structs.Elf_RiscV_Attribute_Tag, stream)

    def __init__(self, structs: ELFStructs, stream: IO[bytes]) -> None:
        super().__init__(structs, stream)

        if self.tag in ('TAG_FILE', 'TAG_SECTION', 'TAG_SYMBOL'):
            self.value = struct_parse(structs.Elf_word('value'), stream)

            if self.tag != 'TAG_FILE':
                s_number: int = struct_parse(structs.Elf_uleb128('s_number'), stream)

                while s_number != 0:
                    self.extra.append(s_number)
                    s_number = struct_parse(structs.Elf_uleb128('s_number'),
                                            stream)

        elif self.tag == 'TAG_ARCH':
            self.value = struct_parse(structs.Elf_ntbs('value',
                                                       encoding='utf-8'),
                                      stream)

        else:
            self.value = struct_parse(structs.Elf_uleb128('value'), stream)


class RISCVAttributesSubsubsection(AttributesSubsubsection):
    """ Subsubsection of an ELF .riscv.attributes subsection.
    """
    attribute = RISCVAttribute


class RISCVAttributesSubsection(AttributesSubsection):
    """ Subsection of an ELF .riscv.attributes section.
    """
    subsubsection = RISCVAttributesSubsubsection


class RISCVAttributesSection(AttributesSection):
    """ ELF .riscv.attributes section.
    """
    subsection = RISCVAttributesSubsection
