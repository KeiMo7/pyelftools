#-------------------------------------------------------------------------------
# elftools: elf/sections.py
#
# ELF sections
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import annotations

import sys
import zlib
from collections import defaultdict
from functools import cached_property
from typing import IO, TYPE_CHECKING, Any, Literal, cast, overload

from elftools.construct.lib.container import Container

from ..common.exceptions import ELFCompressionError, ELFParseError
from ..common.utils import (
    check_file_range,
    elf_assert,
    parse_cstring_in_range,
    stream_read,
    struct_parse,
)
from .constants import SH_FLAGS
from .notes import iter_notes
from .report import StructureRejection, TableScan

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .elffile import ELFFile
    from .structs import ELFStructs


class Section:
    """ Base class for ELF sections. Also used for all sections types that have
        no special functionality.

        Allows dictionary-like access to the section header. For example:
         > sec = Section(...)
         > sec['sh_type']  # section type
    """
    # Upper bound on a buffer materialized from a declared size that is not
    # backed by file data (NOBITS sections and decompression output). Without
    # it a corrupt size field could make the parser allocate gigabytes from a
    # tiny, truncated sample. Legitimate sections are far smaller; callers
    # that genuinely need more can raise this limit.
    MAX_MATERIALIZED_SIZE = 1 << 30

    def __init__(self, header: Container, name: str | None, elffile: ELFFile) -> None:
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
                raise ELFCompressionError(
                    'Compressed section is too small for its header',
                    structure='compressed section',
                    expected=self.structs.Elf_Chdr.sizeof(),
                    actual=self['sh_size'],
                )
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

        All bounds are checked before a byte is read or an output buffer is
        allocated, and every failure is reported as :class:`ELFParseError`
        (or its :class:`ELFCompressionError` subclass).
        """
        # If this section is NOBITS, there is no data. provide a dummy answer
        if self.header['sh_type'] == 'SHT_NOBITS':
            # A declared size beyond what a bytes object can represent is
            # rejected before any allocation; the natural OverflowError is
            # chained for callers that inspect the cause.
            if self.data_size > sys.maxsize:
                try:
                    b'\0' * self.data_size
                except OverflowError as e:
                    raise ELFParseError(
                        f'NOBITS section size {self.data_size} is too large',
                        structure='NOBITS section',
                        expected=self.data_size,
                        actual=0,
                    ) from e
            # Never materialize more NOBITS data than the parser allows: the
            # declared size is not bounded by the file at all, so a corrupt
            # field must not turn into a huge allocation.
            if self.data_size > self.MAX_MATERIALIZED_SIZE:
                raise ELFParseError(
                    f'NOBITS section size {self.data_size} is too large to '
                    'materialize',
                    structure='NOBITS section',
                    expected=self.data_size,
                    actual=0,
                )
            return b'\0' * self.data_size

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
                    self.stream, self['sh_offset'] + hdr_size,
                    self['sh_size'] - hdr_size, structure='compressed section')

                decomp = zlib.decompressobj()
                # Bound the produced buffer independently of the declared
                # size: a corrupt declaration (or a decompression bomb) must
                # not be allowed to allocate without limit.
                limit = min(self.data_size, self.MAX_MATERIALIZED_SIZE + 1)
                try:
                    result = decomp.decompress(compressed, limit)
                except (zlib.error, OverflowError, ValueError) as e:
                    raise ELFCompressionError(
                        'Compressed section stream is corrupt',
                        structure='compressed section',
                    ) from e
                if len(result) > self.MAX_MATERIALIZED_SIZE:
                    raise ELFCompressionError(
                        f'Decompressed data exceeds the {self.MAX_MATERIALIZED_SIZE}'
                        '-byte materialization limit',
                        structure='compressed section',
                        expected=self._decompressed_size,
                        actual=len(result),
                    )
                # A stream that is cut off before its end marker is corrupt
                # even if the bytes produced so far happen to match the
                # declared size.
                if not decomp.eof:
                    raise ELFCompressionError(
                        'Compressed section stream is truncated',
                        structure='compressed section',
                        expected=self._decompressed_size,
                        actual=len(result),
                    )
            elif isinstance(c_type, int):
                raise ELFCompressionError(
                    f'Unknown compression type: {c_type:#0x}',
                    structure='compressed section',
                )
            else:
                raise ELFCompressionError(
                    f'Unknown compression type: {c_type!r}',
                    structure='compressed section',
                )

            if len(result) != self._decompressed_size:
                raise ELFCompressionError(
                    f'Decompressed data is {len(result)} bytes long, should be {self._decompressed_size} bytes'
                    ' long',
                    structure='compressed section',
                    expected=self._decompressed_size,
                    actual=len(result),
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
    def get_string(self, offset: int) -> str:
        """ Get the string stored at the given offset in this string table.

        Raises :class:`ELFParseError` if ``offset`` is outside the table or
        the string is not terminated inside it. The lookup never reads past
        the end of the table, so a corrupt offset cannot pick up bytes from an
        adjacent section.
        """
        table_offset = self['sh_offset']
        table_end = table_offset + self['sh_size']
        if offset < 0 or table_offset + offset >= table_end:
            raise ELFParseError(
                f'string offset {offset} is outside the string table of '
                f'{self["sh_size"]} bytes',
                structure='string table',
                index=offset,
                expected=self['sh_size'],
                actual=0,
            )
        s = parse_cstring_in_range(
            self.stream, table_offset + offset, table_end,
            structure='string table', index=offset)
        if s is None:
            raise ELFParseError(
                f'string at offset {offset} is not terminated within the '
                f'string table of {self["sh_size"]} bytes',
                structure='string table',
                index=offset,
                expected=self['sh_size'],
                actual=table_end - (table_offset + offset),
            )
        return s.decode('utf-8', errors='replace')

    def get_string_lenient(self, offset: int) -> str | None:
        """ Like :meth:`get_string`, but return None instead of raising when
            the string cannot be resolved.

            An empty string is a real, empty name; None means the name is
            unavailable. Callers can therefore tell ``"no name"`` from ``"name
            could not be read"``.
        """
        table_offset = self['sh_offset']
        table_end = table_offset + self['sh_size']
        if offset < 0 or table_offset + offset >= table_end:
            return None
        s = parse_cstring_in_range(
            self.stream, table_offset + offset, table_end,
            structure='string table', index=offset)
        if s is None:
            return None
        return s.decode('utf-8', errors='replace')


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
        name: str | None,
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
        if self['sh_entsize'] > 0:
            count = self['sh_size'] // self['sh_entsize']
        else:
            count = 0
        if n < 0 or n >= count:
            raise ELFParseError(
                f'section index entry {n} is outside the {count} entries in '
                'the table',
                structure='symbol table index section',
                index=n,
                first_invalid_index=count,
                declared_count=count,
                available_count=count,
            )
        return struct_parse(self.elffile.structs.Elf_word(''), self.stream,
                            self['sh_offset'] + n * self['sh_entsize'])


class SymbolTableSection(Section):
    """ ELF symbol table section. Has an associated StringTableSection that's
        passed in the constructor.
    """
    def __init__(
        self,
        header: Container,
        name: str | None,
        elffile: ELFFile,
        stringtable: StringTableSection,
    ) -> None:
        super().__init__(header, name, elffile)
        self.stringtable = stringtable
        if self['sh_entsize'] == 0:
            raise ELFParseError(
                f'symbol table section {name!r} declares an entry size of 0',
                structure='symbol table',
                expected=1,
                actual=0,
            )
        if self['sh_size'] % self['sh_entsize'] != 0:
            raise ELFParseError(
                f'symbol table section {name!r} has a size of '
                f'{self["sh_size"]} bytes that is not a multiple of its '
                f'{self["sh_entsize"]}-byte entry size',
                structure='symbol table',
                expected=self['sh_entsize'],
                actual=self['sh_size'] % self['sh_entsize'],
            )

    def num_symbols(self) -> int:
        """ Number of symbols in the table
        """
        return self['sh_size'] // self['sh_entsize']

    def get_symbol(self, n: int) -> Symbol:
        """ Get the symbol at index #n from the table (Symbol object)

            Raises :class:`ELFParseError` if #n is outside the table, before
            reading anything: an out-of-range index must never turn adjacent
            section bytes into a plausible-looking symbol.
        """
        num_symbols = self.num_symbols()
        if n < 0 or n >= num_symbols:
            raise ELFParseError(
                f'symbol index {n} is outside the {num_symbols} symbols in '
                'the table',
                structure='symbol table',
                index=n,
                first_invalid_index=num_symbols,
                declared_count=num_symbols,
                available_count=num_symbols,
            )
        # Grab the symbol's entry from the stream, after checking that the
        # whole entry lies inside both the table and the file.
        entry_size = self.structs.Elf_Sym.sizeof()
        entry_offset = self['sh_offset'] + n * self['sh_entsize']
        table_end = self['sh_offset'] + self['sh_size']
        if entry_offset + entry_size > table_end:
            raise ELFParseError(
                f'symbol {n} at offset {entry_offset} does not fit in the '
                f'{self["sh_size"]}-byte symbol table',
                structure='symbol table',
                index=n,
                expected=entry_size,
                actual=table_end - entry_offset,
            )
        file_len = getattr(self.elffile, 'stream_len', None)
        if file_len is not None and entry_offset + entry_size > file_len:
            raise ELFParseError(
                f'symbol {n} at offset {entry_offset} is truncated: expected '
                f'{entry_size} bytes, found {max(0, file_len - entry_offset)}',
                structure='symbol table',
                index=n,
                expected=entry_size,
                actual=max(0, file_len - entry_offset),
            )
        entry = struct_parse(
            self.structs.Elf_Sym,
            self.stream,
            stream_pos=entry_offset)
        # Find the symbol name in the associated string table. A name that
        # cannot be resolved is reported as None, never as an empty string.
        name = self.stringtable.get_string_lenient(entry['st_name'])
        return Symbol(entry, name)

    def get_symbol_by_name(self, name: str) -> list[Symbol] | None:
        """ Get a symbol(s) by name. Return None if no symbol by the given name
            exists.
        """
        symnums = self._symbol_name_map.get(name)
        return [self.get_symbol(i) for i in symnums] if symnums else None

    @cached_property
    def _symbol_name_map(self) -> dict[str, list[int]]:
        smap: dict[str, list[int]] = defaultdict(list)
        for i, sym in enumerate(self.iter_symbols()):
            if sym.name is not None:
                smap[sym.name].append(i)
        return smap

    def iter_symbols_lenient(self) -> TableScan[Symbol]:
        """ Enumerate every usable symbol, together with the reason each
            rejected one is unavailable.

            A single damaged entry does not abort the scan; the returned
            :class:`TableScan` yields the usable symbols and records one
            :class:`StructureRejection` per rejected index.
        """
        declared = self.num_symbols()
        symbols: list[Symbol] = []
        rejections: list[StructureRejection] = []
        for i in range(declared):
            try:
                symbols.append(self.get_symbol(i))
            except ELFParseError as error:
                rejections.append(StructureRejection(
                    structure='symbol table', index=i, error=error))
        return TableScan(tuple(symbols), tuple(rejections), declared, declared)

    def iter_symbols(self) -> Iterator[Symbol]:
        """ Yield all the symbols in the table
        """
        for i in range(self.num_symbols()):
            yield self.get_symbol(i)


class Symbol:
    """ Symbol object - representing a single symbol entry from a symbol table
        section.

        Similarly to Section objects, allows dictionary-like access to the
        symbol entry.
    """
    def __init__(self, entry: Container, name: str | None) -> None:
        self.entry = entry
        # None means the name could not be resolved; '' is a real empty name.
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
        name: str | None,
        elffile: ELFFile,
        symboltable: SymbolTableSection,
    ) -> None:
        super().__init__(header, name, elffile)
        self.symboltable = symboltable

    def num_symbols(self) -> int:
        """ Number of symbols in the table
        """
        if self['sh_entsize'] == 0:
            return 0
        return self['sh_size'] // self['sh_entsize'] - 1

    def get_symbol(self, n: int) -> Symbol:
        """ Get the symbol at index #n from the table (Symbol object).
            It begins at 1 and not 0 since the first entry is used to
            store the current version of the syminfo table.
        """
        num_symbols = self.num_symbols()
        if n < 1 or n > num_symbols:
            raise ELFParseError(
                f'syminfo index {n} is outside the {num_symbols} entries in '
                'the table',
                structure='syminfo table',
                index=n,
                first_invalid_index=num_symbols + 1,
                declared_count=num_symbols,
                available_count=num_symbols,
            )
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
        entry_size = self.structs.Elf_Stabs.sizeof()
        file_len = getattr(self.elffile, 'stream_len', None)
        if file_len is not None and offset + size > file_len:
            raise ELFParseError(
                f'stab section of {size} bytes at offset {offset} extends '
                f'past the end of the file',
                structure='stab section',
                expected=size,
                actual=max(0, file_len - offset),
            )
        end = offset + size
        while offset < end:
            if entry_size <= 0 or offset + entry_size > end:
                raise ELFParseError(
                    f'stab entry at offset {offset} does not fit in the '
                    f'{size}-byte section',
                    structure='stab section',
                    expected=entry_size,
                    actual=end - offset,
                )
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

    def __init__(self, header: Container, name: str | None, elffile: ELFFile) -> None:
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
