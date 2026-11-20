#-------------------------------------------------------------------------------
# elftools tests
#
# Tests for bounded-memory symbol table access: streaming iteration, name
# lookup that only parses matching entries, bounded LRU caches, independent
# readers, early-exit cleanup, and per-entry error reporting.
#
# This code is in the public domain
#-------------------------------------------------------------------------------
"""Synthetic ELF64 files with large symbol tables.

The builder writes a minimal little-endian ELF64 with one .symtab, its linked
.strtab, a .shstrtab and a section header table. No compiler is involved, so a
test can create tables with a million entries in a moment.
"""
from __future__ import annotations

import io
import struct
from typing import NamedTuple

ELF_HEADER_SIZE = 64
SECTION_HEADER_SIZE = 64


class SymDef(NamedTuple):
    name: str
    value: int = 0
    size: int = 0
    info: int = 0
    other: int = 0
    shndx: int = 0


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def build_elf64_with_symtab(
    symbols: list[SymDef],
    *,
    debug_sections: bool = False,
    strtab_pad: int = 0,
) -> bytes:
    """ Build an ELF64 executable image carrying `symbols` in .symtab.

        Symbol 0 is always the required null entry. Trailing `strtab_pad`
        zero bytes make the string table much larger than the symbol table,
        modeling a pathological layout without allocating it up front.
    """
    if not symbols or symbols[0] != SymDef(''):
        symbols = [SymDef('')] + list(symbols)

    strtab = bytearray(b'\x00')
    name_offsets: list[int] = []
    for sym in symbols:
        if sym.name == '':
            name_offsets.append(0)
            continue
        name_offsets.append(len(strtab))
        strtab += sym.name.encode('utf-8') + b'\x00'
    # Trailing padding counts toward sh_size but is written lazily below.
    strtab_size = len(strtab) + strtab_pad

    shstrtab = bytearray(b'\x00')
    section_names: dict[str, int] = {}

    def add_name(name: str) -> int:
        if name not in section_names:
            section_names[name] = len(shstrtab)
            shstrtab.extend(name.encode() + b'\x00')
        return section_names[name]

    def symtab_bytes() -> bytes:
        out = bytearray()
        for sym, st_name in zip(symbols, name_offsets):
            out += struct.pack('<IBBHQQ',
                               st_name, sym.info, sym.other, sym.shndx,
                               sym.value, sym.size)
        return bytes(out)

    # Section layout (indices are fixed up below).
    #   0: NULL, 1: .symtab, 2: .strtab, 3: .shstrtab, [4: .debug_info]
    num_sections = 5 if debug_sections else 4
    shstrndx = 3
    strtab_index = 2

    sections: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    # name_offset, type, flags, offset, size, link, info, addralign, entsize

    ehdr_end = ELF_HEADER_SIZE
    body_start = _align(ehdr_end, 8)

    body = bytearray()
    cursor = body_start

    def place(data: bytes, alignment: int = 1) -> int:
        nonlocal cursor
        offset = _align(cursor, alignment)
        body.extend(b'\x00' * (offset - cursor))
        body.extend(data)
        cursor = offset + len(data)
        return offset

    symtab_data = symtab_bytes()
    symtab_offset = place(symtab_data, 8)
    strtab_offset = place(bytes(strtab) + b'\x00' * strtab_pad, 1)

    sections.append((0, 0, 0, 0, 0, 0, 0, 0, 0))
    sections.append((add_name('.symtab'), 2, 0, symtab_offset, len(symtab_data),
                     strtab_index, 1, 8, 24))
    sections.append((add_name('.strtab'), 3, 0, strtab_offset,
                     strtab_size, 0, 0, 1, 0))
    sections.append((0, 3, 0, 0, 0, 0, 0, 1, 0))  # .shstrtab, filled below
    if debug_sections:
        sections.append((add_name('.debug_info'), 1, 0, 0,
                         512, 0, 0, 1, 0))
    # Section names are now final; lay out the string table itself.
    sections[3] = (add_name('.shstrtab'), 3, 0, 0, 0, 0, 0, 1, 0)
    shstrtab_offset = place(bytes(shstrtab), 1)
    shstrtab_size = len(shstrtab)
    sections[3] = (sections[3][0], 3, 0, shstrtab_offset, shstrtab_size,
                   0, 0, 1, 0)
    if debug_sections:
        debug_offset = place(b'\x00' * 512, 1)
        sections[4] = (sections[4][0], 1, 0, debug_offset, 512, 0, 0, 1, 0)

    shoff = _align(cursor, 8)

    shdr = bytearray()
    for (sh_name, sh_type, sh_flags, sh_offset, sh_size, sh_link, sh_info,
            sh_addralign, sh_entsize) in sections:
        shdr += struct.pack('<IIQQQQIIQQ', sh_name, sh_type, sh_flags,
                            sh_offset, sh_offset, sh_size, sh_link, sh_info,
                            sh_addralign, sh_entsize)

    ident = b'\x7fELF' + bytes([2, 1, 1, 0]) + b'\x00' * 8
    ehdr = ident + struct.pack('<HHIQQQIHHHHHH',
                               2,         # e_type = ET_EXEC
                               62,        # e_machine = EM_X86_64
                               1,         # e_version
                               0,         # e_entry
                               0,         # e_phoff
                               shoff,     # e_shoff
                               0,         # e_flags
                               ELF_HEADER_SIZE,
                               56,        # e_phentsize
                               0,         # e_phnum
                               SECTION_HEADER_SIZE,
                               num_sections,
                               shstrndx)
    image = bytearray(ehdr)
    image.extend(b'\x00' * (body_start - len(image)))
    image.extend(body)
    image.extend(b'\x00' * (shoff - len(image)))
    image.extend(shdr)
    return bytes(image)


def sym_name(index: int) -> str:
    """ The name `make_symbols` assigns to entry `index`.
    """
    return f'sym_{index:024d}'


def make_symbols(count: int, *, duplicate_every: int = 0,
                 name_width: int = 24) -> list[SymDef]:
    """ Deterministic symbol list with mostly-unique names.

        When `duplicate_every` is positive, every so many entries reuses the
        name of entry 1, producing a large multi-entry result set.
    """
    symbols = [SymDef('')]
    for i in range(1, count):
        if duplicate_every and i % duplicate_every == 0:
            name = 'dup_symbol'
        else:
            name = sym_name(i)
        symbols.append(SymDef(name, value=i * 16, size=8, info=0x12,
                              shndx=1))
    return symbols


class CountingReader(io.BytesIO):
    """ BytesIO that counts seeks, reads and bytes read.

        Used by tests to assert that a query does not parse more of the file
        than it has to.
    """
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.seek_count = 0
        self.read_count = 0
        self.bytes_read = 0
        self.reset_counts()

    def reset_counts(self) -> None:
        self.seek_count = 0
        self.read_count = 0
        self.bytes_read = 0

    def seek(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.seek_count += 1
        return super().seek(*args, **kwargs)

    def read(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        data = super().read(*args, **kwargs)
        self.read_count += 1
        self.bytes_read += len(data)
        return data
