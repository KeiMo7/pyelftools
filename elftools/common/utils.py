#-------------------------------------------------------------------------------
# elftools: common/utils.py
#
# Miscellaneous utilities for elftools
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import IO, TYPE_CHECKING, Any, TypeVar, overload

from ..construct import ConstructError
from .exceptions import DWARFError, ELFError, ELFParseError

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from ..construct import Construct, FormatField
    from ..dwarf.dwarfinfo import DebugSectionDescriptor
    from .construct_utils import SLEB128, ULEB128, UBInt24, ULInt24

    _T = TypeVar("_T")
    _K = TypeVar("_K")
    _V = TypeVar("_V")


def merge_dicts(*dicts: Mapping[_K, _V]) -> dict[_K, _V]:
    """Given any number of dicts, merges them into a new one."""
    result: dict[_K, _V] = {}
    for d in dicts:
        result.update(d)
    return result

def bytes2str(b: bytes) -> str:
    """Decode a bytes object into a string."""
    return b.decode('latin-1')


# Use @overload to get more specific type, e.g. [SU][BLN]{EB,Int}{8,16,24,32,64,128} -> int
@overload
def struct_parse(struct: FormatField[_T] | ULEB128 | SLEB128 | UBInt24 | ULInt24, stream: IO[bytes], stream_pos: int | None = ...) -> _T: ...
@overload
def struct_parse(struct: Construct, stream: IO[bytes], stream_pos: int | None = ...) -> Any: ...
def struct_parse(struct: Construct, stream: IO[bytes], stream_pos: int | None = None) -> Any:
    """ Convenience function for using the given struct to parse a stream.
        If stream_pos is provided, the stream is seeked to this position before
        the parsing is done. Otherwise, the current position of the stream is
        used.
        Wraps invalid seek positions and Construct errors with ELFParseError.
    """
    if stream_pos is not None:
        try:
            stream.seek(stream_pos)
        except (OverflowError, ValueError, OSError) as e:
            raise ELFParseError(f'Invalid seek offset {stream_pos}') from e
    try:
        return struct.parse_stream(stream)
    except ConstructError as e:
        raise ELFParseError(str(e))


def parse_cstring_from_stream(stream: IO[bytes], stream_pos: int | None = None) -> bytes | None:
    """ Parse a C-string from the given stream. The string is returned without
        the terminating \x00 byte. If the terminating byte wasn't found, None
        is returned (the stream is exhausted).
        If stream_pos is provided, the stream is seeked to this position before
        the parsing is done. Otherwise, the current position of the stream is
        used.
        Note: a bytes object is returned here, because this is what's read from
        the binary file.
    """
    if stream_pos is not None:
        try:
            stream.seek(stream_pos)
        except (OverflowError, ValueError, OSError) as e:
            raise ELFParseError(f'Invalid seek offset {stream_pos}') from e
    CHUNKSIZE = 64
    chunks = []
    while True:
        try:
            chunk, sep, _tail = stream.read(CHUNKSIZE).partition(b'\x00')
        except (OverflowError, ValueError, OSError) as e:
            raise ELFParseError('Could not read a string from the stream') from e
        chunks.append(chunk)
        if sep:
            return b''.join(chunks)
        if len(chunk) < CHUNKSIZE:
            return None


def parse_cstring_in_range(
    stream: IO[bytes],
    start: int,
    end: int,
    *,
    structure: str = 'string table',
    index: int | None = None,
) -> bytes | None:
    """ Parse a NUL-terminated string that must lie entirely in [start, end).

        This never reads past ``end``, so a corrupt string offset cannot pick
        up bytes belonging to an adjacent section.  Returns None if ``start``
        is outside the range or the string is not terminated within it.
    """
    if start < 0 or start >= end:
        return None
    try:
        stream.seek(start)
        chunks = []
        remaining = end - start
        while remaining > 0:
            chunk = stream.read(min(64, remaining))
            if not chunk:
                return None
            remaining -= len(chunk)
            head, sep, _tail = chunk.partition(b'\x00')
            chunks.append(head)
            if sep:
                return b''.join(chunks)
        return None
    except (OverflowError, ValueError, OSError) as e:
        raise ELFParseError(
            'Could not read a string from the stream',
            structure=structure,
            index=index,
        ) from e


def check_file_range(
    offset: int,
    size: int,
    file_size: int,
    *,
    structure: str | None = None,
    index: int | None = None,
) -> None:
    """ Reject a byte range outside the file before seeking or allocating.
    """
    if offset < 0 or size < 0 or offset > file_size or size > file_size - offset:
        what = structure or 'file range'
        raise ELFParseError(
            f'{what} at offset {offset} with size {size} exceeds file size {file_size}',
            structure=structure,
            index=index,
            expected=size if size >= 0 else None,
            actual=max(0, file_size - offset) if offset >= 0 else 0,
        )


def stream_read(
    stream: IO[bytes],
    offset: int,
    size: int,
    *,
    structure: str | None = None,
    index: int | None = None,
) -> bytes:
    """ Read an entire file-backed range, checking its bounds before reading.
    """
    try:
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()
    except (OverflowError, ValueError, OSError) as e:
        raise ELFParseError('Could not determine the stream size') from e
    check_file_range(offset, size, file_size, structure=structure, index=index)
    try:
        stream.seek(offset)
        data = stream.read(size)
    except (OverflowError, ValueError, OSError) as e:
        raise ELFParseError(
            f'Could not read {size} bytes at offset {offset}',
            structure=structure,
            index=index,
        ) from e
    if len(data) != size:
        raise ELFParseError(
            f'Expected {size} bytes at offset {offset}, found {len(data)}',
            structure=structure,
            index=index,
            expected=size,
            actual=len(data),
        )
    return data


def elf_assert(cond: object, msg: str = '') -> None:
    """ Assert that cond is True, otherwise raise ELFError(msg)
    """
    _assert_with_exception(cond, msg, ELFError)


def dwarf_assert(cond: object, msg: str = '') -> None:
    """ Assert that cond is True, otherwise raise DWARFError(msg)
    """
    _assert_with_exception(cond, msg, DWARFError)


@contextmanager
def preserve_stream_pos(stream: IO[bytes]) -> Generator[None, None, None]:
    """ Usage:
        # stream has some position FOO (return value of stream.tell())
        with preserve_stream_pos(stream):
            # do stuff that manipulates the stream
        # stream still has position FOO
    """
    saved_pos = stream.tell()
    yield
    stream.seek(saved_pos)


def roundup(num: int, bits: int) -> int:
    """ Round up a number to nearest multiple of 2^bits. The result is a number
        where the least significant bits passed in bits are 0.
    """
    return (num - 1 | (1 << bits) - 1) + 1


def save_dwarf_section(section: DebugSectionDescriptor, filename: str) -> None:
    """Debug helper: dump section contents into a file
    Section is expected to be one of the debug_xxx_sec elements of DWARFInfo
    """
    stream = section.stream
    with preserve_stream_pos(stream), open(filename, 'wb') as file:
        stream.seek(0, os.SEEK_SET)
        data = stream.read(section.size)
        file.write(data)


#------------------------- PRIVATE -------------------------

def _assert_with_exception(cond: object, msg: str, exception_type: type[BaseException]) -> None:
    if not cond:
        raise exception_type(msg)
