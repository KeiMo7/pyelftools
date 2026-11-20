#-------------------------------------------------------------------------------
# elftools: common/compression.py
#
# Bounded, explainable decompression of compressed ELF sections.
#
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import annotations

import enum
import zlib
from typing import IO, TYPE_CHECKING

from .exceptions import ELFCompressionError, ELFParseError
from .utils import check_file_range

if TYPE_CHECKING:
    from collections.abc import Iterator


# Default upper bound for the number of bytes a single section may expand to.
# 256 MiB is far above real debug sections but small enough that a malicious
# compression bomb is rejected without exhausting the process.  Callers can
# lower or raise it per section (see Section.data) or per file (see
# ELFFile.decompression_limit).
DEFAULT_MAX_DECOMPRESSED_SIZE = 256 * 1024 * 1024

# Decompression advances in bounded steps: input is read from the file in
# chunks of at most DECOMPRESS_INPUT_CHUNK_SIZE bytes and each call to zlib is
# allowed to produce at most DECOMPRESS_OUTPUT_CHUNK_SIZE bytes.  Neither an
# input buffer nor an output buffer is ever sized from the declared length.
DECOMPRESS_INPUT_CHUNK_SIZE = 16 * 1024
DECOMPRESS_OUTPUT_CHUNK_SIZE = 64 * 1024


class CompressionFailure(str, enum.Enum):
    """Machine-readable failure categories.

    These are attached to ELFCompressionError instances as ``category`` rather
    than introducing new exception classes, so existing ``except
    ELFCompressionError`` handlers keep working.  Being ``str`` subclasses, the
    members also compare equal to their textual value.
    """
    #: The compression algorithm identifier is not one we can decompress.
    UNSUPPORTED_ALGORITHM = 'unsupported-algorithm'
    #: ch_size is unreasonably large; rejected before any allocation.
    DECLARED_SIZE_TOO_LARGE = 'declared-size-too-large'
    #: Decompressed output crossed the configured limit (compression bomb).
    OUTPUT_LIMIT_EXCEEDED = 'output-limit-exceeded'
    #: The stream decompressed to more than ch_size bytes.
    DATA_LONGER_THAN_DECLARED = 'data-longer-than-declared'
    #: The stream decompressed to fewer than ch_size bytes.
    DATA_SHORTER_THAN_DECLARED = 'data-shorter-than-declared'
    #: Exactly ch_size bytes were produced, but data follows the stream.
    TRAILING_DATA = 'trailing-data'
    #: zlib reported corrupt data in the middle of the stream.
    CORRUPT_STREAM = 'corrupt-stream'
    #: The compressed stream ends before its end marker is seen.
    INCOMPLETE_STREAM = 'incomplete-stream'
    #: ch_size is zero while compressed bytes are present.
    EMPTY_DECLARED_SIZE = 'empty-declared-size'


def compression_error(
    category: CompressionFailure,
    message: str,
    **details: object,
) -> ELFCompressionError:
    """ Build an ELFCompressionError carrying a failure category and details.

    The details become attributes of the exception (e.g. ``section``,
    ``declared_size``, ``decompressed_size``, ``max_output_size``,
    ``missing_bytes``, ``trailing_bytes``, ``compressed_offset``).
    """
    error = ELFCompressionError(message)
    error.category = category
    for name, value in details.items():
        setattr(error, name, value)
    return error


def unsupported_algorithm_error(
    algorithm: int | str | bytes,
    section_label: str,
) -> ELFCompressionError:
    """ Report a compression algorithm we cannot decompress.

    The message carries both the algorithm identifier and the section
    location; the same values are available as the ``algorithm`` and
    ``section`` attributes.
    """
    shown = f'{algorithm:#0x}' if isinstance(algorithm, int) else repr(algorithm)
    return compression_error(
        CompressionFailure.UNSUPPORTED_ALGORITHM,
        f'Unknown compression type: {shown} in {section_label}',
        algorithm=algorithm,
        section=section_label)


def decompress_zlib_section(
    stream: IO[bytes],
    offset: int,
    size: int,
    declared_size: int,
    *,
    section_label: str,
    max_output_size: int = DEFAULT_MAX_DECOMPRESSED_SIZE,
) -> bytes:
    """ Decompress exactly ``size`` zlib bytes at file ``offset``.

    The result must be exactly ``declared_size`` bytes long and the compressed
    stream must end exactly at ``offset + size``; anything else raises
    ELFCompressionError with a distinct ``category``.  Work is performed in
    bounded chunks and the returned bytes are only produced on full success.

    ``section_label`` locates the section in error messages; the same string is
    stored on the exception as ``section``.
    """
    if declared_size < 0 or size < 0 or offset < 0:
        raise compression_error(
            CompressionFailure.CORRUPT_STREAM,
            f'Invalid compressed range in {section_label}: '
            f'offset {offset}, size {size}, declared size {declared_size}',
            section=section_label, declared_size=declared_size,
            compressed_size=size)

    # Reject an absurd declaration before reading or allocating anything.
    if declared_size > max_output_size:
        raise compression_error(
            CompressionFailure.DECLARED_SIZE_TOO_LARGE,
            f'Declared uncompressed size {declared_size} in {section_label} '
            f'exceeds the decompression limit of {max_output_size} bytes',
            section=section_label, declared_size=declared_size,
            max_output_size=max_output_size, decompressed_size=0)

    # Check the file-backed range once, up front; chunks are re-checked as read.
    try:
        stream.seek(0, 2)
        file_size = stream.tell()
    except (OSError, ValueError) as e:
        raise ELFParseError(
            f'Cannot determine file size for {section_label}') from e
    check_file_range(offset, size, file_size)

    if size == 0:
        if declared_size == 0:
            return b''
        # No compressed bytes at all: nothing can provide declared_size bytes.
        raise compression_error(
            CompressionFailure.DATA_SHORTER_THAN_DECLARED,
            f'Decompressed data is 0 bytes long, should be {declared_size} '
            f'bytes long in {section_label} (no compressed data present)',
            section=section_label, declared_size=declared_size,
            decompressed_size=0, missing_bytes=declared_size,
            compressed_size=0)

    if declared_size == 0:
        raise compression_error(
            CompressionFailure.EMPTY_DECLARED_SIZE,
            f'Declared uncompressed size is 0 in {section_label}, '
            f'but {size} compressed bytes are present',
            section=section_label, declared_size=0, compressed_size=size,
            decompressed_size=0)

    decomp = zlib.decompressobj()
    chunks: list[bytes] = []
    produced = 0            # total bytes zlib has emitted
    buffered = 0            # bytes kept for the result (never > declared_size)
    buffering = True        # False once declared_size is exceeded
    fed = 0                 # compressed bytes handed to zlib so far
    reached_eof = False
    unused_at_eof = 0

    for chunk in _iter_file_chunks(stream, offset, size):
        fed += len(chunk)
        data = chunk
        while data:
            # Offset of ``data[0]`` within the compressed section.
            chunk_start = fed - len(data)
            try:
                piece = decomp.decompress(data, DECOMPRESS_OUTPUT_CHUNK_SIZE)
            except zlib.error as e:
                raise compression_error(
                    CompressionFailure.CORRUPT_STREAM,
                    f'Corrupt compressed data in {section_label} at compressed '
                    f'offset {offset + chunk_start:#x} after producing '
                    f'{produced} decompressed bytes: {e}',
                    section=section_label, declared_size=declared_size,
                    decompressed_size=produced, compressed_size=size,
                    compressed_offset=offset + chunk_start) from e

            produced += len(piece)

            # Enforce the total output cap as soon as it is crossed.
            if produced > max_output_size:
                raise compression_error(
                    CompressionFailure.OUTPUT_LIMIT_EXCEEDED,
                    f'Decompression limit of {max_output_size} bytes exceeded '
                    f'in {section_label}: declared size {declared_size}, '
                    f'produced at least {produced} bytes',
                    section=section_label, declared_size=declared_size,
                    decompressed_size=produced,
                    max_output_size=max_output_size, compressed_size=size)

            if buffering:
                room = declared_size - buffered
                if len(piece) <= room:
                    chunks.append(piece)
                    buffered += len(piece)
                else:
                    if room > 0:
                        chunks.append(piece[:room])
                        buffered += room
                    # Output beyond the declaration is still counted (to
                    # report the real length) but never returned.
                    buffering = False

            if decomp.eof:
                reached_eof = True
                unused_at_eof = len(decomp.unused_data)
                break

            tail = decomp.unconsumed_tail
            if not tail:
                break
            if tail == data and not piece:
                raise compression_error(
                    CompressionFailure.CORRUPT_STREAM,
                    f'Corrupt compressed data in {section_label}: decompressor '
                    f'made no progress after producing {produced} bytes',
                    section=section_label, declared_size=declared_size,
                    decompressed_size=produced, compressed_size=size,
                    compressed_offset=offset + chunk_start)
            data = tail

        if reached_eof:
            break

    if not reached_eof:
        # No end marker: the stream was cut short (possibly exactly at the
        # declared length, which would otherwise look like success).
        if produced > declared_size:
            raise compression_error(
                CompressionFailure.DATA_LONGER_THAN_DECLARED,
                f'Decompressed data is longer than declared in {section_label}: '
                f'decompressed at least {produced} bytes, should be '
                f'{declared_size} bytes long',
                section=section_label, declared_size=declared_size,
                decompressed_size=produced,
                extra_bytes=produced - declared_size, compressed_size=size)
        if produced == declared_size:
            raise compression_error(
                CompressionFailure.INCOMPLETE_STREAM,
                f'Compressed stream in {section_label} ended prematurely after '
                f'{produced} decompressed bytes without reaching its end marker',
                section=section_label, declared_size=declared_size,
                decompressed_size=produced, compressed_size=size)
        raise compression_error(
            CompressionFailure.DATA_SHORTER_THAN_DECLARED,
            f'Decompressed data is {produced} bytes long, should be '
            f'{declared_size} bytes long ({declared_size - produced} bytes '
            f'missing) in {section_label}',
            section=section_label, declared_size=declared_size,
            decompressed_size=produced,
            missing_bytes=declared_size - produced, compressed_size=size)

    # Reached the end of the deflate stream: classify the length relation.
    if produced > declared_size:
        raise compression_error(
            CompressionFailure.DATA_LONGER_THAN_DECLARED,
            f'Decompressed data is {produced} bytes long, should be '
            f'{declared_size} bytes long ({produced - declared_size} extra '
            f'bytes) in {section_label}',
            section=section_label, declared_size=declared_size,
            decompressed_size=produced,
            extra_bytes=produced - declared_size, compressed_size=size)

    if produced < declared_size:
        missing = declared_size - produced
        raise compression_error(
            CompressionFailure.DATA_SHORTER_THAN_DECLARED,
            f'Decompressed data is {produced} bytes long, should be '
            f'{declared_size} bytes long',
            section=section_label, declared_size=declared_size,
            decompressed_size=produced,
            missing_bytes=missing, compressed_size=size)

    # Bytes belonging to the deflate stream, and bytes after it.
    consumed = fed - unused_at_eof
    trailing = unused_at_eof + (size - fed)
    if trailing:
        raise compression_error(
            CompressionFailure.TRAILING_DATA,
            f'Unexpected trailing data after the compressed stream in '
            f'{section_label}: {trailing} trailing bytes starting at '
            f'compressed offset {offset + consumed:#x}',
            section=section_label, declared_size=declared_size,
            decompressed_size=produced, trailing_bytes=trailing,
            compressed_offset=offset + consumed, compressed_size=size)

    return b''.join(chunks)


def decompress_zlib_gnu_section(
    stream: IO[bytes],
    offset: int,
    size: int,
    *,
    section_label: str,
    max_output_size: int = DEFAULT_MAX_DECOMPRESSED_SIZE,
) -> bytes:
    """ Decompress a GNU-style ``.zdebug_*`` section.

    Such a section starts with the 4-byte magic ``b'ZLIB'`` followed by the
    uncompressed size as a big-endian 64-bit integer, then a zlib stream.  The
    same bounded, categorised machinery as decompress_zlib_section is used.
    """
    if size < 12:
        raise compression_error(
            CompressionFailure.INCOMPLETE_STREAM,
            f'GNU-compressed section {section_label} is too small for its '
            f'{12}-byte header (size {size})',
            section=section_label, compressed_size=size, decompressed_size=0)
    try:
        stream.seek(offset)
        header = stream.read(12)
    except (OSError, ValueError) as e:
        raise ELFParseError(
            f'Cannot read compression header of {section_label}') from e
    if len(header) != 12:
        raise ELFParseError(
            f'Expected 12 bytes at offset {offset}, found {len(header)}')
    magic, declared_size = header[:4], int.from_bytes(header[4:], 'big')
    if magic != b'ZLIB':
        raise compression_error(
            CompressionFailure.UNSUPPORTED_ALGORITHM,
            f'Unknown compression type: {magic!r} in {section_label}',
            algorithm=magic, section=section_label)

    return decompress_zlib_section(
        stream, offset + 12, size - 12, declared_size,
        section_label=section_label, max_output_size=max_output_size)


def _iter_file_chunks(
    stream: IO[bytes],
    offset: int,
    size: int,
) -> Iterator[bytes]:
    """ Yield file-backed bytes in bounded chunks, re-checking each read.
    """
    position = offset
    remaining = size
    while remaining:
        count = min(DECOMPRESS_INPUT_CHUNK_SIZE, remaining)
        try:
            stream.seek(position)
            data = stream.read(count)
        except (OSError, ValueError) as e:
            raise ELFParseError(
                f'Cannot read {count} bytes at offset {position}') from e
        if len(data) != count:
            raise ELFParseError(
                f'Expected {count} bytes at offset {position}, '
                f'found {len(data)}')
        yield data
        position += count
        remaining -= count
