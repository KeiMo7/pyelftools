#-------------------------------------------------------------------------------
# elftools: common/exceptions.py
#
# Exception classes for elftools
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import annotations


class ELFError(Exception):
    """ Base class for every error raised by elftools for malformed or
        otherwise unusable input.

        Callers that treat an ELF file as untrusted data can catch this single
        class and know that no exception from the lower-level byte parsing,
        decompression or stream seeking escaped the library.
    """


class ELFRelocationError(ELFError):
    pass


class ELFParseError(ELFError):
    """ A single structure (or a bounded part of the file) is unusable.

        Instances raised by the library carry machine-readable location
        information in the attributes below.  The values are counts and
        indices only - the error message never embeds bytes from the file.

        structure:
            Human readable kind of the rejected structure, e.g.
            ``'section header table'`` or ``'symbol table'``.
        index:
            Zero-based index of the rejected item within its structure, if
            the rejection concerns an item rather than a whole structure.
        expected, actual:
            Number of bytes the structure required and the number of bytes
            actually available (or produced), when known.
        first_invalid_index:
            For table-level rejections, the first entry index that cannot be
            used.  Entries before it form a usable prefix.
        declared_count, available_count:
            For table-level rejections, the entry count claimed by the file
            and the number of whole entries that actually fit in the file.
    """
    def __init__(
        self,
        message: str = '',
        *,
        structure: str | None = None,
        index: int | None = None,
        expected: int | None = None,
        actual: int | None = None,
        first_invalid_index: int | None = None,
        declared_count: int | None = None,
        available_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.structure = structure
        self.index = index
        self.expected = expected
        self.actual = actual
        self.first_invalid_index = first_invalid_index
        self.declared_count = declared_count
        self.available_count = available_count

    @property
    def scope(self) -> str:
        """ ``'file'`` for whole-file rejections, ``'structure'`` otherwise.
        """
        return 'structure'


class ELFUnparsableFileError(ELFParseError):
    """ The file as a whole cannot be parsed.

        This is distinct from :class:`ELFParseError`, which reports a single
        unavailable structure while the rest of the file remains usable.
        Raised, for example, when the ELF header or its identification bytes
        are truncated.
    """
    @property
    def scope(self) -> str:
        return 'file'


class ELFCompressionError(ELFParseError):
    """ A compressed section is unusable.

        Covers a truncated compression header, a decompressed size that does
        not match the declaration and a corrupt or truncated compressed
        stream.  It is a subclass of :class:`ELFParseError` so that callers
        can handle every malformed-input rejection with one ``except``.
    """


class DWARFError(Exception):
    pass

