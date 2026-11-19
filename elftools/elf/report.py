#-------------------------------------------------------------------------------
# elftools: elf/report.py
#
# Result types for lenient structural scans.
#
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..common.exceptions import ELFParseError

__all__ = ['StructureRejection', 'TableScan']

_T = TypeVar('_T')


@dataclass(frozen=True)
class StructureRejection:
    """ Why one structure (or one table entry) was rejected.

        ``error`` is the :class:`ELFParseError` the strict API would raise.
        It carries the machine-readable location fields (structure, index,
        expected/actual sizes), while ``reason`` is the human readable text.
        The text never contains bytes taken from the input file.
    """
    structure: str
    index: int | None
    error: ELFParseError

    @property
    def reason(self) -> str:
        return str(self.error)


@dataclass(frozen=True)
class TableScan(Generic[_T]):
    """ Usable prefix of a table plus a reason for every rejected entry.

        ``declared`` is the entry count claimed by the file, ``available`` the
        number of whole entries the file can hold. ``truncated_at`` is the
        first index that could not be used because the table was truncated
        (None for an untruncated table).

        Iterating the scan yields only the usable entries, so a partial table
        still produces the successfully parsed part rather than an empty
        result.
    """
    entries: tuple[_T, ...]
    rejections: tuple[StructureRejection, ...]
    declared: int
    available: int
    truncated_at: int | None = None

    def __iter__(self) -> Iterator[_T]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)
