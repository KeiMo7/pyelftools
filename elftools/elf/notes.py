#-------------------------------------------------------------------------------
# elftools: elf/notes.py
#
# ELF notes
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from ..common.exceptions import ELFNoteDecodeError, ELFParseError
from ..common.utils import bytes2str, roundup, struct_parse
from ..construct import ConstructError, CString
from ..construct.lib.container import Container
from .enums import ENUM_ANDROID_NOTE_N_TYPE, ENUM_CORE_NOTE_N_TYPE, ENUM_NOTE_N_TYPE

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from .elffile import ELFFile


# Note type numbers have meaning only within their owner's namespace.
_NOTE_TYPE_NAMES = {
    owner: {value: name for name, value in enum.items() if name != '_default_'}
    for owner, enum in (
        ('GNU', ENUM_NOTE_N_TYPE),
        ('Android', ENUM_ANDROID_NOTE_N_TYPE),
    )
}
_CORE_NOTE_TYPE_NAMES = {
    value: name for name, value in ENUM_CORE_NOTE_N_TYPE.items() if name != '_default_'
}

# Core note types that carry per-thread state.  Their numbers are only
# meaningful for notes owned by "CORE".
_CORE_THREAD_STATUS_TYPE = 1        # NT_PRSTATUS
_CORE_THREAD_REGISTER_TYPES = {2}   # NT_FPREGSET / NT_PRFPREG

# Offset of pr_pid within an elf_prstatus descriptor, by ELF class.
_PRSTATUS_PID_OFFSET = {32: 24, 64: 32}


class NoteSourceKind(StrEnum):
    """The kind of ELF structure a note range was taken from."""
    SECTION = 'section'
    SEGMENT = 'segment'


class NoteTruncation(StrEnum):
    """Where parsing of a note source stopped because the file ended.

    ``HEADER`` means the note header itself is incomplete, ``NAME`` that the
    owner name is incomplete and ``DESCRIPTOR`` that the descriptor is
    incomplete.  A source that parses all of its declared notes has no
    truncation status at all.
    """
    HEADER = 'header'
    NAME = 'name'
    DESCRIPTOR = 'descriptor'


@dataclass(frozen=True, slots=True)
class NoteSource:
    """A section or segment that contains notes.

    ``index`` is the index in the file's section or program header table, and
    is stable for a given file.  ``offset``/``size`` describe the declared
    range in the file.  ``name`` is set for sections, ``type`` is the ELF
    structure type (``SHT_NOTE`` or ``PT_NOTE``).
    """
    kind: NoteSourceKind
    index: int
    offset: int
    size: int
    name: str | None = None
    type: str = ''

    @property
    def end(self) -> int:
        return self.offset + self.size

    def __str__(self) -> str:
        label = self.name if self.name is not None else self.type
        return (f'{self.kind.value}[{self.index}] {label} '
                f'@ {self.offset:#x}+{self.size:#x}')


@dataclass(frozen=True, slots=True)
class NoteTruncationInfo:
    """Describes an incomplete note at the end of a truncated note source.

    ``offset`` is the absolute file offset at which the incomplete note
    starts.  ``consumed`` is the number of bytes of the source that belong to
    complete notes before it, ``available`` is how many bytes the file
    actually provides for the incomplete note, and ``expected`` is how many
    bytes the note requires.  ``namesz``/``descsz`` are present only when the
    header was complete enough to read them.
    """
    source: NoteSource
    truncation: NoteTruncation
    offset: int
    consumed: int
    available: int
    expected: int | None
    note_index: int
    namesz: int | None = None
    descsz: int | None = None

    def __str__(self) -> str:
        return (f'{self.truncation.value} truncation in {self.source} at '
                f'{self.offset:#x}: consumed {self.consumed} bytes, '
                f'available {self.available} of {self.expected}')


@dataclass(frozen=True, slots=True)
class ThreadNoteKey:
    """Stable identity of a thread note group.

    ``pid`` is the thread id read from an ``NT_PRSTATUS`` note when one is
    available.  ``anchor`` is the global sequence number of the first record
    that established the group.  Together they remain distinct for different
    threads and identical when the same dump is parsed again.  For groups
    that have register notes but no status note, ``pid`` is ``None`` and
    ``anchor`` identifies the group.
    """
    pid: int | None
    anchor: int


class NoteRecord:
    """One complete note, addressed by its position in the file.

    The record holds only metadata (owner, type, sizes and provenance); the
    descriptor is read and decoded on demand by :meth:`read`.  This keeps
    memory use independent of the total size of all note descriptors.

    ``source``/``index_in_source``/``sequence`` are stable provenance fields:
    the global sequence only depends on file order, never on scheduling.
    ``n_type`` and ``n_name`` use the same names and semantics as the
    containers returned by the legacy ``iter_notes`` API.
    """
    def __init__(
        self,
        elffile: ELFFile,
        source: NoteSource,
        index_in_source: int,
        sequence: int,
        offset: int,
        namesz: int,
        descsz: int,
        type_raw: int,
        name: str | None,
        n_type: str | int,
        name_size: int,
        desc_offset: int,
        size: int,
    ) -> None:
        self.elffile = elffile
        self.source = source
        self.index_in_source = index_in_source
        self.sequence = sequence
        #: Absolute file offset of the note header.
        self.offset = offset
        self.namesz = namesz
        self.descsz = descsz
        #: Numeric n_type, before owner-specific symbolic resolution.
        self.type_raw = type_raw
        #: Owner name, or None for a note without an owner.
        self.name = name
        #: Symbolic type for known core/GNU/Android notes, numeric otherwise.
        self.n_type = n_type
        self.name_size = name_size
        #: Absolute file offset of the descriptor.
        self.desc_offset = desc_offset
        #: Total on-disk size of the note, including alignment padding.
        self.size = size
        #: Set by read() when the descriptor could not be decoded.
        self.decode_error: str | None = None

    @property
    def owner(self) -> str | None:
        """Alias for ``name`` (the note's owner)."""
        return self.name

    @property
    def descriptor_offset(self) -> int:
        return self.desc_offset

    @property
    def descriptor_size(self) -> int:
        return self.descsz

    def __repr__(self) -> str:
        return (
            f'<NoteRecord {self.name!r} {self.n_type!r} '
            f'seq={self.sequence} source={self.source.kind.value}'
            f'[{self.source.index}] index={self.index_in_source} '
            f'offset={self.offset:#x}>')

    @property
    def is_thread_status(self) -> bool:
        return self.name == 'CORE' and (
            self.n_type == 'NT_PRSTATUS'
            or self.type_raw == _CORE_THREAD_STATUS_TYPE)

    @property
    def is_thread_register(self) -> bool:
        return self.name == 'CORE' and (
            self.n_type in ('NT_FPREGSET', 'NT_PRFPREG')
            or self.type_raw in _CORE_THREAD_REGISTER_TYPES)

    @property
    def is_thread_note(self) -> bool:
        return self.is_thread_status or self.is_thread_register

    def raw_descriptor(self) -> bytes:
        """Read the raw descriptor bytes without decoding them."""
        stream = self.elffile.stream
        stream.seek(self.desc_offset)
        data = stream.read(self.descsz)
        if len(data) != self.descsz:
            raise ELFParseError(
                f'Expected {self.descsz} descriptor bytes at offset '
                f'{self.desc_offset}, found {len(data)}')
        return data

    def read_pid(self) -> int | None:
        """Read the thread id from an NT_PRSTATUS descriptor, if possible.

        Only the four bytes holding ``pr_pid`` are read, so probing a whole
        dump does not load the descriptors.
        """
        if not self.is_thread_status:
            return None
        offset = _PRSTATUS_PID_OFFSET.get(self.elffile.elfclass)
        if offset is None or self.descsz < offset + 4:
            return None
        stream = self.elffile.stream
        stream.seek(self.desc_offset + offset)
        data = stream.read(4)
        if len(data) != 4:
            return None
        return int.from_bytes(
            data, 'little' if self.elffile.little_endian else 'big')

    def read(self, strict: bool = False) -> Container:
        """Read and decode the note now.

        The returned container has exactly the fields of the legacy
        ``iter_notes`` API, plus provenance fields prefixed with ``n_``.

        By default a descriptor that cannot be decoded does not raise: the
        raw bytes are returned in ``n_desc`` and the failure is recorded in
        ``decode_error``, so the rest of the notes stay usable and the
        failure can be listed with :meth:`NoteIndex.iter_decode_errors`.
        With ``strict=True`` the decoding error is raised, which is what the
        legacy ``iter_notes`` API did.
        """
        stream = self.elffile.stream
        stream.seek(self.desc_offset)
        disk_descsz = roundup(self.descsz, 2)
        data = stream.read(disk_descsz)
        if len(data) != disk_descsz:
            self.decode_error = (
                f'Short read for descriptor at offset {self.desc_offset}: '
                f'expected {disk_descsz} bytes, found {len(data)}')
            if strict:
                raise ELFNoteDecodeError(self.decode_error)
            desc_data = data[:self.descsz]
        else:
            self.decode_error = None
            desc_data = data[:self.descsz]

        note = _new_note_container(self, desc_data)
        try:
            note['n_desc'] = _decode_descriptor(
                self.elffile, note['n_type'], desc_data)
        except Exception as e:
            self.decode_error = f'{type(e).__name__}: {e}'
            if strict:
                if isinstance(e, ELFNoteDecodeError):
                    raise
                raise ELFNoteDecodeError(self.decode_error) from e
            note['n_desc'] = desc_data
        return note


class ThreadNoteGroup:
    """The notes belonging to a single thread.

    Status notes (``NT_PRSTATUS``) and register notes (``NT_FPREGSET`` and
    friends) are kept together.  Repeated notes for the same thread are all
    preserved: the first of each kind is the primary record and the remaining
    ones are exposed as duplicates.
    """
    def __init__(self, key: ThreadNoteKey) -> None:
        self.key = key
        self._records: list[NoteRecord] = []
        self._status: list[NoteRecord] = []
        self._registers: list[NoteRecord] = []

    def _append_status(self, record: NoteRecord) -> None:
        self._records.append(record)
        self._status.append(record)

    def _append_register(self, record: NoteRecord) -> None:
        self._records.append(record)
        self._registers.append(record)

    @property
    def pid(self) -> int | None:
        return self.key.pid

    @property
    def records(self) -> tuple[NoteRecord, ...]:
        return tuple(self._records)

    @property
    def status_records(self) -> tuple[NoteRecord, ...]:
        return tuple(self._status)

    @property
    def register_records(self) -> tuple[NoteRecord, ...]:
        return tuple(self._registers)

    @property
    def primary_status(self) -> NoteRecord | None:
        """The first status note, or None if the group is incomplete."""
        return self._status[0] if self._status else None

    @property
    def primary_register(self) -> NoteRecord | None:
        """The first register note, or None if none was captured."""
        return self._registers[0] if self._registers else None

    @property
    def duplicate_status_records(self) -> tuple[NoteRecord, ...]:
        return tuple(self._status[1:])

    @property
    def duplicate_register_records(self) -> tuple[NoteRecord, ...]:
        return tuple(self._registers[1:])

    @property
    def has_registers(self) -> bool:
        return bool(self._registers)

    @property
    def is_complete(self) -> bool:
        """True when the group has a status note."""
        return bool(self._status)

    @property
    def complete(self) -> bool:
        """Alias for :attr:`is_complete`."""
        return self.is_complete

    @property
    def register_view_available(self) -> bool:
        """Whether a register view can be built for this group."""
        return self.is_complete and self.has_registers

    def iter_records(self) -> Iterator[NoteRecord]:
        return iter(self._records)

    def iter_register_records(self) -> Iterator[NoteRecord]:
        return iter(self._registers)

    def iter_register_notes(self) -> Iterator[Container]:
        """Decode register notes one at a time, in capture order."""
        for record in self._registers:
            yield record.read()

    def __repr__(self) -> str:
        return (
            f'<ThreadNoteGroup pid={self.pid} key={self.key!r} '
            f'status={len(self._status)} registers={len(self._registers)}>')


class NoteIndex:
    """Lazily indexed notes from every note section/segment in an ELF file.

    Creating the index reads note headers and owner names only, never
    descriptors.  Records are produced in a deterministic global order:
    note sections first (in section-header order), then note segments that
    are not already represented by a section (in program-header order).
    Notes repeated because a segment overlaps a section are emitted once.
    Repeated parsing of the same file therefore yields identical ordering,
    provenance and grouping.

    Incomplete notes stop only their own source.  They are reported through
    :attr:`truncations` and never raise.
    """
    def __init__(self, elffile: ELFFile) -> None:
        self.elffile = elffile
        self._sources = tuple(_collect_note_sources(elffile))
        self._truncations: list[NoteTruncationInfo] = []
        self._scanned = False

    def _ensure_scanned(self) -> None:
        """Complete a metadata-only scan so truncations are observable."""
        if not self._scanned:
            for _ in self.iter_records():
                pass

    @property
    def sources(self) -> tuple[NoteSource, ...]:
        return self._sources

    @property
    def truncations(self) -> tuple[NoteTruncationInfo, ...]:
        """Incomplete notes in the file, in file order.

        Querying this before iterating records completes a metadata-only scan,
        so callers do not have to drain the iterator to observe truncation.
        """
        self._ensure_scanned()
        return tuple(self._truncations)

    @property
    def is_complete(self) -> bool:
        """True if no note source ended in an incomplete note."""
        return not self.truncations

    @property
    def truncation(self) -> NoteTruncation | None:
        """The first truncation status, or None for a complete dump."""
        truncations = self.truncations
        return truncations[0].truncation if truncations else None

    def iter_records(self) -> Iterator[NoteRecord]:
        """Yield complete notes in stable global order.

        Incomplete notes stop the affected source only; the rest of that
        source is not parsed and later sources are unaffected.  Truncations
        are collected in :attr:`truncations` as the scan proceeds, so they
        are complete once the iterator is exhausted.
        """
        self._truncations = []
        sequence = 0
        seen_offsets: set[int] = set()
        for source in self._sources:
            truncations: list[NoteTruncationInfo] = []
            for record in _iter_source_records(self.elffile, source, truncations):
                if record.offset in seen_offsets:
                    # This note was already emitted from an overlapping source.
                    continue
                seen_offsets.add(record.offset)
                record.sequence = sequence
                sequence += 1
                yield record
            self._truncations.extend(truncations)
        self._scanned = True

    def records(self) -> tuple[NoteRecord, ...]:
        """All complete notes.  Only metadata is retained, not descriptors."""
        return tuple(self.iter_records())

    def iter_decode_errors(self) -> Iterator[NoteRecord]:
        """Yield notes whose descriptors failed to decode.

        Each descriptor is read one at a time, so listing failures does not
        retain the decoded data.  Failures are isolated to their own note.
        """
        for record in self.iter_records():
            record.read()
            if record.decode_error is not None:
                yield record

    def iter_process_records(self) -> Iterator[NoteRecord]:
        """Yield records that are not per-thread status/register notes."""
        for record in self.iter_records():
            if not record.is_thread_note:
                yield record

    def process_records(self) -> tuple[NoteRecord, ...]:
        return tuple(self.iter_process_records())

    def iter_thread_groups(self) -> Iterator[ThreadNoteGroup]:
        """Group per-thread notes, yielding groups in first-appearance order.

        Association uses note order within a source: register notes follow
        the ``NT_PRSTATUS`` note they belong to.  Threads are keyed by the
        ``pr_pid`` from their status note, so duplicate status/register notes
        for the same thread land in the same group.  If a register note has no
        preceding status note, it forms an incomplete group of its own.
        """
        groups: dict[ThreadNoteKey, ThreadNoteGroup] = {}
        pid_keys: dict[int, ThreadNoteKey] = {}
        current: ThreadNoteKey | None = None

        for record in self.iter_records():
            if record.is_thread_status:
                pid = record.read_pid()
                if pid is not None:
                    key = pid_keys.get(pid)
                    if key is None:
                        key = ThreadNoteKey(pid, record.sequence)
                        pid_keys[pid] = key
                else:
                    key = ThreadNoteKey(None, record.sequence)
                current = key
                group = groups.get(key)
                if group is None:
                    group = groups[key] = ThreadNoteGroup(key)
                group._append_status(record)
            elif record.is_thread_register:
                if current is None:
                    current = ThreadNoteKey(None, record.sequence)
                group = groups.get(current)
                if group is None:
                    group = groups[current] = ThreadNoteGroup(current)
                group._append_register(record)
            # Non-thread notes may be interleaved with thread notes; they do
            # not end the current thread context.

        yield from groups.values()

    def thread_groups(self) -> tuple[ThreadNoteGroup, ...]:
        return tuple(self.iter_thread_groups())


#-------------------------------------------------------------------------------
# Note scanning
#-------------------------------------------------------------------------------

def iter_notes(elffile: ELFFile, offset: int, size: int) -> Iterator[Container]:
    """ Yield complete notes within a section or segment.

        n_type is a symbolic name for known types in the owner's namespace,
        or the numeric type for unknown types and owners.

        This is the legacy per-range API.  Truncation ends the iteration
        without raising; use :class:`NoteIndex` to observe where and why it
        stopped.  Note fields and ordering are unchanged.  As before, a
        descriptor that cannot be decoded raises.
    """
    source = NoteSource(
        kind=NoteSourceKind.SEGMENT, index=-1, offset=offset,
        size=max(0, size), type='')
    truncations: list[NoteTruncationInfo] = []
    for sequence, record in enumerate(
            _iter_source_records(elffile, source, truncations)):
        record.sequence = sequence
        yield record.read(strict=True)


def _collect_note_sources(elffile: ELFFile) -> Iterator[NoteSource]:
    """Pick deterministic note sources, avoiding duplicated section/segment data."""
    section_ranges: list[tuple[int, int]] = []
    for index, section in enumerate(elffile.iter_sections()):
        if section['sh_type'] != 'SHT_NOTE':
            continue
        offset = section['sh_offset']
        size = section['sh_size']
        section_ranges.append((offset, offset + size))
        yield NoteSource(
            kind=NoteSourceKind.SECTION, index=index, offset=offset,
            size=size, name=section.name, type='SHT_NOTE')

    for index, segment in enumerate(elffile.iter_segments()):
        if segment['p_type'] != 'PT_NOTE':
            continue
        offset = segment['p_offset']
        size = segment['p_filesz']
        if _range_covered(offset, offset + size, section_ranges):
            # This segment only repeats notes already exposed as sections.
            continue
        yield NoteSource(
            kind=NoteSourceKind.SEGMENT, index=index, offset=offset,
            size=size, type='PT_NOTE')


def _range_covered(start: int, end: int, ranges: Iterable[tuple[int, int]]) -> bool:
    if end <= start:
        return True
    cursor = start
    for range_start, range_end in sorted(ranges):
        if range_end <= cursor:
            continue
        if range_start > cursor:
            return False
        cursor = max(cursor, range_end)
        if cursor >= end:
            return True
    return False


def _iter_source_records(
    elffile: ELFFile,
    source: NoteSource,
    truncations: list[NoteTruncationInfo],
) -> Iterator[NoteRecord]:
    """Scan one source, yielding complete notes and recording truncations.

    Reading stops at the source boundary even when the file has more data, so
    a bogus descriptor size can never consume notes from a later segment.
    """
    nhdr_size = elffile.structs.Elf_Nhdr.sizeof()
    stream = elffile.stream
    start = source.offset
    declared_end = source.end
    file_truncated = declared_end > elffile.stream_len
    limit = min(declared_end, elffile.stream_len)
    offset = start
    note_index = 0

    while offset < limit:
        available = limit - offset

        if available < nhdr_size:
            if available == declared_end - offset:
                # The file has every byte the source declares; the remaining
                # bytes are trailing alignment padding of a complete source.
                return
            truncations.append(NoteTruncationInfo(
                source=source, truncation=NoteTruncation.HEADER,
                offset=offset, consumed=offset - start, available=available,
                expected=nhdr_size, note_index=note_index))
            return

        try:
            header: Container = struct_parse(
                elffile.structs.Elf_Nhdr, stream, stream_pos=offset)
        except ELFParseError:
            truncations.append(NoteTruncationInfo(
                source=source, truncation=NoteTruncation.HEADER,
                offset=offset, consumed=offset - start, available=available,
                expected=nhdr_size, note_index=note_index))
            return

        namesz: int = header['n_namesz']
        descsz: int = header['n_descsz']
        type_raw = cast('int', header['n_type'])
        name_disk = roundup(namesz, 2)
        desc_disk = roundup(descsz, 2)
        remaining = available - nhdr_size

        if remaining < name_disk:
            truncations.append(NoteTruncationInfo(
                source=source, truncation=NoteTruncation.NAME,
                offset=offset, consumed=offset - start, available=available,
                expected=nhdr_size + name_disk, note_index=note_index,
                namesz=namesz, descsz=descsz))
            return

        stream.seek(offset + nhdr_size)
        name_data = stream.read(name_disk)
        if len(name_data) != name_disk:
            truncations.append(NoteTruncationInfo(
                source=source, truncation=NoteTruncation.NAME,
                offset=offset, consumed=offset - start, available=available,
                expected=nhdr_size + name_disk, note_index=note_index,
                namesz=namesz, descsz=descsz))
            return
        name = _decode_name(name_data) if namesz else None
        n_type = _resolve_note_type(elffile, name, type_raw)

        if remaining - name_disk < desc_disk:
            truncations.append(NoteTruncationInfo(
                source=source, truncation=NoteTruncation.DESCRIPTOR,
                offset=offset, consumed=offset - start, available=available,
                expected=nhdr_size + name_disk + desc_disk,
                note_index=note_index, namesz=namesz, descsz=descsz))
            return

        size = nhdr_size + name_disk + desc_disk
        yield NoteRecord(
            elffile=elffile,
            source=source,
            index_in_source=note_index,
            sequence=0,
            offset=offset,
            namesz=namesz,
            descsz=descsz,
            type_raw=type_raw,
            name=name,
            n_type=n_type,
            name_size=name_disk,
            desc_offset=offset + nhdr_size + name_disk,
            size=size)

        offset += size
        note_index += 1

    if file_truncated and offset < declared_end:
        # The source claims more bytes than the file provides, and the last
        # complete note ended exactly at EOF: the next header is missing.
        truncations.append(NoteTruncationInfo(
            source=source, truncation=NoteTruncation.HEADER,
            offset=offset, consumed=offset - start, available=0,
            expected=nhdr_size, note_index=note_index))


def _decode_name(name_data: bytes) -> str:
    """Decode an owner name, tolerating a missing NUL terminator."""
    try:
        return bytes2str(CString('').parse(name_data))
    except ConstructError:
        return bytes2str(name_data.split(b'\x00', 1)[0])


def _resolve_note_type(
    elffile: ELFFile, name: str | None, type_raw: int) -> str | int:
    note_types = _NOTE_TYPE_NAMES.get(name, {}) if name is not None else {}
    if name == 'CORE' and elffile['e_type'] == 'ET_CORE':
        note_types = _CORE_NOTE_TYPE_NAMES
    return note_types.get(type_raw, type_raw)


def _new_note_container(record: NoteRecord, desc_data: bytes) -> Container:
    """Build the legacy-shaped container for a record."""
    note = Container(
        n_namesz=record.namesz,
        n_descsz=record.descsz,
        n_type=record.type_raw)
    note['n_name'] = record.name
    note['n_type'] = record.n_type
    note['n_descdata'] = desc_data
    note['n_offset'] = record.offset
    note['n_size'] = record.size
    # Additive provenance fields.  Existing field names keep their meaning.
    note['n_source_kind'] = record.source.kind.value
    note['n_source_index'] = record.source.index
    note['n_index_in_source'] = record.index_in_source
    note['n_sequence'] = record.sequence
    return note


def _decode_descriptor(
    elffile: ELFFile, note_type: str | int, desc_data: bytes) -> Any:
    """Decode a descriptor exactly like the legacy parser did."""
    # Structured decoders must stay within this descriptor, even when its
    # contents are malformed and more bytes follow in the file.
    if note_type == 'NT_GNU_ABI_TAG':
        return struct_parse(elffile.structs.Elf_abi, BytesIO(desc_data))
    elif note_type == 'NT_GNU_BUILD_ID':
        return bytes(desc_data).hex()
    elif note_type == 'NT_GNU_GOLD_VERSION':
        return bytes2str(desc_data)
    elif note_type == 'NT_PRPSINFO':
        return struct_parse(elffile.structs.Elf_Prpsinfo, BytesIO(desc_data))
    elif note_type == 'NT_FILE':
        return struct_parse(elffile.structs.Elf_Nt_File, BytesIO(desc_data))
    elif note_type == 'NT_GNU_PROPERTY_TYPE_0':
        off = 0
        desc_stream = BytesIO(desc_data)
        props: list[Container] = []
        while off < len(desc_data):
            p: Container = struct_parse(elffile.structs.Elf_Prop, desc_stream, off)
            off += roundup(p.pr_datasz + 8, 2 if elffile.elfclass == 32 else 3)
            props.append(p)
        return props
    else:
        return desc_data


__all__ = [
    'ELFNoteDecodeError',
    'NoteIndex',
    'NoteRecord',
    'NoteSource',
    'NoteSourceKind',
    'NoteTruncation',
    'NoteTruncationInfo',
    'ThreadNoteGroup',
    'ThreadNoteKey',
    'iter_notes',
]
