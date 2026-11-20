#-------------------------------------------------------------------------------
# elftools example: elf_notes.py
#
# An example of obtaining note sections from an ELF file and examining
# the notes it contains. It also shows the lazy, provenance-aware note index
# that is useful for core dumps.
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
import sys

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import NoteSection


def describe_desc(note):
    desc = note['n_desc']
    if note['n_type'] == 'NT_GNU_ABI_TAG':
        return (
            f"{desc['abi_os']}, ABI: {desc['abi_major']:d}."
            f"{desc['abi_minor']:d}.{desc['abi_tiny']:d}")
    if note['n_type'] in {'NT_GNU_BUILD_ID', 'NT_GNU_GOLD_VERSION'}:
        return desc
    return bytes(desc).hex()


def process_file(filename):
    print('Processing file:', filename)
    with open(filename, 'rb') as f:
        elf = ELFFile(f)
        for sect in elf.iter_sections():
            if not isinstance(sect, NoteSection):
                continue
            print(
                f'  Note section "{sect.name}" at offset '
                f"0x{sect.header['sh_offset']:08x} with size "
                f"{sect.header['sh_size']:d}"
            )
            for note in sect.iter_notes():
                print('    Name:', note['n_name'])
                print('    Type:', note['n_type'])
                print('    Desc:', describe_desc(note))

        # The note index covers every note section/segment and records where
        # each note came from. Descriptors are only read when accessed, so
        # this works the same for small objects and multi-gigabyte cores.
        print('  Note index:')
        index = elf.note_index()
        for record in index.iter_records():
            print(
                f"    seq={record.sequence} "
                f"{record.source.kind}[{record.source.index}] "
                f"#{record.index_in_source} "
                f"{record.name} {record.n_type}")
        for info in index.truncations:
            print('    Incomplete:', info)


if __name__ == '__main__':
    if sys.argv[1] == '--test':
        for filename in sys.argv[2:]:
            process_file(filename)
