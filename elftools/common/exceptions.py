#-------------------------------------------------------------------------------
# elftools: common/exceptions.py
#
# Exception classes for elftools
#
# Eli Bendersky (eliben@gmail.com)
# This code is in the public domain
#-------------------------------------------------------------------------------
class ELFError(Exception):
    pass

class ELFRelocationError(ELFError):
    pass

class ELFParseError(ELFError):
    pass

class ELFNoteDecodeError(ELFParseError):
    """A note's descriptor could not be decoded.

    The note's header, owner and raw descriptor bytes are still available, and
    decoding failures are isolated to the note that caused them.
    """

class ELFCompressionError(ELFError):
    pass

class DWARFError(Exception):
    pass
