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

class ELFCompressionError(ELFError):
    pass

class ELFMemoryLimitError(ELFError):
    """Raised when a bounded-memory access path refuses to exceed its
       configured limit.

       The `category` attribute names the kind of data that hit the limit
       (currently 'symbols' or 'symbol-names'), so a caller can tell which
       buffer or cache was exhausted. Nothing is ever silently truncated.
    """
    def __init__(self, message: str, category: str) -> None:
        super().__init__(message)
        self.category = category

class ELFSymbolError(ELFParseError):
    """An error raised while parsing a single symbol table entry.

       `index` is the index of the offending entry in its table, so a failure
       in a table with millions of entries points at the exact entry.
    """
    def __init__(self, message: str, index: int) -> None:
        super().__init__(f'symbol {index}: {message}')
        self.index = index

class DWARFError(Exception):
    pass
