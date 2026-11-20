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
    """Raised when a compressed section cannot be decompressed exactly.

    Instances carry a ``category`` attribute (a member of
    :class:`elftools.common.compression.CompressionFailure`) plus numeric
    details such as ``declared_size``, ``decompressed_size``,
    ``max_output_size`` and ``section``.
    """
    category: object = None

class DWARFError(Exception):
    pass
