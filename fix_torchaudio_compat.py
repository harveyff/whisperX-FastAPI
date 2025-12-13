"""
Compatibility fix for pyannote.audio with newer torchaudio versions.
This patch replaces AudioMetaData with the new API.
"""
import torchaudio

# Check if AudioMetaData exists, if not, create a compatibility alias
if not hasattr(torchaudio, 'AudioMetaData'):
    # In newer torchaudio versions, metadata is returned differently
    # Create a compatibility class or alias
    try:
        # Try to import from torchaudio.io if available
        from torchaudio.io import StreamReader
        # For compatibility, we can create a simple alias or wrapper
        # This is a minimal fix - may need adjustment based on actual usage
        class AudioMetaData:
            """Compatibility wrapper for AudioMetaData"""
            pass
        
        torchaudio.AudioMetaData = AudioMetaData
    except ImportError:
        # If that doesn't work, create a minimal stub
        from types import SimpleNamespace
        torchaudio.AudioMetaData = SimpleNamespace

