"""
Patch torchaudio to add list_audio_backends() method if missing.
This is needed for pyannote.audio>=4.0.1 compatibility.
"""
import torchaudio

# Check if list_audio_backends exists
if not hasattr(torchaudio, 'list_audio_backends'):
    # Add a compatibility function
    def list_audio_backends():
        """Compatibility function for list_audio_backends."""
        try:
            # Try to get backends from torchaudio backend
            if hasattr(torchaudio, 'backend'):
                return torchaudio.backend.list_audio_backends()
            # Fallback: return empty list or default backends
            return []
        except Exception:
            return []
    
    torchaudio.list_audio_backends = list_audio_backends
    print("✓ Patched torchaudio.list_audio_backends()")

