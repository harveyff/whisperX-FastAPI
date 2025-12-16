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
            if hasattr(torchaudio, 'backend') and hasattr(torchaudio.backend, 'list_audio_backends'):
                backends = torchaudio.backend.list_audio_backends()
                if backends:
                    return backends
        except Exception:
            pass
        # Fallback: return default backends that pyannote.audio expects
        # Common backends: "soundfile", "sox", "ffmpeg"
        # Try to detect which ones are actually available
        available_backends = []
        try:
            import soundfile
            available_backends.append("soundfile")
        except ImportError:
            pass
        try:
            import sox
            available_backends.append("sox")
        except ImportError:
            pass
        # Always include soundfile as fallback (it's the most common)
        if not available_backends:
            available_backends = ["soundfile"]
        return available_backends
    
    torchaudio.list_audio_backends = list_audio_backends
    print("✓ Patched torchaudio.list_audio_backends()")

