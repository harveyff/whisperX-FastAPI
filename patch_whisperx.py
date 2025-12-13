#!/usr/bin/env python3
"""
Patch whisperx to replace use_auth_token with token for pyannote.audio>=4.0.1 compatibility.
"""
import re
import glob
import sys
import os

# Find all whisperx Python files - try multiple patterns
whisperx_paths = []

# Try glob patterns
for pattern in [
    '/usr/local/lib/python3.*/dist-packages/whisperx/**/*.py',
    '/usr/local/lib/python3.*/site-packages/whisperx/**/*.py',
]:
    whisperx_paths.extend(glob.glob(pattern, recursive=True))

# Also try direct path walking
for base_path in ['/usr/local/lib/python3.11/dist-packages', '/usr/local/lib/python3.10/dist-packages']:
    whisperx_dir = os.path.join(base_path, 'whisperx')
    if os.path.exists(whisperx_dir):
        for root, dirs, files in os.walk(whisperx_dir):
            for file in files:
                if file.endswith('.py'):
                    whisperx_paths.append(os.path.join(root, file))

whisperx_paths = list(set(whisperx_paths))  # Remove duplicates

print(f"Found {len(whisperx_paths)} whisperx Python files to patch", file=sys.stderr)
if len(whisperx_paths) == 0:
    print("WARNING: No whisperx files found! Trying to locate...", file=sys.stderr)
    import subprocess
    try:
        result = subprocess.run(['find', '/usr/local/lib', '-name', 'whisperx', '-type', 'd', '2>/dev/null'], 
                              shell=True, capture_output=True, text=True, timeout=5)
        if result.stdout:
            for dir_path in result.stdout.strip().split('\n'):
                if dir_path:
                    for root, dirs, files in os.walk(dir_path):
                        for file in files:
                            if file.endswith('.py'):
                                whisperx_paths.append(os.path.join(root, file))
        print(f"After find, found {len(whisperx_paths)} files", file=sys.stderr)
    except Exception as e:
        print(f"Find command failed: {e}", file=sys.stderr)

for filepath in whisperx_paths:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace ALL occurrences of use_auth_token with token
        # This is safe because we're doing a consistent replacement across the entire codebase
        # We skip comments and strings by processing line by line
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            original_line = line
            
            # Skip comment lines
            if line.strip().startswith('#'):
                new_lines.append(line)
                continue
            
            # Replace use_auth_token with token (all occurrences)
            # This handles:
            # - Parameter definitions: use_auth_token= -> token=
            # - Variable references: use_auth_token -> token
            # - Function calls: use_auth_token=value -> token=value
            line = re.sub(r'\buse_auth_token\b', 'token', line)
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        # Also patch diarize.py for pyannote.audio>=4.0.1 compatibility
        if 'diarize.py' in filepath:
            # Fix itertracks API change
            # Old: diarization.itertracks(yield_label=True) 
            # New: DiarizeOutput doesn't have itertracks, need to access .annotation first
            if 'itertracks' in content:
                # Fix pd.DataFrame(var.itertracks(yield_label=True), columns=...)
                # Replace with version that handles DiarizeOutput.annotation
                content = re.sub(
                    r'pd\.DataFrame\((\w+)\.itertracks\(yield_label=True\)',
                    r'pd.DataFrame([(segment, label, label) for segment, track, label in (\1.annotation if hasattr(\1, "annotation") else \1).itertracks(yield_label=True)]',
                    content
                )
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Patched: {filepath}", file=sys.stderr)
            # Count replacements
            replacements = original_content.count('use_auth_token') - content.count('use_auth_token')
            if replacements > 0:
                print(f"  Replaced {replacements} occurrences", file=sys.stderr)
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        pass
