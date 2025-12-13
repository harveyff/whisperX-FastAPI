#!/usr/bin/env python3
"""
Patch whisperx to replace use_auth_token with token for pyannote.audio>=4.0.1 compatibility.
"""
import re
import glob
import sys

# Find all whisperx Python files
whisperx_paths = glob.glob('/usr/local/lib/python3.*/dist-packages/whisperx/**/*.py', recursive=True)

print(f"Found {len(whisperx_paths)} whisperx Python files to patch", file=sys.stderr)

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
