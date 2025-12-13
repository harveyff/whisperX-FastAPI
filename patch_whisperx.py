#!/usr/bin/env python3
"""
Patch whisperx to replace use_auth_token with token for pyannote.audio>=4.0.1 compatibility.
"""
import re
import glob
import sys

# Find all whisperx Python files
whisperx_paths = glob.glob('/usr/local/lib/python3.*/dist-packages/whisperx/**/*.py', recursive=True)

for filepath in whisperx_paths:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Step 1: Replace parameter definitions: use_auth_token= -> token=
        content = re.sub(r'\buse_auth_token\s*=', 'token=', content)
        
        # Step 2: Replace use_auth_token=None with token=None
        content = re.sub(r'use_auth_token\s*=\s*None', 'token=None', content)
        
        # Step 3: Replace variable references in function bodies
        # But preserve use_auth_token when it's used as a value (e.g., token=use_auth_token should become token=token)
        # Actually, we need to replace the variable name itself in function bodies
        # Replace use_auth_token as a variable name with token, but be careful
        
        # Replace standalone use_auth_token (not in parameter definitions or assignments)
        # This is tricky - we want to replace variable references but not break things
        # Let's replace use_auth_token when it appears as a variable (not in strings/comments)
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            # Skip comments and strings
            if line.strip().startswith('#') or ('"' in line and 'use_auth_token' in line) or ("'" in line and 'use_auth_token' in line):
                new_lines.append(line)
                continue
            
            # Replace use_auth_token as variable name
            # After replacing use_auth_token= with token=, we need to replace the variable name itself
            # So token=use_auth_token becomes token=token (variable name also changed)
            if 'use_auth_token' in line:
                # Replace all occurrences of use_auth_token as a variable/parameter name
                # This includes cases like token=use_auth_token which should become token=token
                line = re.sub(r'\buse_auth_token\b', 'token', line)
            
            new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Patched: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        pass

