#!/usr/bin/env python3
"""
Patch whisperx diarize.py for pyannote.audio>=4.0.1 DiarizeOutput API compatibility.
"""
import re
import sys
import glob

# Find all diarize.py files
diarize_files = glob.glob('/usr/local/lib/python3.*/dist-packages/whisperx/diarize.py')

for filepath in diarize_files:
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_content = content

        # Find and replace the problematic line
        # Pattern: pd.DataFrame(...itertracks...)
        # Replace with code that handles DiarizeOutput
        
        # First, find the variable name used (usually "diarization")
        var_match = re.search(r'(\w+)(?:\.annotation|\))?\.itertracks', content)
        var_name = var_match.group(1) if var_match else 'diarization'
        
        # Replace the problematic DataFrame line
        # Match: pd.DataFrame([(segment, label, label) for ... in (var.annotation if hasattr(var, "annotation") else var).itertracks(...)], ...)
        # Or: pd.DataFrame(var.itertracks(...), ...)
        
        pattern1 = r'pd\.DataFrame\(\[\(segment, label, label\) for segment, track, label in \([^)]+\)\.itertracks\(yield_label=True\)\], columns=\[.+\]\)'
        pattern2 = r'pd\.DataFrame\([^)]+\.itertracks\(yield_label=True\)[^)]*\)'
        
        replacement = f'''# Compatibility fix for pyannote.audio>=4.0.1 DiarizeOutput
from pyannote.core import Annotation
# Get annotation from DiarizeOutput
if hasattr({var_name}, "annotation"):
    annotation_obj = {var_name}.annotation
elif isinstance({var_name}, Annotation):
    annotation_obj = {var_name}
else:
    annotation_obj = getattr({var_name}, "annotation", None) or getattr({var_name}, "_annotation", None)
    if annotation_obj is None:
        raise AttributeError(f"{var_name} does not have annotation attribute")
diarize_df = pd.DataFrame([(segment, label, label) for segment, track, label in annotation_obj.itertracks(yield_label=True)], columns=['segment', 'label', 'speaker'])'''

        # Try pattern 1 first (already patched version)
        if re.search(pattern1, content):
            content = re.sub(pattern1, replacement, content)
        # Try pattern 2 (original version)
        elif re.search(pattern2, content):
            content = re.sub(pattern2, replacement, content)
        # Try simpler pattern
        else:
            # Just replace any line with itertracks
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if 'pd.DataFrame' in line and 'itertracks' in line:
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    new_lines.append(indent_str + '# Compatibility fix for pyannote.audio>=4.0.1 DiarizeOutput')
                    new_lines.append(indent_str + 'from pyannote.core import Annotation')
                    new_lines.append(indent_str + f'# Get annotation from DiarizeOutput')
                    new_lines.append(indent_str + f'if hasattr({var_name}, "annotation"):')
                    new_lines.append(indent_str + f'    annotation_obj = {var_name}.annotation')
                    new_lines.append(indent_str + f'elif isinstance({var_name}, Annotation):')
                    new_lines.append(indent_str + f'    annotation_obj = {var_name}')
                    new_lines.append(indent_str + f'else:')
                    new_lines.append(indent_str + f'    annotation_obj = getattr({var_name}, "annotation", None) or getattr({var_name}, "_annotation", None)')
                    new_lines.append(indent_str + f'    if annotation_obj is None:')
                    new_lines.append(indent_str + f'        raise AttributeError(f"{var_name} does not have annotation attribute")')
                    new_lines.append(indent_str + 'diarize_df = pd.DataFrame([(segment, label, label) for segment, track, label in annotation_obj.itertracks(yield_label=True)], columns=[\'segment\', \'label\', \'speaker\'])')
                else:
                    new_lines.append(line)
            content = '\n'.join(new_lines)

        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Patched: {filepath}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        pass
