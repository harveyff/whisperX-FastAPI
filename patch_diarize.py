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
            lines = f.readlines()

        original_lines = lines[:]
        new_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            # Find the problematic line with itertracks
            if 'pd.DataFrame' in line and 'itertracks' in line:
                # Extract indentation
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                
                # Extract variable name (diarization)
                var_match = re.search(r'(\w+)', line)
                var_name = var_match.group(1) if var_match else 'diarization'
                
                # Replace with fixed code
                new_lines.append(indent_str + '# Compatibility fix for pyannote.audio>=4.0.1 DiarizeOutput\n')
                new_lines.append(indent_str + 'from pyannote.core import Annotation\n')
                new_lines.append(indent_str + '# Handle DiarizeOutput which does not have itertracks directly\n')
                new_lines.append(indent_str + f'if hasattr({var_name}, "annotation"):\n')
                new_lines.append(indent_str + f'    annotation_obj = {var_name}.annotation\n')
                new_lines.append(indent_str + f'elif isinstance({var_name}, Annotation):\n')
                new_lines.append(indent_str + f'    annotation_obj = {var_name}\n')
                new_lines.append(indent_str + f'else:\n')
                new_lines.append(indent_str + f'    # Try to get annotation from DiarizeOutput\n')
                new_lines.append(indent_str + f'    annotation_obj = getattr({var_name}, "annotation", None)\n')
                new_lines.append(indent_str + f'    if annotation_obj is None:\n')
                new_lines.append(indent_str + f'        # DiarizeOutput might have different structure, try _annotation\n')
                new_lines.append(indent_str + f'        annotation_obj = getattr({var_name}, "_annotation", None)\n')
                new_lines.append(indent_str + f'    if annotation_obj is None:\n')
                new_lines.append(indent_str + f'        # Last resort: check if it has get_timeline or similar\n')
                new_lines.append(indent_str + f'        if hasattr({var_name}, "get_timeline"):\n')
                new_lines.append(indent_str + f'            annotation_obj = Annotation()\n')
                new_lines.append(indent_str + f'            for segment, track, label in {var_name}.get_timeline():\n')
                new_lines.append(indent_str + f'                annotation_obj[segment, track] = label\n')
                new_lines.append(indent_str + f'        else:\n')
                new_lines.append(indent_str + f'            raise AttributeError(f"{var_name} does not have annotation or itertracks method")\n')
                # Replace the original line with fixed version
                new_lines.append(indent_str + 'diarize_df = pd.DataFrame([(segment, label, label) for segment, track, label in annotation_obj.itertracks(yield_label=True)], columns=[\'segment\', \'label\', \'speaker\'])\n')
            else:
                new_lines.append(line)
            i += 1

        if new_lines != original_lines:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            print(f"Patched: {filepath}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        pass

