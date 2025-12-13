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
                var_match = re.search(r'(\w+)(?:\.annotation|\))?\.itertracks', line)
                if not var_match:
                    var_match = re.search(r'\((\w+)', line)
                var_name = var_match.group(1) if var_match else 'diarization'
                
                # Replace with code that handles DiarizeOutput properly
                # DiarizeOutput in pyannote.audio>=4.0.1 might be directly iterable
                # or might need to be converted differently
                new_lines.append(indent_str + '# Compatibility fix for pyannote.audio>=4.0.1 DiarizeOutput\n')
                new_lines.append(indent_str + 'from pyannote.core import Annotation\n')
                new_lines.append(indent_str + 'import inspect\n')
                new_lines.append(indent_str + f'# Handle DiarizeOutput - try multiple ways to get annotation\n')
                new_lines.append(indent_str + f'annotation_obj = None\n')
                new_lines.append(indent_str + f'# Method 1: Check if it has annotation attribute\n')
                new_lines.append(indent_str + f'if hasattr({var_name}, "annotation"):\n')
                new_lines.append(indent_str + f'    annotation_obj = {var_name}.annotation\n')
                new_lines.append(indent_str + f'# Method 2: Check if it is already an Annotation\n')
                new_lines.append(indent_str + f'elif isinstance({var_name}, Annotation):\n')
                new_lines.append(indent_str + f'    annotation_obj = {var_name}\n')
                new_lines.append(indent_str + f'# Method 3: Check if DiarizeOutput has _annotation (private attribute)\n')
                new_lines.append(indent_str + f'elif hasattr({var_name}, "_annotation"):\n')
                new_lines.append(indent_str + f'    annotation_obj = {var_name}._annotation\n')
                new_lines.append(indent_str + f'# Method 4: Check if DiarizeOutput is directly iterable (itertracks-like)\n')
                new_lines.append(indent_str + f'elif hasattr({var_name}, "__iter__"):\n')
                new_lines.append(indent_str + f'    # Try to iterate directly - DiarizeOutput might be iterable\n')
                new_lines.append(indent_str + f'    try:\n')
                new_lines.append(indent_str + f'        # Check if it has itertracks method\n')
                new_lines.append(indent_str + f'        if hasattr({var_name}, "itertracks"):\n')
                new_lines.append(indent_str + f'            annotation_obj = {var_name}\n')
                new_lines.append(indent_str + f'        else:\n')
                new_lines.append(indent_str + f'            # Convert iterable to Annotation\n')
                new_lines.append(indent_str + f'            annotation_obj = Annotation()\n')
                new_lines.append(indent_str + f'            # Try to iterate as (segment, track, label) tuples\n')
                new_lines.append(indent_str + f'            for item in {var_name}:\n')
                new_lines.append(indent_str + f'                if isinstance(item, tuple) and len(item) >= 3:\n')
                new_lines.append(indent_str + f'                    segment, track, label = item[0], item[1], item[2]\n')
                new_lines.append(indent_str + f'                    annotation_obj[segment, track] = label\n')
                new_lines.append(indent_str + f'                elif hasattr(item, "segment") and hasattr(item, "track") and hasattr(item, "label"):\n')
                new_lines.append(indent_str + f'                    annotation_obj[item.segment, item.track] = item.label\n')
                new_lines.append(indent_str + f'    except Exception:\n')
                new_lines.append(indent_str + f'        pass\n')
                new_lines.append(indent_str + f'# Method 5: Check all attributes to find annotation\n')
                new_lines.append(indent_str + f'if annotation_obj is None:\n')
                new_lines.append(indent_str + f'    # Inspect all attributes\n')
                new_lines.append(indent_str + f'    for attr_name in dir({var_name}):\n')
                new_lines.append(indent_str + f'        if attr_name.startswith("_"):\n')
                new_lines.append(indent_str + f'            continue\n')
                new_lines.append(indent_str + f'        attr = getattr({var_name}, attr_name)\n')
                new_lines.append(indent_str + f'        if isinstance(attr, Annotation) or (hasattr(attr, "itertracks") and callable(getattr(attr, "itertracks"))):\n')
                new_lines.append(indent_str + f'            annotation_obj = attr\n')
                new_lines.append(indent_str + f'            break\n')
                new_lines.append(indent_str + f'# Final fallback: try to use DiarizeOutput directly if it has itertracks\n')
                new_lines.append(indent_str + f'if annotation_obj is None:\n')
                new_lines.append(indent_str + f'    if hasattr({var_name}, "itertracks") and callable(getattr({var_name}, "itertracks")):\n')
                new_lines.append(indent_str + f'        annotation_obj = {var_name}\n')
                new_lines.append(indent_str + f'    else:\n')
                new_lines.append(indent_str + f'        # Last resort: print debug info and raise error\n')
                new_lines.append(indent_str + f'        import pprint\n')
                new_lines.append(indent_str + f'        print(f"DEBUG: {var_name} type: {{type({var_name})}}", file=sys.stderr)\n')
                new_lines.append(indent_str + f'        print(f"DEBUG: {var_name} dir: {{[x for x in dir({var_name}) if not x.startswith(\'_\')]}}", file=sys.stderr)\n')
                new_lines.append(indent_str + f'        raise AttributeError(f"{var_name} does not have accessible annotation or itertracks method. Type: {{type({var_name})}}, Attributes: {{[x for x in dir({var_name}) if not x.startswith(\'_\')]}}")\n')
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
        import traceback
        traceback.print_exc(file=sys.stderr)
        pass
