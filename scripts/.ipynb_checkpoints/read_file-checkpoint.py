import os
from striprtf.striprtf import rtf_to_text

def read_file(file_path):
    try:
        # Attempt to read as RTF
        with open(file_path, 'r') as file:
            rtf_content = file.read()
            # Try converting to plain text
            story = rtf_to_text(rtf_content)
            # Check if the conversion was successful (basic check)
            if '{\\rtf' not in rtf_content:
                raise ValueError("Not an RTF file.")
    except ValueError:
        # If it's not an RTF file, read as plain text
        with open(file_path, 'r') as file:
            story = file.read()
    
    return story