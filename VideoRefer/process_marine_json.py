#!/usr/bin/env python3
"""
Script to process marine.json and add <video>\n prefix to human conversation values.
"""

import json
import sys
from pathlib import Path


def process_marine_json(input_file, output_file=None):
    """
    Process a JSON file by adding <video>\n to the beginning of human conversation values.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to the output JSON file. If None, overwrites input.
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Error: File '{input_file}' not found.")
        return False
    
    try:
        # Read the JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if not isinstance(data, list):
            print("⚠️  Warning: JSON root is not a list. Attempting to process it as a single object...")
            data = [data]
        
        # Process each entry
        modified_count = 0
        for entry in data:
            if 'conversations' in entry and isinstance(entry['conversations'], list):
                for conv in entry['conversations']:
                    # Check if this is a human message
                    if conv.get('from') == 'human' and 'value' in conv:
                        value = conv['value']
                        # Only add <video>\n if it's not already there
                        if not value.startswith('<video>'):
                            conv['value'] = f'<video>\n{value}'
                            modified_count += 1
        
        # Determine output file
        if output_file is None:
            output_path = input_path
        else:
            output_path = Path(output_file)
        
        # Write the modified data back to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Successfully processed '{input_file}'")
        print(f"   Modified {modified_count} human conversation value(s)")
        print(f"   Output saved to '{output_path}'")
        return True
    
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file - {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_marine_json.py <input_file> [output_file]")
        print()
        print("Example:")
        print("  python process_marine_json.py marine.json")
        print("  python process_marine_json.py marine.json marine_processed.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = process_marine_json(input_file, output_file)
    sys.exit(0 if success else 1)
