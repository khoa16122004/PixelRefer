"""Python script to generate JSON dataset from OceanCap CSV format."""

import json
import os
import math
from typing import Any, Dict, List, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import pandas as pd
from tqdm import tqdm

flags.DEFINE_string("csv_path", "data/raw/OceanCap - remote - MSC.csv", "Input csv path.")
flags.DEFINE_string("example_json_path", "data/examples/stage2_example.json", "Input example json path for annotations.")
flags.DEFINE_string("output_path", "data/processed/marine.json", "Output json path.")
flags.DEFINE_string("video_folder", "", "Optional: Output video folder prefix.")
FLAGS = flags.FLAGS

# Fixed prompt for the conversation
HUMAN_PROMPT = (
    "Describe the video"
)

def process_video_row(
    row: pd.Series, num_columns: int, video_json_map: Dict[str, List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Process a single row from the CSV and aggregate matching JSON segments."""
    video_name_raw = row.iloc[0]  # Column 0: Video
    
    if pd.isna(video_name_raw) or not str(video_name_raw).strip():
        return None

    video_name = str(video_name_raw).strip()
    if not video_name.lower().endswith(".mp4"):
        video_name += ".mp4"

    # Intersection check: Only process if video exists in annotations
    if video_name not in video_json_map:
        return None
    
    # Get List of JSON items for this video, assumed to be in temporal order
    json_items = video_json_map[video_name]
    
    grouped_boundaries = []
    grouped_conversations = []
    grouped_annotations = []

    # Structure: Video, Summary, [Start, End, Gemini, GPT, Qwen] * N
    # First data column group starts at index 2
    segment_index = 0
    
    for i in range(2, num_columns, 5):
        # Ensure we have a full group of columns
        if i + 4 >= num_columns:
            break

        # Stop if we run out of JSON matched items
        if segment_index >= len(json_items):
            break

        # Extract columns indices
        start_idx = i
        end_idx = i + 1
        gemini_idx = i + 2
        gpt_idx = i + 3
        qwen_idx = i + 4

        # Extract values
        try:
            start_val = row.iloc[start_idx]
            end_val = row.iloc[end_idx]
            gemini_val = row.iloc[gemini_idx]
            gpt_val = row.iloc[gpt_idx]
            qwen_val = row.iloc[qwen_idx]
        except IndexError:
            break

        # Check if this segment is valid
        if pd.isna(start_val):
            continue

        try:
            s_float = float(start_val)
            e_float = float(end_val) if not pd.isna(end_val) else s_float
        except (ValueError, TypeError):
            continue
            
        # Match with the JSON item at the current segment_index
        json_item = json_items[segment_index]
        match_annotation = json_item.get("annotation", [])

        # Filter out segments with empty annotations
        # if not match_annotation:
        #     segment_index += 1
        #     continue

        # Build conversation list
        conversation = [{"from": "human", "value": HUMAN_PROMPT}]
        
        # Helper function to get value or NaN
        def get_val_or_nan(val):
            if isinstance(val, str) and val.strip():
                return val.strip()
            return float('nan')

        conversation.append({"from": "GPT", "value": get_val_or_nan(gpt_val)})
        conversation.append({"from": "Gemini", "value": get_val_or_nan(gemini_val)})
        conversation.append({"from": "Qwen", "value": get_val_or_nan(qwen_val)})

        # Collect data
        grouped_boundaries.append([s_float, e_float])
        grouped_conversations.append(conversation)
        grouped_annotations.append(match_annotation)
        
        segment_index += 1

    # If no segments matched, return None
    if not grouped_boundaries:
        return None

    # Construct group matching video object
    return {
        "video": video_name,
        "timestamp": grouped_boundaries,
        "conversation": grouped_conversations,
        "annotation": grouped_annotations
    }


def main(argv):
    del argv  # Unused.

    if not os.path.exists(FLAGS.csv_path):
        logging.error("CSV file not found at %s", FLAGS.csv_path)
        return

    # Load matched annotations
    if not os.path.exists(FLAGS.example_json_path):
        logging.error("Example JSON file not found at %s", FLAGS.example_json_path)
        return

    logging.info("Reading example JSON for annotations...")
    with open(FLAGS.example_json_path, "r", encoding="utf-8") as f:
        example_data = json.load(f)
    
    # Group JSON items by video name
    video_json_map = {}
    for item in example_data:
        v_name = item.get("video")
        if not v_name:
            continue
        
        if v_name not in video_json_map:
            video_json_map[v_name] = []
        video_json_map[v_name].append(item)

    # Note: We assume the order in JSON matches the order in CSV if multiple clips exists for one video.
    
    logging.info("Reading input CSV...")
    df = pd.read_csv(FLAGS.csv_path)
    num_columns = len(df.columns)
    
    output_data = []
    global_id_counter = 0
    
    logging.info("Processing rows...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        video_entry = process_video_row(row, num_columns, video_json_map)
        if video_entry:
            video_entry["id"] = f"MSC_clip_{global_id_counter}"
            # Move id to the front for better readability (optional, but requested implicitly by JSON structure)
            # Python dictionaries rely on insertion order since 3.7
            # ordered_entry = {"id": video_entry["id"]}
            ordered_entry = video_entry
            # ordered_entry.update(video_entry)
            
            output_data.append(ordered_entry)
            global_id_counter += 1

    logging.info("Writing output to %s...", FLAGS.output_path)
    with open(FLAGS.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    logging.info("Successfully generated %d items.", len(output_data))


if __name__ == "__main__":
    app.run(main)
