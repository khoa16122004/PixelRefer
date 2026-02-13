import torch
from typing import List, Dict, Optional, Union
import transformers


def timestampify_pt(
    start: torch.Tensor,
    end: torch.Tensor,
    duration: Union[float, torch.Tensor],
    abs_time_token: bool = False,
    num_bins: int = 100,
    vocabulary_size: int = 32128,
    time_format: str = 'se',
    t: int = 1000000
) -> torch.Tensor:
    """Tokenize timestamps into discrete time token IDs.
    
    Converts continuous time values (start, end) into discrete token indices
    that are offset from the text vocabulary to avoid collision. Time tokens
    can be either relative (normalized by video duration) or absolute.
    
    Args:
        start: Start times of events in seconds. Shape: [n_events]
        end: End times of events in seconds. Shape: [n_events]
        duration: Duration of the video in seconds. Can be float or Tensor.
        abs_time_token: If True, use absolute time tokens. If False, use
            relative time tokens normalized by duration. Default: False
        num_bins: Number of quantization bins for discretizing time values.
            Default: 100 (supports 0-99 time token IDs)
        vocabulary_size: Size of text vocabulary. Time tokens are offset by
            this value to avoid collision with text tokens. Default: 32128
        time_format: Format for timestamp representation:
            'se' (start-end): Use [start, end] as-is
            'cd' (center-duration): Use [(start+end)/2, end-start]
            Default: 'se'
        t: Scaling factor for absolute time tokens (FPS * 1000000).
            Default: 1000000 (equivalent to 1 FPS)
    
    Returns:
        Tensor of shape [n_events, 2] containing time token IDs.
        Each row contains [start_token, end_token] or [center_token, duration_token]
        depending on time_format.
    """
    # Convert duration to tensor if needed for consistent operations
    if isinstance(duration, (int, float)):
        duration = torch.tensor(duration, dtype=torch.float64, device=start.device)
    
    # Format timestamps based on time_format parameter
    if time_format == 'cd':
        # Center-duration format: [center=(start+end)/2, duration=end-start]
        timestamp = torch.stack([(start + end) / 2, end - start], dim=1)
    else:
        # Start-end format (default): [start, end]
        timestamp = torch.stack([start, end], dim=1)
    
    # Clamp timestamps to not exceed video duration
    timestamp = torch.minimum(timestamp, duration)
    
    if not abs_time_token:
        # Relative time tokens: normalized by video duration
        # Formula: token = (time / duration) * (num_bins - 1) + vocabulary_size
        max_offset = num_bins - 1
        rel_timestamp = timestamp / duration  # Normalize to [0, 1]
        timestamp_token = (rel_timestamp * max_offset).long() + vocabulary_size
    else:
        # Absolute time tokens: based on actual time values
        # Formula: token = int(time / t) + vocabulary_size
        timestamp_token = (timestamp / t).long() + vocabulary_size
    
    return timestamp_token


def merge_cap_time_tokens_pt(
    caption_tokens: torch.Tensor,
    timestamp_tokens: torch.Tensor,
    order: str = 'ld'
) -> torch.Tensor:
    """Merge caption tokens and timestamp tokens into unified sequences.
    
    Combines text token sequences with temporal token pairs to create
    event sequences that contain both semantic (caption) and temporal
    (timestamp) information. The order parameter controls whether timestamps
    appear before or after the caption text.
    
    Args:
        caption_tokens: Text tokens for each event caption.
            Shape: [n_events, caption_seq_len]
            Expected format: [BOS, word1, word2, ..., wordN, EOS, PAD, ...]
        timestamp_tokens: Time tokens for each event.
            Shape: [n_events, 2] where each row is [start_token, end_token]
        order: Order of merging:
            'ld' (location-description): timestamps first, then caption
                Result: [BOS, start, end, word1, word2, ..., wordN]
            'dl' (description-location): caption first, then timestamps
                Result: [word1, word2, ..., wordN, start, end]
            Default: 'ld'
    
    Returns:
        Merged sequence tensor of shape [n_events, merged_seq_len]
        where merged_seq_len = caption_seq_len + 2 (for the two time tokens)
    """
    if order == 'ld':
        # Location-description: [BOS, timestamps, caption_text]
        # Keep BOS (first token), insert timestamps, then add caption text
        seq = torch.cat([
            caption_tokens[:, :1],      # BOS token
            timestamp_tokens,            # [start_token, end_token]
            caption_tokens[:, 1:-2]     # caption words (skip BOS and last 2 tokens)
        ], dim=1)
    else:
        # Description-location: [caption_text, timestamps]
        # Take caption without last 2 tokens, then add timestamps
        seq = torch.cat([
            caption_tokens[:, :-2],     # caption tokens (excluding last 2)
            timestamp_tokens            # [start_token, end_token]
        ], dim=1)
    
    return seq


def process_event_timestamps(
    captions: List[str],
    starts: List[float],
    ends: List[float],
    duration: float,
    tokenizer: transformers.PreTrainedTokenizer,
    abs_time_token: bool = False,
    num_bins: int = 100,
    vocabulary_size: int = 32128,
    time_format: str = 'se',
    order: str = 'ld',
    max_caption_length: int = 128,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """Process multiple events with captions and timestamps into tokenized sequences.
    
    This is the main high-level function that combines tokenization of both
    captions (text) and timestamps (temporal) into unified event sequences.
    It handles the complete pipeline:
    1. Tokenize caption text strings
    2. Convert timestamps to time tokens
    3. Merge caption and time tokens
    4. Flatten and prepare for model input
    
    Args:
        captions: List of caption text strings, one per event.
        starts: List of event start times in seconds.
        ends: List of event end times in seconds.
        duration: Total video duration in seconds.
        tokenizer: Pretrained tokenizer for text encoding.
        abs_time_token: Use absolute vs relative time tokens. Default: False
        num_bins: Number of time quantization bins. Default: 100
        vocabulary_size: Text vocabulary size for token offset. Default: 32128
        time_format: 'se' (start-end) or 'cd' (center-duration). Default: 'se'
        order: 'ld' (timestamps first) or 'dl' (text first). Default: 'ld'
        max_caption_length: Maximum caption length in tokens. Default: 128
        device: Device to place tensors on. Default: 'cpu'
    
    Returns:
        Dictionary containing:
        - 'event_token_ids': Flattened tensor of all merged event tokens
        - 'event_attention_mask': Attention mask (1 for real tokens, 0 for padding)
        - 'num_events': Number of events processed
    """
    if len(captions) == 0:
        return {
            'event_token_ids': torch.zeros(0, dtype=torch.long, device=device),
            'event_attention_mask': torch.zeros(0, dtype=torch.long, device=device),
            'num_events': 0
        }
    
    # Validate input lengths match
    assert len(captions) == len(starts) == len(ends), \
        f"Mismatched lengths: captions={len(captions)}, starts={len(starts)}, ends={len(ends)}"
    
    # Step 1: Tokenize captions
    encoded = tokenizer(
        captions,
        padding='max_length',
        truncation=True,
        max_length=max_caption_length,
        return_tensors='pt'
    )
    caption_tokens = encoded['input_ids'].to(device)
    
    # Step 2: Convert timestamps to time tokens
    start_tensor = torch.tensor(starts, dtype=torch.float32, device=device)
    end_tensor = torch.tensor(ends, dtype=torch.float32, device=device)
    
    timestamp_tokens = timestampify_pt(
        start=start_tensor,
        end=end_tensor,
        duration=duration,
        abs_time_token=abs_time_token,
        num_bins=num_bins,
        vocabulary_size=vocabulary_size,
        time_format=time_format
    )
    
    # Step 3: Merge caption and timestamp tokens
    merged_tokens = merge_cap_time_tokens_pt(
        caption_tokens=caption_tokens,
        timestamp_tokens=timestamp_tokens,
        order=order
    )
    
    # Step 4: Flatten merged tokens for model input
    flattened_tokens = merged_tokens.flatten()
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (flattened_tokens != tokenizer.pad_token_id).long()
    
    return {
        'event_token_ids': flattened_tokens,
        'event_attention_mask': attention_mask,
        'num_events': len(captions)
    }


def timestamp_to_time_token(
    start: float,
    end: float,
    duration: float,
    num_bins: int = 100
) -> List[str]:
    """Convert a timestamp range (start, end) and video duration to time token strings.
    
    Args:
        start: Start time in seconds.
        end: End time in seconds.
        duration: Video duration in seconds.
        num_bins: Number of bins for time quantization. Default: 100.
    
    Returns:
        List of two strings: [start_token_str, end_token_str]
    """
    if duration <= 0:
        duration = 1.0  # Avoid division by zero
        
    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    
    max_offset = num_bins - 1
    
    start_idx = int(round((start / duration) * max_offset))
    end_idx = int(round((end / duration) * max_offset))
    
    # Clamp indices to ensure they are within [0, num_bins-1]
    start_idx = max(0, min(start_idx, max_offset))
    end_idx = max(0, min(end_idx, max_offset))
    
    return [f"<time_{start_idx}>", f"<time_{end_idx}>"]
