#!/usr/bin/env python3
"""
Script to convert wandb run names to timestamps and match them with weight folders.

Usage:
    python time_conversion.py "run-20260205_192938-00uc1yjr"
    python time_conversion.py "apollo-lab/stream-rl-robotics-final/81vp1ksa"
    python time_conversion.py "81vp1ksa"
    python time_conversion.py --timestamp 1771802098
"""

import datetime
import argparse
import os
from pathlib import Path


def find_run_by_id(run_id, wandb_dir="wandb"):
    """
    Find wandb run folder by run ID.
    
    Args:
        run_id: Run ID (e.g., "81vp1ksa")
        wandb_dir: Directory containing wandb runs
    
    Returns:
        Full run folder name or None if not found
    """
    wandb_path = Path(wandb_dir)
    if not wandb_path.exists():
        return None
    
    for folder in wandb_path.iterdir():
        if folder.is_dir() and folder.name.startswith("run-"):
            # Format: run-20260220_043837-81vp1ksa
            if folder.name.endswith(f"-{run_id}"):
                return folder.name
    
    return None


def parse_wandb_input(input_str, wandb_dir="wandb"):
    """
    Parse various wandb input formats.
    
    Args:
        input_str: Can be:
            - Full run name: "run-20260205_192938-00uc1yjr"
            - Wandb path: "apollo-lab/stream-rl-robotics-final/81vp1ksa"
            - Just run ID: "81vp1ksa"
        wandb_dir: Directory containing wandb runs
    
    Returns:
        Full run folder name
    """
    # If it contains slashes, extract the run ID (last part)
    if "/" in input_str:
        run_id = input_str.split("/")[-1]
        run_name = find_run_by_id(run_id, wandb_dir)
        if run_name:
            return run_name
        raise ValueError(f"Run ID '{run_id}' not found in {wandb_dir}")
    
    # If it starts with "run-", use it directly
    if input_str.startswith("run-"):
        return input_str
    
    # Otherwise, assume it's just a run ID
    run_name = find_run_by_id(input_str, wandb_dir)
    if run_name:
        return run_name
    
    raise ValueError(f"Run ID '{input_str}' not found in {wandb_dir}")


def wandb_run_to_timestamp(run_name):
    """
    Convert wandb run name to Unix timestamp.
    
    Args:
        run_name: Run name in format "run-YYYYMMDD_HHMMSS-{id}" or just "YYYYMMDD_HHMMSS"
    
    Returns:
        Unix timestamp (integer)
    """
    # Extract datetime part from run name
    if run_name.startswith("run-"):
        # Format: run-20260205_192938-00uc1yjr
        datetime_part = run_name.split("-", 1)[1].rsplit("-", 1)[0]
    else:
        datetime_part = run_name
    
    # Parse: YYYYMMDD_HHMMSS
    dt = datetime.datetime.strptime(datetime_part, "%Y%m%d_%H%M%S")
    
    # Convert to timestamp
    timestamp = int(dt.timestamp())
    
    return timestamp


def timestamp_to_datetime(timestamp):
    """Convert Unix timestamp to human-readable datetime."""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def find_matching_weights(timestamp, weights_dir="weights", tolerance=300):
    """
    Find weight folders with timestamps close to the given timestamp.
    
    Args:
        timestamp: Unix timestamp to search for
        weights_dir: Directory containing weight folders
        tolerance: Time tolerance in seconds (default: 5 minutes)
    
    Returns:
        List of matching folder names
    """
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        return []
    
    matches = []
    
    for folder in weights_path.iterdir():
        if folder.is_dir():
            # Extract timestamp from folder name
            parts = folder.name.split("_")
            if len(parts) >= 2:
                try:
                    folder_timestamp = int(parts[-1])
                    # Check if timestamps are within tolerance
                    if abs(folder_timestamp - timestamp) <= tolerance:
                        time_diff = folder_timestamp - timestamp
                        matches.append((folder.name, folder_timestamp, time_diff))
                except ValueError:
                    continue
    
    # Sort by time difference
    matches.sort(key=lambda x: abs(x[2]))
    
    return matches


def main():
    parser = argparse.ArgumentParser(
        description="Convert wandb run names to timestamps and find matching weight folders"
    )
    parser.add_argument(
        "run_input",
        nargs="?",
        help='Wandb run (e.g., "81vp1ksa", "apollo-lab/project/81vp1ksa", or "run-20260205_192938-00uc1yjr")'
    )
    parser.add_argument(
        "--timestamp", "-t",
        type=int,
        help="Convert timestamp to datetime (reverse operation)"
    )
    parser.add_argument(
        "--find-match", "-f",
        action="store_true",
        help="Find matching weight folders (enabled by default)"
    )
    parser.add_argument(
        "--no-match",
        action="store_true",
        help="Don't search for matching weight folders"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=300,
        help="Time tolerance in seconds for matching (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--weights-dir", "-w",
        default="weights",
        help="Path to weights directory (default: weights)"
    )
    parser.add_argument(
        "--wandb-dir",
        default="wandb",
        help="Path to wandb directory (default: wandb)"
    )
    
    args = parser.parse_args()
    
    # Reverse operation: timestamp to datetime
    if args.timestamp:
        dt_str = timestamp_to_datetime(args.timestamp)
        print(f"Timestamp: {args.timestamp}")
        print(f"Datetime:  {dt_str}")
        return
    
    # Main operation: run name to timestamp
    if not args.run_input:
        parser.print_help()
        return
    
    try:
        # Parse the input (handles run ID, wandb path, or full run name)
        run_name = parse_wandb_input(args.run_input, args.wandb_dir)
        timestamp = wandb_run_to_timestamp(run_name)
        dt_str = timestamp_to_datetime(timestamp)
        
        print(f"Run name:  {run_name}")
        print(f"Timestamp: {timestamp}")
        print(f"Datetime:  {dt_str}")
        
        # Find matching weight folders by default (unless --no-match is specified)
        if not args.no_match:
            print(f"\nSearching for matches in '{args.weights_dir}' (tolerance: ±{args.tolerance}s)...")
            matches = find_matching_weights(
                timestamp, 
                weights_dir=args.weights_dir,
                tolerance=args.tolerance
            )
            
            if matches:
                print(f"\nFound {len(matches)} matching folder(s):")
                for folder_name, folder_ts, diff in matches:
                    diff_str = f"+{diff}s" if diff > 0 else f"{diff}s"
                    print(f"  • {folder_name}")
                    print(f"    └─ Timestamp: {folder_ts} ({diff_str} difference)")
            else:
                print("\nNo matching folders found.")
                print(f"Try increasing --tolerance (current: {args.tolerance}s)")
    
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
