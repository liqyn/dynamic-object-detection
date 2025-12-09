import os
import re
import torch
import argparse # New import for handling command-line arguments
from tqdm import tqdm

def combine_tensors_by_index(folder_path: str, base_name: str):
    """
    Searches a folder for file pairs matching patterns and concatenates them.
    ... (Docstring remains the same) ...
    """
    
    # 1. Define the regex patterns
    pattern_normal = re.compile(rf"^{base_name}(\d{{5}})\.pt$")
    pattern_dino = re.compile(rf"^{base_name}(\d{{5}})dino\.pt$")
    
    # 2. Group files by their 5-digit index
    file_groups = {}
    
    print(f"Scanning folder: {folder_path}...")
    
    # Ensure the folder exists before attempting to list contents
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    for filename in os.listdir(folder_path):
        match_normal = pattern_normal.match(filename)
        if match_normal:
            index = match_normal.group(1)
            file_groups.setdefault(index, {})['normal'] = os.path.join(folder_path, filename)
            continue
            
        match_dino = pattern_dino.match(filename)
        if match_dino:
            index = match_dino.group(1)
            file_groups.setdefault(index, {})['dino'] = os.path.join(folder_path, filename)
            continue
            
    # 3. Process and Concatenate Matching Pairs
    
    print(f"Found {len(file_groups)} unique indices. Processing matches...")

    sorted_indices = sorted(file_groups.keys())
    
    for index in tqdm(sorted_indices, desc="Concatenating files"):
        group = file_groups[index]
        
        if 'normal' in group and 'dino' in group:
            path_normal = group['normal']
            path_dino = group['dino']
            
            output_filename = f"{base_name}{index}_combined.pt"
            output_path = os.path.join(folder_path, output_filename)
            
            try:
                # Load tensors (using map_location='cpu' for safer loading)
                tensor_normal = torch.load(path_normal, map_location='cpu')
                tensor_dino = torch.load(path_dino, map_location='cpu')
                
                if not isinstance(tensor_normal, torch.Tensor) or not isinstance(tensor_dino, torch.Tensor):
                    print(f"Skipping index {index}: Files do not contain valid PyTorch tensors.")
                    continue

                # Concatenate along the last dimension (dim=-1)
                combined_tensor = torch.cat((tensor_normal, tensor_dino), dim=-1)
                
                # Save the combined tensor (saving to disk will save it on the CPU by default)
                torch.save(combined_tensor, output_path)
                
            except Exception as e:
                print(f"Error processing index {index}: {e}")
                
    print("Processing complete.")

# --- Command-Line Argument Handling ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Combine pairs of PyTorch tensor files by index and concatenate them along the last dimension."
    )
    
    parser.add_argument(
        'folder_path',
        type=str,
        help="The path to the folder containing the .pt files (e.g., ../out)."
    )
    
    parser.add_argument(
        'base_name',
        type=str,
        help="The common base name for the files (e.g., 'two_objs'). Files should match 'base_name00001.pt' and 'base_name00001dino.pt'."
    )
    
    args = parser.parse_args()
    combine_tensors_by_index(args.folder_path, args.base_name)
