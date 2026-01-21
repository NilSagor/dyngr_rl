# check_data_sizes.py
import numpy as np
import pandas as pd
from pathlib import Path

def check_wikipedia_data():
    data_dir = Path("experiment_framework/data/processed/wikipedia")
    
    print("=== Checking Wikipedia Data Sizes ===")
    
    # Check each file
    files = {
        "ml_wikipedia.csv": "CSV data",
        "ml_wikipedia.npy": "Edge features",
        "ml_wikipedia_node.npy": "Node features"
    }
    
    for filename, description in files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"✗ {filename} not found")
            continue
            
        try:
            if filename.endswith('.npy'):
                data = np.load(filepath)
                print(f"\n{filename} ({description}):")
                print(f"  Shape: {data.shape}")
                print(f"  Dtype: {data.dtype}")
                print(f"  Size in memory: {data.nbytes / (1024*1024):.2f} MB")
                
                # Check for problematic values
                if np.any(np.isnan(data)):
                    print(f"  WARNING: Contains NaN values!")
                if np.any(np.isinf(data)):
                    print(f"  WARNING: Contains Inf values!")
                if data.size > 0:
                    print(f"  Min: {data.min()}, Max: {data.max()}")
                    
                    # Check if indices are within bounds
                    if filename == "ml_wikipedia_node.npy":
                        if data.shape[0] > 100000:
                            print(f"  WARNING: Very large number of nodes: {data.shape[0]}")
                
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                print(f"\n{filename} ({description}):")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")
                
                # Check specific columns
                if 'u' in df.columns and 'i' in df.columns:
                    print(f"  Node indices - u: {df['u'].min()} to {df['u'].max()}")
                    print(f"  Node indices - i: {df['i'].min()} to {df['i'].max()}")
                    
                    # Check for negative indices
                    if (df['u'] < 0).any() or (df['i'] < 0).any():
                        print(f"  WARNING: Negative node indices found!")
                    
                if 'ts' in df.columns:
                    print(f"  Timestamps: {df['ts'].min()} to {df['ts'].max()}")
                    
        except Exception as e:
            print(f"\n✗ Error loading {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Check if dimensions match
    try:
        node_features = np.load(data_dir / "ml_wikipedia_node.npy")
        edge_features = np.load(data_dir / "ml_wikipedia.npy")
        df = pd.read_csv(data_dir / "ml_wikipedia.csv")
        
        print(f"\n=== Consistency Check ===")
        print(f"Node features shape: {node_features.shape}")
        print(f"Edge features shape: {edge_features.shape}")
        print(f"Number of edges in CSV: {len(df)}")
        
        if len(df) != edge_features.shape[0]:
            print(f"WARNING: Mismatch between CSV edges ({len(df)}) and edge features ({edge_features.shape[0]})")
        
    except:
        pass

if __name__ == "__main__":
    check_wikipedia_data()