import json
import os
import argparse
import sys

def check_file(filepath, file_type="results"):
    print(f"\n--- Checking {file_type.upper()} file: {filepath} ---")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return 0

    count = 0
    valid_json = 0
    has_mechanistic = 0
    
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                count += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    valid_json += 1
                    
                    # Specific checks based on file type
                    if file_type == "results":
                        # Check for ID and Pred
                        if "id" not in data:
                            print(f"⚠️  Line {i+1}: Missing 'id' field")
                        
                        # Check if mechanistic stats exist
                        mech = data.get("mechanistic")
                        if mech and isinstance(mech, dict) and len(mech) > 0:
                            has_mechanistic += 1
                            
                    elif file_type == "trace":
                        # Trace file usually contains raw activations
                        mech = data.get("mechanistic")
                        if mech and isinstance(mech, dict) and len(mech) > 0:
                            has_mechanistic += 1
                            # Optional: Check first neuron to ensure it's a list of floats
                            first_key = next(iter(mech))
                            if not isinstance(mech[first_key], list):
                                print(f"⚠️  Line {i+1}: Mechanistic data is not a list (Found {type(mech[first_key])})")

                except json.JSONDecodeError as e:
                    print(f"❌ Line {i+1}: Corrupt JSON - {e}")
                except Exception as e:
                    print(f"❌ Line {i+1}: Unexpected error - {e}")

        print(f"✅ Read {valid_json}/{count} valid lines.")
        
        if file_type == "results":
            print(f"📊 Entries with Mechanistic Stats: {has_mechanistic}/{valid_json}")
        elif file_type == "trace":
            print(f"🧠 Entries with Raw Activations: {has_mechanistic}/{valid_json}")

        return valid_json

    except Exception as e:
        print(f"❌ Critical Error reading file: {e}")
        return 0

def main():
    
    results_path = "unified_pipeline/results/gsm8k/unsloth/Meta-Llama-3.1-8B/entropic_logit_gap_mech_interp_heuristic/max_dev_var_10/test1_results.jsonl"
    trace_path = "unified_pipeline/results/gsm8k/unsloth/Meta-Llama-3.1-8B/entropic_logit_gap_mech_interp_heuristic/max_dev_var_10/test1_trace.jsonl"

    # 2. Check Results
    results_count = check_file(results_path, "results")

    # 3. Check Traces
    trace_count = check_file(trace_path, "trace")

    # 4. Compare
    print("\n--- Integrity Report ---")
    if results_count == trace_count:
        print(f"✅ SUCCESS: Counts match ({results_count})")
    else:
        print(f"⚠️  WARNING: Mismatch! Results: {results_count} vs Traces: {trace_count}")
        print("    (This implies alignment is broken between the two files)")

if __name__ == "__main__":
    main()