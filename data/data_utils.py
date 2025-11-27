
import pandas as pd
import requests
import io
import json

def load_gsm8k(limit=100):
    """
    Loads the GSM8K dataset. Attempts to fetch from the official repository, 
    otherwise falls back to mock data.
    """
    print(f"Attempting to load GSM8K data (limit={limit})...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    
    try:
        # Attempt to fetch real data
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.text.strip().split('\n')
        
        records = []
        count = 0
        for line in data:
            if count >= limit:
                break
            item = json.loads(line)
            # Extract the final answer from the solution text
            answer = item['answer'].split('####')[-1].strip().replace(',', '')
            records.append({
                'question': item['question'],
                'gold_answer': answer,
                'solution': item['answer']
            })
            count += 1
        
        print(f"Successfully loaded {len(records)} records.")
        return pd.DataFrame(records)

    except Exception as e:
        print(f"Failed to load GSM8K data from web: {e}. Returning mock data.")
        # Return dummy data if loading fails
        dummy_data = [
            {"question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remaining eggs for $2 each. How much money does she make daily?", "gold_answer": "18"},
            {"question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "gold_answer": "3"},
            {"question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Berlin. If the total number of sheep is 390. How many sheep does Toulouse have?", "gold_answer": "240"}
        ]
        return pd.DataFrame(dummy_data * (limit // len(dummy_data) + 1))[:limit]
