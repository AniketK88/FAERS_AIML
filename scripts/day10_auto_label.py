import pandas as pd
import ollama
import json
import time

def label_row(row):
    prompt = f"""
You are a pharmacovigilance medical coder.
Review this 200-character preview of a drug review:
Text: "{row['review_text_preview']}"
Primary drug: "{row['drug_norm']}"

Extract the following information and return ONLY valid JSON:
1. "drugs_gt": A comma-separated list of all drugs mentioned. Include the primary drug if it's implied or explicitly stated. Prefer generic lowercase names. (e.g. "ibuprofen, aspirin")
2. "reactions_gt": A comma-separated list of ALL adverse reactions, side effects, or negative symptoms mentioned. Be specific (e.g., "stomach pain", "nausea"). If NO adverse reactions are mentioned, return an empty string "".
3. "severity_gt": EXACTLY ONE OF: "serious", "non-serious", "unknown".
   - "serious": hospitalization, ER visit, life-threatening, surgery, disability, stopped working, death, severe/extreme symptoms requiring medical intervention.
   - "non-serious": mild to moderate symptoms managed without hospitalization.
   - "unknown": cannot determine from the preview alone.
4. "notes": One brief sentence explaining the severity decision.

JSON Format:
{{
  "drugs_gt": "...",
  "reactions_gt": "...",
  "severity_gt": "...",
  "notes": "..."
}}
"""
    try:
        response = ollama.chat(
            model='llama3.1:8b-instruct-q4_K_M',
            messages=[{'role': 'user', 'content': prompt}],
            format='json',
            options={'temperature': 0.0}
        )
        res = json.loads(response['message']['content'])
        
        # Ensure severity is strictly one of the allowed values
        sev = str(res.get('severity_gt', 'unknown')).lower()
        if sev not in ['serious', 'non-serious', 'unknown']:
            sev = 'unknown'
            
        return pd.Series([
            str(res.get('drugs_gt', row['drug_norm'])).strip(), 
            str(res.get('reactions_gt', '')).strip(), 
            sev, 
            str(res.get('notes', '')).strip()
        ])
    except Exception as e:
        print(f"Error on row {row['review_id']}: {e}")
        return pd.Series([row['drug_norm'], '', 'unknown', 'Error parsing'])

def main():
    print("Reading labeling_template.csv...")
    df = pd.read_csv('data/ground_truth/labeling_template.csv')
    
    # Make sure NaN columns are strings
    for col in ['drugs_gt', 'reactions_gt', 'severity_gt', 'notes']:
        df[col] = df[col].fillna('')
    
    print("Labeling rows via local Ollama...")
    start_time = time.time()
    
    results = df.apply(label_row, axis=1)
    df[['drugs_gt', 'reactions_gt', 'severity_gt', 'notes']] = results
    
    df.to_csv('data/ground_truth/labeling_template.csv', index=False)
    
    print(f"\nFinished in {time.time() - start_time:.1f} seconds")
    
    print("\nSummary:")
    print(f"Total rows labeled: {len(df)}/50")
    print(f"Serious: {(df['severity_gt'] == 'serious').sum()}, Non-serious: {(df['severity_gt'] == 'non-serious').sum()}, Unknown: {(df['severity_gt'] == 'unknown').sum()}")
    
    reactions_empty = df['reactions_gt'].replace('', pd.NA).isna()
    print(f"Rows with reactions found: {(~reactions_empty).sum()}")
    print(f"Rows with empty reactions (no AE in preview): {reactions_empty.sum()}")

if __name__ == "__main__":
    main()
