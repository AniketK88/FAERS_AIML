import pandas as pd
import json
from aetse.utils.db import get_duckdb_connection
import spacy

def calc_f1(gt_list, pred_list):
    gt_set = set([x.strip().lower() for x in gt_list if x.strip()])
    pred_set = set([x.strip().lower() for x in pred_list if x.strip()])
    
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
        
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def run_evaluation():
    # 1. Load Ground Truth
    gt_df = pd.read_csv('data/ground_truth/labeling_template.csv')
    gt_df = gt_df[gt_df['drugs_gt'].notna() & (gt_df['drugs_gt'] != '')]
    
    # 2. Connect to DB and pull pipeline results
    with get_duckdb_connection(read_only=True) as conn:
        res_df = conn.execute(f"""
            SELECT p.review_id, p.extracted_drugs as drugs, p.extracted_reactions as reactions, p.severity, p.meddra_pts as mapped_terms, r.review_text
            FROM pipeline_results p
            JOIN drug_reviews r ON p.review_id = r.review_id
            WHERE p.review_id IN (SELECT unnest(?))
        """, [gt_df['review_id'].tolist()]).df()
        
    if len(res_df) == 0:
        print(json.dumps({"error": "No pipeline results found for GT reviews"}))
        return
        
    # Merge
    merged = gt_df.merge(res_df, on='review_id')
    n_evaluated = len(merged)
    
    # Load scispacy baseline model
    try:
        nlp = spacy.load('en_ner_bc5cdr_md')
    except Exception:
        nlp = None
        
    drug_f1_scores = []
    reaction_f1_scores = []
    baseline_f1_scores = []
    severity_correct = 0
    total_extracted_reactions = 0
    total_mapped_reactions = 0
    
    for _, row in merged.iterrows():
        # GT lists
        gt_drugs = str(row['drugs_gt']).split(',') if pd.notna(row['drugs_gt']) else []
        gt_reactions = str(row['reactions_gt']).split(',') if pd.notna(row['reactions_gt']) else []
        
        # LLM Predictions
        pred_drugs = json.loads(row['drugs'] or '[]')
        pred_reactions = json.loads(row['reactions'] or '[]')
        
        drug_f1_scores.append(calc_f1(gt_drugs, pred_drugs))
        reaction_f1_scores.append(calc_f1(gt_reactions, pred_reactions))
        
        # Severity
        if str(row['severity_gt']).lower() == str(row['severity']).lower():
            severity_correct += 1
            
        # MedDRA mapping
        mapped_terms = json.loads(row['mapped_terms'] or '[]')
        total_extracted_reactions += len(pred_reactions)
        total_mapped_reactions += len(mapped_terms)
        
        # Baseline SciSpacy F1 for drugs
        if nlp:
            # We use the full review_text to be fair to the baseline
            doc = nlp(row['review_text'])
            baseline_drugs = [ent.text for ent in doc.ents if ent.label_ == 'CHEMICAL']
            baseline_f1_scores.append(calc_f1(gt_drugs, baseline_drugs))
            
    drug_f1_avg = sum(drug_f1_scores) / n_evaluated
    reaction_f1_avg = sum(reaction_f1_scores) / n_evaluated
    severity_accuracy = severity_correct / n_evaluated
    meddra_accuracy = total_mapped_reactions / total_extracted_reactions if total_extracted_reactions > 0 else 0.0
    scispacy_drug_f1_avg = sum(baseline_f1_scores) / n_evaluated if baseline_f1_scores else 0.0
    
    metrics = {
        "n_evaluated": n_evaluated,
        "drug_extraction_f1": round(drug_f1_avg, 3),
        "reaction_extraction_f1": round(reaction_f1_avg, 3),
        "severity_accuracy": round(severity_accuracy, 3),
        "meddra_mapping_accuracy": round(meddra_accuracy, 3),
        "scispacy_drug_f1_avg": round(scispacy_drug_f1_avg, 3)
    }
    
    print(json.dumps(metrics, indent=2))
    
    # Save to file
    with open('data/eval_results/eval_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
