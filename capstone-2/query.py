import os
import requests
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable
dune_api_key = os.getenv("DUNE_BASE_ML_PREDICTION_API_KEY")

if not dune_api_key:
    raise ValueError("API key not found in environment variables")

# ensure the output directory exists
os.makedirs("data", exist_ok=True)

query_ids = [6357600]
headers = {"X-DUNE-API-KEY": dune_api_key}

for query_id in query_ids:
    url = f"https://api.dune.com/api/v1/query/{query_id}/results/csv"
    response = requests.get(url, headers=headers)

    # Save raw CSV string directly to file
    csv_file = f"data/query_{query_id}_data.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved CSV for query {query_id}: {csv_file}")

    # If you also want to *load* the CSV into a DataFrame
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded CSV into DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print("Failed to load CSV into DataFrame:", e)

    # Optionally save the CSV as joblib too for later reuse
    joblib_file = f"data/query_{query_id}_data.joblib"
    joblib.dump(response.text, joblib_file)
    print(f"Saved joblib for query {query_id}: {joblib_file}")

    print(f"Query {query_id}: {response.status_code}")
