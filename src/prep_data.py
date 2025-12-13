import pandas as pd

def split_by_time_gcs(
    input_gcs_path: str,
    output_bucket_path: str
):
    # Read CSV directly from GCS
    df = pd.read_csv(input_gcs_path)

    # Sort by Time column
    df = df.sort_values("Time").reset_index(drop=True)

    # Split into two halves
    midpoint = len(df) // 2
    df_2022 = df.iloc[:midpoint]
    df_2023 = df.iloc[midpoint:]

    # Output paths in GCS
    out_2022 = f"{output_bucket_path}/orig_data/v0/transactions_2022.csv"
    out_2023 = f"{output_bucket_path}/orig_data/v1/transactions_2023.csv"

    # Write back to GCS
    df_2022.to_csv(out_2022, index=False)
    df_2023.to_csv(out_2023, index=False)

    print("✅ Data split completed and uploaded to GCS")
    print(f"2022 rows: {len(df_2022)} → {out_2022}")
    print(f"2023 rows: {len(df_2023)} → {out_2023}")

if __name__ == "__main__":
    split_by_time_gcs(
        input_gcs_path="gs://practice-mlops-oppe/data/data.csv",
        output_bucket_path="gs://practice-mlops-oppe"
    )