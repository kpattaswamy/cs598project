import pandas as pd

df_paths = pd.read_csv("paths.csv", dtype={"study_id": str})
df_labels = pd.read_csv("labels.csv", dtype={"study_id": str})

matched = df_paths[df_paths["study_id"].isin(df_labels["study_id"])]

base_url = "https://physionet.org/files/mimic-cxr/2.1.0/"
matched["full_url"] = base_url + matched["path"]

matched["full_url"].to_csv("mimic_report_urls.txt", index=False, header=False)
