import pandas as pd
import os

def extract_impression_section(text):
    lines = text.split('\n')
    impression_lines = []
    found = False

    for line in lines:
        if not found and 'IMPRESSION' in line.upper():
            found = True
            parts = line.upper().split('IMPRESSION', maxsplit=1)
            after = line[len(parts[0]) + len("IMPRESSION"):].strip(": ").strip()
            if after:
                impression_lines.append(after)
        elif found:
            impression_lines.append(line.strip())

    return '\n'.join(impression_lines).strip() if impression_lines else None

def build_final_dataset(labels_csv, report_folder, paths_csv, output_csv):
    df_labels = pd.read_csv(labels_csv, dtype={"study_id": str})
    df_paths = pd.read_csv(paths_csv, dtype={"study_id": str})

    # Extract just the filename from the path
    df_paths["filename"] = df_paths["path"].apply(os.path.basename)

    # Merge on study_id
    df = pd.merge(df_labels, df_paths[["study_id", "filename"]], on="study_id", how="inner")

    # Extract impressions
    impressions = []
    missing = 0

    for _, row in df.iterrows():
        file_path = os.path.join(report_folder, row["filename"])
        if not os.path.exists(file_path):
            impressions.append(None)
            missing += 1
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            impression = extract_impression_section(text)
            impressions.append(impression)

    df["Report Impression"] = impressions
    df = df.dropna(subset=["Report Impression"])

    label_columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                     'Airspace Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    final_df = df[label_columns + ["Report Impression"]]
    final_df[label_columns] = final_df[label_columns].fillna(0)

    final_df.to_csv(output_csv, index=False)


build_final_dataset(
    labels_csv="labels.csv",
    report_folder="mimic_cxr_reports",
    paths_csv="paths.csv",
    output_csv="clean_dataset.csv"
)
