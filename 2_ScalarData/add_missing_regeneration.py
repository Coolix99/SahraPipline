from pathlib import Path
import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
WT_FILE = DOWNLOADS / "WT_scalars.csv"
MEMBRANE_FILE = DOWNLOADS / "membrane_dynamics_FINAL.csv"
OUTPUT_FILE = DOWNLOADS / "WT_scalars_final.csv"
REPORT_FILE = DOWNLOADS / "WT_scalars_final_report.csv"

ID_COL = "Mask Folder"
GROUP_COLS = ["condition", "time in hpf", "experimentalist", "genotype"]


def read_csv_as_text(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def check_columns(df, required, filename):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")


wt = read_csv_as_text(WT_FILE)
membrane = read_csv_as_text(MEMBRANE_FILE)

check_columns(wt, [ID_COL] + GROUP_COLS, WT_FILE.name)
check_columns(membrane, [ID_COL] + GROUP_COLS, MEMBRANE_FILE.name)

# Compare IDs without accidental leading/trailing spaces.
wt["_merge_id"] = wt[ID_COL].str.strip()
membrane["_merge_id"] = membrane[ID_COL].str.strip()

for df, filename in [(wt, WT_FILE.name), (membrane, MEMBRANE_FILE.name)]:
    duplicates = df.loc[df["_merge_id"].duplicated(keep=False), ID_COL].tolist()
    if duplicates:
        raise ValueError(f"Duplicate '{ID_COL}' values in {filename}: {duplicates[:10]}")

# Only regeneration rows are candidates for addition.
membrane_reg = membrane[
    membrane["condition"].str.strip().str.casefold().eq("regeneration")
].copy()

wt_ids = set(wt["_merge_id"])
membrane_reg["already_in_WT"] = membrane_reg["_merge_id"].isin(wt_ids)
missing_reg = membrane_reg.loc[~membrane_reg["already_in_WT"]].copy()

# Match the WT schema: extra membrane columns are ignored; missing WT columns stay blank.
wt_columns = [column for column in wt.columns if column != "_merge_id"]
rows_to_add = missing_reg.reindex(columns=wt_columns, fill_value="")

final = pd.concat(
    [wt[wt_columns], rows_to_add],
    ignore_index=True,
)

if final[ID_COL].str.strip().duplicated().any():
    raise RuntimeError("Merge created duplicate sample IDs; output was not written.")

final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# Per-condition audit of all regeneration groups found in the membrane file.
membrane_reg["row_source"] = membrane_reg["already_in_WT"].map(
    {True: "already in WT_scalars.csv", False: "added from membrane_dynamics_FINAL.csv"}
)

report = (
    membrane_reg.groupby(GROUP_COLS, dropna=False)
    .agg(
        membrane_regeneration_rows=(ID_COL, "size"),
        already_in_WT=("already_in_WT", "sum"),
        added_from_membrane=("already_in_WT", lambda values: (~values).sum()),
        added_Mask_Folders=(
            ID_COL,
            lambda values: "; ".join(
                membrane_reg.loc[values.index]
                .loc[~membrane_reg.loc[values.index, "already_in_WT"], ID_COL]
            ),
        ),
    )
    .reset_index()
)

wt_before = wt.groupby(GROUP_COLS, dropna=False).size().rename("WT_rows_before")
wt_after = final.groupby(GROUP_COLS, dropna=False).size().rename("WT_rows_final")

report = (
    report.merge(wt_before, on=GROUP_COLS, how="left")
    .merge(wt_after, on=GROUP_COLS, how="left")
    .fillna({"WT_rows_before": 0, "WT_rows_final": 0})
)

for column in [
    "membrane_regeneration_rows",
    "already_in_WT",
    "added_from_membrane",
    "WT_rows_before",
    "WT_rows_final",
]:
    report[column] = report[column].astype(int)

report["status"] = report["added_from_membrane"].apply(
    lambda count: "correct already" if count == 0 else f"added {count} from membrane"
)
report["row_sources"] = report["added_from_membrane"].apply(
    lambda count: (
        "WT_scalars.csv"
        if count == 0
        else "WT_scalars.csv + membrane_dynamics_FINAL.csv"
    )
)

# Sort time points numerically when possible.
report["_time_sort"] = pd.to_numeric(report["time in hpf"], errors="coerce")
report = report.sort_values(
    ["condition", "_time_sort", "experimentalist", "genotype"],
    na_position="last",
).drop(columns="_time_sort")

report.to_csv(REPORT_FILE, index=False, encoding="utf-8")

print(f"WT rows before: {len(wt)}")
print(f"Regeneration rows in membrane source: {len(membrane_reg)}")
print(f"Rows added: {len(rows_to_add)}")
print(f"WT rows final: {len(final)}")
print(f"\nSaved merged file:\n  {OUTPUT_FILE}")
print(f"Saved condition report:\n  {REPORT_FILE}\n")

print(
    report[
        GROUP_COLS
        + [
            "WT_rows_before",
            "already_in_WT",
            "added_from_membrane",
            "WT_rows_final",
            "status",
        ]
    ].to_string(index=False)
)
