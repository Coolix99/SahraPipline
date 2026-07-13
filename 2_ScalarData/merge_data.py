import os
import sys
from typing import Dict, List

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *


WT_INPUT_FILE = os.path.join(scalar_path, 'scalarGrowthData_meshBased.csv')
WT_OUTPUT_FILE = os.path.join(scalar_path, 'WT_scalars.csv')

SMOC_CSV_FILE = os.path.join(
    scalar_path,
    'scalarGrowthData_meshBased_Smoc12.csv',
)
SMOC_DEV_FILE = os.path.join(
    scalar_path,
    'Smoc1_Smoc2_dev.xlsx',
)
SMOC_REG_FILE = os.path.join(
    scalar_path,
    'Smoc1_Smoc2_reg.xlsx',
)
SMOC_OUTPUT_FILE = os.path.join(
    scalar_path,
    'Smoc12_scalars.csv',
)


COLUMN_RENAMES: Dict[str, str] = {
    'Name': 'Mask Folder',
    'surface_area': 'Surface Area',
    'L_PD': 'L_PD_BB',
    'L_AP': 'L_AP_BB',
    'Time in hpf': 'time in hpf',
    'Experimentalist': 'experimentalist',
    'Genotype': 'genotype',
    'Condition': 'condition',
    'Is Control': 'Is control',
}


PREFERRED_COLUMN_ORDER: List[str] = [
    'Mask Folder',
    'Volume',
    'Surface Area',
    'Integrated Thickness',
    'Mean_thickness',
    'L_PD_BB',
    'L_PD_midline',
    'L_AP_BB',
    'L_AP_40line',
    'L_AP_longline',
    'PD_long_rel',
    'L_DV',
    'L_DV_npts',
    'condition',
    'time in hpf',
    'experimentalist',
    'genotype',
    'Is control',
]


def check_file_exists(file_path: str) -> None:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'Input file not found: {file_path}')


def load_csv(file_path: str) -> pd.DataFrame:
    check_file_exists(file_path)
    return pd.read_csv(file_path)


def load_excel(file_path: str) -> pd.DataFrame:
    check_file_exists(file_path)
    return pd.read_excel(file_path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    rename_map = {
        old_name: new_name
        for old_name, new_name in COLUMN_RENAMES.items()
        if old_name in df.columns
    }
    df = df.rename(columns=rename_map)

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            f'Duplicated columns after normalization: {duplicated_columns}'
        )

    if 'Mask Folder' not in df.columns:
        raise KeyError(
            "No 'Mask Folder' or equivalent 'Name' column found."
        )

    df['Mask Folder'] = (
        df['Mask Folder']
        .astype('string')
        .str.strip()
    )

    missing_names = (
        df['Mask Folder'].isna()
        | df['Mask Folder'].eq('')
    )
    if missing_names.any():
        raise ValueError(
            'Empty Mask Folder values found at rows: '
            f'{df.index[missing_names].tolist()}'
        )

    return df


def add_source_information(
    df: pd.DataFrame,
    source_file: str,
) -> pd.DataFrame:
    df = df.copy()
    df['_source_file'] = os.path.basename(source_file)
    return df


def raise_on_cross_file_duplicates(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    if 'Mask Folder' not in df.columns:
        raise KeyError(
            f"{dataset_name} has no 'Mask Folder' column."
        )

    if '_source_file' not in df.columns:
        raise KeyError(
            f"{dataset_name} has no '_source_file' column."
        )

    conflicts = []

    for mask_folder, group in df.groupby(
        'Mask Folder',
        dropna=False,
        sort=True,
    ):
        source_files = group['_source_file'].dropna().unique()

        if len(source_files) > 1:
            conflicts.append((mask_folder, group))

    if not conflicts:
        return

    print()
    print(f'Cross-file Mask Folder conflicts in {dataset_name}:')
    print('-' * 80)

    for mask_folder, group in conflicts:
        source_files = group['_source_file'].unique().tolist()

        print(f'Mask Folder: {mask_folder}')
        print(f'Files: {source_files}')

        for source_file, source_group in group.groupby('_source_file'):
            print(
                f'  {source_file}: rows '
                f'{source_group.index.tolist()}'
            )

        print()

    raise ValueError(
        f'Cross-file Mask Folder conflicts found in {dataset_name}.'
    )


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        column
        for column in PREFERRED_COLUMN_ORDER
        if column in df.columns
    ]

    other_columns = [
        column
        for column in df.columns
        if column not in preferred_columns
        and column != '_source_file'
    ]

    return df[preferred_columns + other_columns]


def print_column_mapping(
    source_file: str,
    original_columns: List[str],
    normalized_columns: List[str],
) -> None:
    print()
    print(os.path.basename(source_file))

    for old_column, new_column in zip(
        original_columns,
        normalized_columns,
    ):
        if old_column == new_column:
            print(f'  {old_column}')
        else:
            print(f'  {old_column} -> {new_column}')


def load_and_normalize_csv(file_path: str) -> pd.DataFrame:
    df = load_csv(file_path)
    original_columns = [str(column).strip() for column in df.columns]
    df = normalize_columns(df)

    print_column_mapping(
        file_path,
        original_columns,
        list(df.columns),
    )

    return add_source_information(df, file_path)


def load_and_normalize_excel(file_path: str) -> pd.DataFrame:
    df = load_excel(file_path)
    original_columns = [str(column).strip() for column in df.columns]
    df = normalize_columns(df)

    print_column_mapping(
        file_path,
        original_columns,
        list(df.columns),
    )

    return add_source_information(df, file_path)


def create_wt_dataset() -> pd.DataFrame:
    df = load_and_normalize_csv(WT_INPUT_FILE)

    if 'genotype' not in df.columns:
        raise KeyError(
            f"No 'genotype' column found in {WT_INPUT_FILE}"
        )

    genotype = (
        df['genotype']
        .astype('string')
        .str.strip()
    )

    wt_df = df.loc[genotype.eq('WT')].copy()
    wt_df = order_columns(wt_df)

    wt_df.to_csv(
        WT_OUTPUT_FILE,
        index=False,
    )

    print()
    print(f'Saved WT dataset: {WT_OUTPUT_FILE}')
    print(f'WT rows: {len(wt_df)}')

    return wt_df


def create_merged_smoc_dataset() -> pd.DataFrame:
    smoc_csv = load_and_normalize_csv(SMOC_CSV_FILE)
    smoc_dev = load_and_normalize_excel(SMOC_DEV_FILE)
    smoc_reg = load_and_normalize_excel(SMOC_REG_FILE)

    datasets = [
        smoc_csv,
        smoc_dev,
        smoc_reg,
    ]

    merged_df = pd.concat(
        datasets,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    raise_on_cross_file_duplicates(
        merged_df,
        dataset_name='combined Smoc datasets',
    )

    merged_df = order_columns(merged_df)

    merged_df.to_csv(
        SMOC_OUTPUT_FILE,
        index=False,
    )

    print()
    print(f'Saved merged Smoc dataset: {SMOC_OUTPUT_FILE}')
    print(f'Merged rows: {len(merged_df)}')
    print(f'{os.path.basename(SMOC_CSV_FILE)}: {len(smoc_csv)}')
    print(f'{os.path.basename(SMOC_DEV_FILE)}: {len(smoc_dev)}')
    print(f'{os.path.basename(SMOC_REG_FILE)}: {len(smoc_reg)}')

    return merged_df


def main() -> None:
    create_wt_dataset()
    create_merged_smoc_dataset()


if __name__ == '__main__':
    main()