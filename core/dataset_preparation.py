import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DatasetMode, RES_DIR, DATA_DIR, CLASS_MAP

DATASETS = {
    DatasetMode.CELEBA: {
        "name": "celeba",
        "images_dir": RES_DIR / "images_celeba",
        "excel_path": RES_DIR / "database_celeba.xlsx"
    },
    DatasetMode.STUDENTS: {
        "name": "students",
        "images_dir": RES_DIR / "images_students",
        "excel_path": RES_DIR / "database_students.xlsx"
    },
}


TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

SEED = 42


def get_selected_datasets(dataset_mode: DatasetMode = DatasetMode.CELEBA) -> list[dict]:
    if dataset_mode == DatasetMode.CELEBA:
        return [DATASETS[DatasetMode.CELEBA]]

    if dataset_mode == DatasetMode.STUDENTS:
        return [DATASETS[DatasetMode.STUDENTS]]

    if dataset_mode == DatasetMode.BOTH:
        return [
            DATASETS[DatasetMode.CELEBA],
            DATASETS[DatasetMode.STUDENTS],
        ]

    raise ValueError(f"Invalid dataset mode: {dataset_mode}")


def load_dataset(dataset_config: dict) -> pd.DataFrame:
    df = pd.read_excel(dataset_config["excel_path"])

    df = df.iloc[:, :2]
    df.columns = ["image_name", "class"]

    df = df.dropna()
    df["class"] = df["class"].astype(int)
    df["source"] = dataset_config["name"]
    df["image_path"] = df["image_name"].apply(
        lambda image_name: dataset_config["images_dir"] / image_name
    )

    return df


def load_selected_datasets(dataset_mode: DatasetMode) -> pd.DataFrame:
    selected_datasets = get_selected_datasets(dataset_mode)

    dataframes = [
        load_dataset(dataset_config)
        for dataset_config in selected_datasets
    ]

    return pd.concat(dataframes, ignore_index=True)


def clear_data_dir() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)


def copy_images(df: pd.DataFrame, split_name: str) -> None:
    for _, row in df.iterrows():
        label = int(row["class"])
        class_name = CLASS_MAP[label]

        source_path = Path(row["image_path"])

        if not source_path.exists():
            print(f"Missing image: {source_path}")
            continue

        destination_dir = DATA_DIR / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        new_image_name = f"{row['source']}_{row['image_name']}"
        destination_path = destination_dir / new_image_name

        shutil.copy2(source_path, destination_path)


def print_split_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\nDataset split completed.")
    print(f"Train:       {len(train_df)} images")
    print(f"Validation:   {len(val_df)} images")
    print(f"Test:         {len(test_df)} images")

    for split_name, df in {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df,
    }.items():
        print(f"\n{split_name} class distribution:")
        print(df["class"].value_counts().sort_index().reset_index().to_string(index=False))


def dataset_preparation(dataset_mode: DatasetMode = DatasetMode.BOTH) -> None:
    df = load_selected_datasets(dataset_mode)

    df = df[df["class"].isin(CLASS_MAP.keys())]

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df["class"],
        random_state=SEED,
    )

    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df["class"],
        random_state=SEED,
    )

    clear_data_dir()

    copy_images(train_df, "train")
    copy_images(val_df, "val")
    copy_images(test_df, "test")

    print_split_stats(train_df, val_df, test_df)
