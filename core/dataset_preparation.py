import shutil
from pathlib import Path

import cv2
import dlib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CropMode, DatasetMode, RES_DIR, DATA_DIR, CLASS_MAP, CROP_MODE, DLIB_LANDMARKS_PATH

DATASETS = {
    DatasetMode.CELEBA: {
        "name": "celeba",
        "images_dir": RES_DIR / "images_celeba",
        "excel_path": RES_DIR / "database_celeba.xlsx"
    }
}

STUDENTS_MANUAL_DIR = RES_DIR / "data_students_manual"

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

SEED = 42

FACE_DETECTOR = dlib.get_frontal_face_detector()
LANDMARK_PREDICTOR = dlib.shape_predictor(str(DLIB_LANDMARKS_PATH))


def get_selected_datasets(dataset_mode: DatasetMode = DatasetMode.CELEBA) -> list[dict]:
    if dataset_mode == DatasetMode.CELEBA:
        return [DATASETS[DatasetMode.CELEBA]]

    if dataset_mode == DatasetMode.STUDENTS:
        return []

    if dataset_mode == DatasetMode.BOTH:
        return [DATASETS[DatasetMode.CELEBA]]

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

    if not selected_datasets:
        return pd.DataFrame(columns=["image_name", "class", "source", "image_path"])

    dataframes = [
        load_dataset(dataset_config)
        for dataset_config in selected_datasets
    ]

    return pd.concat(dataframes, ignore_index=True)


def load_students_manual_dataset() -> pd.DataFrame:
    rows = []
    class_to_index = {
        class_name: class_index
        for class_index, class_name in CLASS_MAP.items()
    }

    for split_name in ["train", "val", "test"]:
        split_dir = STUDENTS_MANUAL_DIR / split_name

        if not split_dir.exists():
            print(f"Missing students manual split directory: {split_dir}")
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            if class_name not in class_to_index:
                print(f"Unknown class folder in students manual dataset: {class_dir}")
                continue

            for image_path in class_dir.glob("*"):
                if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                rows.append({
                    "image_name": image_path.name,
                    "class": class_to_index[class_name],
                    "source": "students",
                    "image_path": image_path,
                    "split": split_name
                })

    return pd.DataFrame(rows)


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df[df["class"].isin(CLASS_MAP.keys())]

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df["class"],
        random_state=SEED
    )

    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df["class"],
        random_state=SEED
    )

    return train_df, val_df, test_df


def clear_data_dir() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)


def detect_largest_face(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_DETECTOR(gray_image, 1)

    if len(faces) == 0:
        return None

    return max(
        faces,
        key=lambda rect: rect.width() * rect.height()
    )


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def crop_fixed_face(image):
    image_height, image_width = image.shape[:2]

    x1 = int(image_width * 0.08)
    x2 = int(image_width * 0.92)

    y1 = int(image_height * 0.05)
    y2 = int(image_height * 0.98)

    return image[y1:y2, x1:x2]


def crop_fixed_lower_face(image):
    image_height, image_width = image.shape[:2]

    x1 = int(image_width * 0.12)
    x2 = int(image_width * 0.88)

    y1 = int(image_height * 0.42)
    y2 = int(image_height * 0.98)

    return image[y1:y2, x1:x2]


def crop_face(image, face_rect):
    image_height, image_width = image.shape[:2]

    x = face_rect.left()
    y = face_rect.top()
    w = face_rect.width()
    h = face_rect.height()

    padding_x = int(w * 0.20)
    padding_y = int(h * 0.20)

    x1 = clamp(x - padding_x, 0, image_width)
    y1 = clamp(y - padding_y, 0, image_height)
    x2 = clamp(x + w + padding_x, 0, image_width)
    y2 = clamp(y + h + padding_y, 0, image_height)

    return image[y1:y2, x1:x2]


def crop_lower_face(image, face_rect):
    image_height, image_width = image.shape[:2]

    shape = LANDMARK_PREDICTOR(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        face_rect
    )

    points = [
        (shape.part(i).x, shape.part(i).y)
        for i in range(68)
    ]

    _, nose_tip_y = points[30]

    x = face_rect.left()
    y = face_rect.top()
    w = face_rect.width()
    h = face_rect.height()

    padding_x = int(w * 0.20)
    padding_bottom = int(h * 0.15)

    y_start = nose_tip_y - int(h * 0.10)

    x1 = clamp(x - padding_x, 0, image_width)
    y1 = clamp(y_start, 0, image_height)
    x2 = clamp(x + w + padding_x, 0, image_width)
    y2 = clamp(y + h + padding_bottom, 0, image_height)

    return image[y1:y2, x1:x2]


def save_processed_image(source_path: Path, destination_path: Path, crop_mode: CropMode) -> bool:
    if crop_mode == CropMode.NONE:
        shutil.copy2(source_path, destination_path)
        return True

    image = cv2.imread(str(source_path))

    if image is None:
        print(f"Could not read image: {source_path}")
        return False

    face_rect = detect_largest_face(image)

    if face_rect is None:
        print(f"No face detected. Applying fixed crop: {source_path}")

        if crop_mode == CropMode.FACE:
            processed_image = crop_fixed_face(image)
        elif crop_mode == CropMode.LOWER_FACE:
            processed_image = crop_fixed_lower_face(image)
        else:
            processed_image = image
    else:
        if crop_mode == CropMode.FACE:
            processed_image = crop_face(image, face_rect)
        elif crop_mode == CropMode.LOWER_FACE:
            processed_image = crop_lower_face(image, face_rect)
        else:
            processed_image = image

    if processed_image.size == 0:
        print(f"Invalid crop. Copying original image: {source_path}")
        shutil.copy2(source_path, destination_path)
        return True

    cv2.imwrite(str(destination_path), processed_image)
    return True


def copy_images(df: pd.DataFrame, split_name: str, crop_mode: CropMode) -> None:
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

        save_processed_image(source_path, destination_path, crop_mode)


def get_students_split_df(students_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if students_df.empty:
        return students_df

    return students_df[students_df["split"] == split_name].copy()


def print_split_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\nDataset split completed.")
    print(f"Train:       {len(train_df)} images")
    print(f"Validation:  {len(val_df)} images")
    print(f"Test:        {len(test_df)} images")

    for split_name, df in {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df
    }.items():
        print(f"\n{split_name} class distribution:")
        print(df["class"].value_counts().sort_index().reset_index().to_string(index=False))


def dataset_preparation(dataset_mode: DatasetMode = DatasetMode.BOTH, crop_mode: CropMode = CROP_MODE) -> None:
    clear_data_dir()

    train_parts = []
    val_parts = []
    test_parts = []

    if dataset_mode in [DatasetMode.CELEBA, DatasetMode.BOTH]:
        celeba_df = load_selected_datasets(DatasetMode.CELEBA)
        celeba_train_df, celeba_val_df, celeba_test_df = split_dataframe(celeba_df)

        train_parts.append(celeba_train_df)
        val_parts.append(celeba_val_df)
        test_parts.append(celeba_test_df)

    if dataset_mode in [DatasetMode.STUDENTS, DatasetMode.BOTH]:
        students_df = load_students_manual_dataset()
        students_df = students_df[students_df["class"].isin(CLASS_MAP.keys())]

        train_parts.append(get_students_split_df(students_df, "train"))
        val_parts.append(get_students_split_df(students_df, "val"))
        test_parts.append(get_students_split_df(students_df, "test"))

    if not train_parts and not val_parts and not test_parts:
        raise ValueError(f"No data was loaded for dataset mode: {dataset_mode}")

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    copy_images(train_df, "train", crop_mode)
    copy_images(val_df, "val", crop_mode)
    copy_images(test_df, "test", crop_mode)

    print_split_stats(train_df, val_df, test_df)

    print(f"\nDataset mode: {dataset_mode.value}")
    print(f"Crop mode: {crop_mode.value}")
