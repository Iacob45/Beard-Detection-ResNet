from enum import Enum
from pathlib import Path


# App config
class AppModes(str, Enum):
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"
    DATASET_PREPARATION = "dataset_preparation"


# Path config
BASE_DIR = Path(__file__).resolve().parent

RES_DIR = BASE_DIR / "resources"

DATA_DIR = BASE_DIR / "data"
CELEBA_DIR = DATA_DIR / "celeba"
STUDENTS_DIR = DATA_DIR / "students"

CELEBA_TRAIN_DIR = CELEBA_DIR / "train"
CELEBA_VAL_DIR = CELEBA_DIR / "val"
CELEBA_TEST_DIR = CELEBA_DIR / "test"

STUDENTS_TRAIN_DIR = STUDENTS_DIR / "train"
STUDENTS_VAL_DIR = STUDENTS_DIR / "val"
STUDENTS_TEST_DIR = STUDENTS_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Classes config
CLASS_MAP = {
    0: "No_Facial_Hair",
    1: "Full_Beard",
    2: "Goatee",
    3: "Mustache",
}


# Dataset config
class DatasetMode(str, Enum):
    CELEBA = "celeba"
    STUDENTS = "students"
    BOTH = "both"
