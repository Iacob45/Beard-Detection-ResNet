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

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "beard_classifier.pt"

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
