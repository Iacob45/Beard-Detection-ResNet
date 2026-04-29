from config import AppModes, DatasetMode
from core.train import train
from core.test import test
from core.predict import predict
from core.dataset_preparation import dataset_preparation


def main(mode: AppModes = AppModes.TEST):
    if mode == AppModes.TRAIN:
        train()
    elif mode == AppModes.TEST:
        test()
    elif mode == AppModes.PREDICT:
        predict()
    elif mode == AppModes.DATASET_PREPARATION:
        dataset_preparation(dataset_mode=DatasetMode.CELEBA)


if __name__ == "__main__":
    main(mode=AppModes.DATASET_PREPARATION)
