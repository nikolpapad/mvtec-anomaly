from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main():
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category="bottle",
        train_batch_size=8,
        eval_batch_size=8,
    )
    model = Patchcore()
    engine = Engine()

    engine.fit(datamodule=datamodule, model=model)
    engine.test(datamodule=datamodule, model=model)


if __name__ == "__main__":
    main()