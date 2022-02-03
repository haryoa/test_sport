from sport_iseng.data import (
    ImageDataset,
    get_data_processer_effnet,
    LitImageDataModule,
)
from pytorch_lightning import Trainer
from sport_iseng.train import LitImageClassifier, LitImageArgs


def test_forward():
    """
    buat test forward dulu
    """
    lit_args = LitImageArgs(
        model_name="effnetb0", is_pretrained=False, learning_rate=1e-3, num_labels=2
    )
    lit_model = LitImageClassifier(lit_args=lit_args)
    dev_imgs = [f"tests/dummy_data/img_{x}.jpg" for x in range(3, 5)]
    dev_labels = ["tennis", "tennis"]
    processer = get_data_processer_effnet(read_file=True)
    im_data = ImageDataset(dev_imgs, labels=dev_labels, processer=processer)
    dl = LitImageDataModule.produce_dataloader(im_data, batch_size=2, mode="eval")
    a = next(iter(dl))
    output = lit_model(a['image'])  # return logits
    assert output.shape == (2, 2)  # expected shape bs, num_class


def test_dev_run():
    image_paths = [f"tests/dummy_data/img_{x}.jpg" for x in range(5)]
    # Callable that read data and return a tensor
    effnet_data_processer = get_data_processer_effnet(read_file=True)
    labels = ["basketball", "basketball", "basketball", "tennis", "tennis"]
    lit_datamodule = LitImageDataModule(
        train_imgs=image_paths,
        dev_imgs=image_paths,
        train_labels=labels,
        dev_labels=labels,
        processer=effnet_data_processer,
        train_batch_size=2,
        dev_batch_size=2,
    )
    lit_args = LitImageArgs(
        model_name="effnetb0", is_pretrained=False, learning_rate=1e-3, num_labels=2
    )
    lit_model = LitImageClassifier(lit_args=lit_args)
    trainer = Trainer(fast_dev_run=True, enable_checkpointing=False)
    trainer.fit(model=lit_model, datamodule=lit_datamodule)
