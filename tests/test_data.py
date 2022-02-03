from sport_iseng.data import (
    ImageDataset,
    get_data_processer_effnet,
    read_image,
    LitImageDataModule,
)


def test_image_read():
    img = read_image("tests/dummy_data/img_0.jpg")
    assert img.shape[2] == 3


def test_data_diff():
    # 3
    image_paths = [f"tests/dummy_data/img_{x}.jpg" for x in range(5)]
    labels = ["basketball", "basketball", "basketball", "tennis", "tennis"]
    effnet_data_processer = get_data_processer_effnet(read_file=True)

    image_dataset = ImageDataset(
        image_paths=image_paths,
        labels=labels,
        processer=effnet_data_processer,
        itos_label_set=['tennis', 'basketball']
    )
    assert image_dataset[3]['label'] == 0


def test_data_read():
    # 1
    image_paths = [f"tests/dummy_data/img_{x}.jpg" for x in range(5)]
    # Callable that read data and return a tensor
    effnet_data_processer = get_data_processer_effnet(read_file=True)
    labels = ["basketball", "basketball", "basketball", "tennis", "tennis"]
    image_dataset = ImageDataset(
        image_paths=image_paths, labels=labels, processer=effnet_data_processer
    )
    assert image_dataset[2]["label"] == 0
    assert len(image_dataset[2]["image"].shape) == 3  # dim: should be 3
    assert image_dataset[2]["image"].shape[0] == 3  # R G B dimension


def test_dl():
    # 2
    # we use Pytorch Lightning DataModule
    train_imgs = [f"tests/dummy_data/img_{x}.jpg" for x in range(3)]
    dev_imgs = [f"tests/dummy_data/img_{x}.jpg" for x in range(3, 5)]

    train_labels = ["basketball", "basketball", "basketball"]
    dev_labels = ["tennis", "tennis"]
    effnet_data_processer = get_data_processer_effnet(read_file=True)
    processer = effnet_data_processer

    gambar = LitImageDataModule(
        train_imgs=train_imgs,
        dev_imgs=dev_imgs,
        train_batch_size=2,
        train_labels=train_labels,
        dev_labels=dev_labels,
        processer=processer,
    )

    gambar.setup()

    sampled = next(iter(gambar.train_dataloader()))
    assert sampled["image"].shape == (2, 3, 224, 224)

    sampled = next(iter(gambar.val_dataloader()))
    assert sampled["image"].shape == (2, 3, 224, 224)

    dl_coba = gambar.produce_dataloader(
        dataset=gambar.train_dataset, batch_size=2, mode="eval"
    )
    sampled = next(iter(dl_coba))
    assert sampled["image"].shape == (2, 3, 224, 224)

    assert gambar.get_num_labels() == 1
