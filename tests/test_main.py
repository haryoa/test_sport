def test_main():
    from sport_iseng.__main__ import run_exp

    yaml_file = "tests/other_util/test_exp.yaml"
    run_exp(cfg=yaml_file, gpus=None, test_run=True)


def test_prepare_data_img():
    from sport_iseng.util import get_image_paths_with_labels

    data_ready = get_image_paths_with_labels(
        csv_file="tests/other_util/data_train.csv",
        base_dir="tests",
        extracted_dataset=["train", "valid", "test"],
    )

    # expected return Dict that contains
    # those key with the contents of
    # image_paths and label Dict[str, str]

    assert "tests\\dummy_data/img_0.jpg" in data_ready["train"]["image_paths"]
    assert len(data_ready["valid"]["image_paths"]) == 3
