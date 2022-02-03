"""
Main module of the experiment!
"""

from typing import List, Optional

import fire

from .experiment import ExperimentRunner


def run_exp(
    cfg: str,
    gpus: Optional[List[int]] = None,
    test_run: bool = False,
    ckpt_path: Optional[str] = None,
) -> None:
    """
    Run experiment directly!
    """
    exp = ExperimentRunner(cfg, test_run, gpus, ckpt_path)
    exp.run_experiment()


if __name__ == "__main__":
    fire.Fire(run_exp)
