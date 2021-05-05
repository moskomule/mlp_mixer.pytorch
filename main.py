from functools import wraps
from typing import Callable, Optional

import chika
import homura
import torch
from homura import init_distributed, is_distributed, reporters
from homura.modules import SmoothedCrossEntropy
from homura.vision.data import DATASET_REGISTRY
from torchvision.transforms import AutoAugment, RandomErasing

from mixer import MLPMixers


def distributed_ready_main(func: Callable = None,
                           backend: Optional[str] = None,
                           init_method: Optional[str] = None,
                           disable_distributed_print: str = False
                           ) -> Callable:
    """ Wrap a main function to make it distributed ready
    """

    if is_distributed():
        init_distributed(backend=backend, init_method=init_method, disable_distributed_print=disable_distributed_print)

    @wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    return inner


@chika.config
class DataConfig:
    batch_size: int = 128
    autoaugment: bool = False
    random_erasing: bool = False


@chika.config
class ModelConfig:
    name: str = chika.choices(*MLPMixers.choices())
    droppath_rate: float = 0.1
    ema: bool = False
    ema_rate: float = chika.bounded(0.999, 0, 1)


@chika.config
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.1
    label_smoothing: float = 0.1
    epochs: int = 300
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    multiplier: int = 1


@chika.config
class Config:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig

    debug: bool = False
    amp: bool = False
    gpu: int = None
    no_save: bool = False

    def __post_init__(self):
        assert self.optim.lr > self.optim.min_lr


@chika.main(cfg_cls=Config, change_job_dir=True)
@distributed_ready_main
def main(cfg: Config):
    if cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
    if homura.is_master():
        import rich
        rich.print(cfg)
    vs = DATASET_REGISTRY("imagenet")
    model = MLPMixers(cfg.model.name)(num_classes=1_000, droppath_rate=cfg.model.droppath_rate)
    train_da = vs.default_train_da.copy()
    if cfg.data.autoaugment:
        train_da.append(AutoAugment())
    post_da = [RandomErasing()] if cfg.data.random_erasing else None
    train_loader, test_loader = vs(batch_size=cfg.data.batch_size,
                                   train_da=train_da,
                                   post_norm_train_da=post_da,
                                   train_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   test_size=cfg.data.batch_size * 50 if cfg.debug else None,
                                   num_workers=8)
    optimizer = homura.optim.AdamW(cfg.optim.lr, weight_decay=cfg.optim.weight_decay, multi_tensor=True)
    scheduler = homura.lr_scheduler.CosineAnnealingWithWarmup(cfg.optim.epochs, multiplier=cfg.optim.multiplier,
                                                              warmup_epochs=cfg.optim.warmup_epochs,
                                                              min_lr=cfg.optim.min_lr)

    with homura.trainers.SupervisedTrainer(model,
                                           optimizer,
                                           SmoothedCrossEntropy(cfg.optim.label_smoothing),
                                           reporters=[reporters.TensorboardReporter(".")],
                                           scheduler=scheduler,
                                           use_amp=cfg.amp,
                                           use_cuda_nonblocking=True,
                                           report_accuracy_topk=5,
                                           optim_cfg=cfg.optim,
                                           debug=cfg.debug
                                           ) as trainer:
        for ep in trainer.epoch_range(cfg.optim.epochs):
            trainer.train(train_loader)
            trainer.test(test_loader)
            trainer.scheduler.step()
            if not cfg.no_save:
                trainer.save(f"outputs/{cfg.model.name}", f"{ep}")

        print(f"Max Test Accuracy={max(trainer.reporter.history('accuracy/test')):.3f}")


if __name__ == '__main__':
    import warnings

    # to suppress annoying warnings from PIL
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
