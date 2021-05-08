from functools import wraps
from typing import Callable, Optional, Tuple

import chika
import homura
import numpy as np
import torch
from homura import init_distributed, is_distributed, reporters
from homura.metrics import accuracy
from homura.modules import SmoothedCrossEntropy
from homura.vision.data import DATASET_REGISTRY
from torchvision.transforms import AutoAugment, RandomErasing

from mixer import MLPMixers


class Trainer(homura.trainers.SupervisedTrainer):

    def iteration(self,
                  data: Tuple[torch.Tensor, torch.Tensor]
                  ) -> None:
        input, target = data
        with torch.cuda.amp.autocast(self._use_amp):
            output = self.model(input)
            loss = self.loss_f(output, target)

        if self.is_train:
            self.optimizer.zero_grad(set_to_none=True)
            if self._use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if self.cfg.grad_clip > 0:
                if self._use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            if self._use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        if self._is_debug and torch.isnan(loss):
            self.logger.warning("loss is NaN")

        self.reporter.add('accuracy', accuracy(output, target))
        self.reporter.add('loss', loss.detach_())
        if self._report_topk is not None:
            for top_k in self._report_topk:
                self.reporter.add(f'accuracy@{top_k}', accuracy(output, target, top_k))


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


def fast_collate(batch: list
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    # from NVidia's Apex
    imgs = [img for img, target in batch]
    targets = torch.tensor([target for img, target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensors = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tensors[i] += torch.from_numpy(nump_array).permute(1, 2, 0).contiguous()
    return tensors, targets


def gen_mixup_collate(alpha):
    # see https://github.com/moskomule/mixup.pytorch
    beta = torch.distributions.Beta(alpha + 1, alpha)

    def f(batch):
        tensors, targets = fast_collate(batch)
        indices = torch.randperm(tensors.size(0))
        _tensors = tensors.clone()[indices]
        gamma = beta.sample()
        tensors.mul_(gamma).add_(_tensors, alpha=1 - gamma)
        return tensors, targets

    return f


@chika.config
class DataConfig:
    batch_size: int = 128
    autoaugment: bool = False
    random_erasing: bool = False
    mixup: float = chika.bounded(0, _from=0)


@chika.config
class ModelConfig:
    name: str = chika.choices(*MLPMixers.choices())
    droppath_rate: float = 0.1
    grad_clip: float = 1
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
    vs.collate_fn = fast_collate if cfg.data.mixup == 0 else gen_mixup_collate(cfg.data.mixup)
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

    with Trainer(model,
                 optimizer,
                 SmoothedCrossEntropy(cfg.optim.label_smoothing),
                 reporters=[reporters.TensorboardReporter(".")],
                 scheduler=scheduler,
                 use_amp=cfg.amp,
                 use_cuda_nonblocking=True,
                 report_accuracy_topk=5,
                 optim_cfg=cfg.optim,
                 debug=cfg.debug,
                 cfg=cfg.model
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
