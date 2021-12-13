# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py.

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python imagenet.py --data-path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python imagenet.py --help
"""
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pl_examples.domain_templates.imagenet import ImageNetLightningModel
from pytorch_lightning.utilities.cli import LightningCLI


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    print(args)

    assert args.default_root_dir is not None, "Please set the path to save the quantized model"

    if args.accelerator == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = ImageNetLightningModel(**vars(args))

    if args.evaluate:
        # load and validate quantized model
        import neural_compressor
        model.model = neural_compressor.utils.pytorch.load(args.default_root_dir, model.model)
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.test(model)
        return

    quantizer_cb = pl.callbacks.INCQuantization('config/quantization.yaml',
                                                monitor="val_acc1",
                                                dirpath=args.default_root_dir)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[quantizer_cb])

    trainer.compressor(model)


def run_cli():
    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true",
                                help="evaluate model on validation set")

    cli = MyLightningCLI(ImageNetLightningModel,
                         seed_everything_default=42,
                         save_config_overwrite=True, run=False)
    if cli.config["evaluate"]:
        from neural_compressor.utils.pytorch import load
        print("Load quantized configure from ", cli.trainer.default_root_dir)
        cli.model.model = load(cli.trainer.default_root_dir, cli.model.model)
        out = cli.trainer.validate(cli.model, datamodule=cli.datamodule)
        print("val_acc1:{}".format(out[0]["val_acc1"]))
    else:
        callback = pl.callbacks.INCQuantization('config/quantization.yaml',
                                                monitor="val_acc1",
                                                dirpath=cli.trainer.default_root_dir)
        cli.trainer.callbacks.append(callback)
        cli.trainer.compress(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    run_cli()
