"""das_whisper CLI entry-point."""

import defopt

from . import predict as predict_module
from . import train as train_module
from . import train_lightning as train_lightning_module


def main(argv=None):
    """Dispatch subcommands such as `das_whisper train ...`."""
    defopt.run({"train": train_module.run, "train-lightning": train_lightning_module.run, "predict": predict_module.run}, argv=argv)


if __name__ == "__main__":
    main()
