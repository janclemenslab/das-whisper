"""das-whisper CLI entry-point."""

import defopt
from . import train as train_module
from . import predict as predict_module
from . import evaluate as evaluate_module


def main(argv=None):
    """Dispatch subcommands such as `das_whisper train ...`."""
    defopt.run(
        {
            "train": train_module.run,
            "predict": predict_module.run,
            "evaluate": evaluate_module.run,
        },
        argv=argv,
    )


if __name__ == "__main__":
    main()
