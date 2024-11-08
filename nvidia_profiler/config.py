import argparse
import os
import tomllib


__all__ = [
    'cliArgumentParser',
    'setupEnvironment'
]


def cliArgumentParser() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Configuration files. If not specified, defaults to /app/.config/train.toml'
    )

    args = parser.parse_args()

    # Use default if no files are provided
    config_files = args.config if args.config else [os.path.abspath('/app/.config/train.toml')]
    kwargs = {
        'config_file': config_files
    }

    return kwargs

def setupEnvironment() -> dict:
    config_params: dict = cliArgumentParser()

    config_file: str = config_params['config_file']
    with open(config_file, 'rb') as file:
        config_input = tomllib.load(file)

    config_benchmark: dict = {}
    if 'training' in config_input:
        config_training: dict = config_input['training']
        if 'num_epochs' in config_training:
            if not isinstance(config_training['num_epochs'], int):
                raise ValueError('num_epoch must be an integer')
            config_benchmark['num_epochs'] = config_training['num_epochs']
        else:
            config_benchmark['num_epochs'] = 10
        if 'batch_size' in config_training:
            if not isinstance(config_training['batch_size'], int):
                raise ValueError('batch_size must be an integer')
            config_benchmark['batch_size'] = config_training['batch_size']
        else:
            config_benchmark['batch_size'] = 8
        if 'prefetch_factor' in config_training:
            if not isinstance(config_training['prefetch_factor'], int):
                raise ValueError('prefetch_factor must be an integer')
            config_benchmark['prefetch_factor'] = config_training['prefetch_factor']
        else:
            config_benchmark['prefetch_factor'] = 2
        if 'num_workers' in config_training:
            if not isinstance(config_training['num_workers'], int):
                raise ValueError('num_workers must be an integer')
            config_benchmark['num_workers'] = config_training['num_workers']
        else:
            config_benchmark['num_workers'] = 4
        if 'device' in config_training:
            if not isinstance(config_training['device'], str):
                raise ValueError('device must be a string')
            config_benchmark['device'] = config_training['device']
    else:
        config_benchmark = {
            'num_epochs': 10,
            'batch_size': 8,
            'num_workers': 4,
            'prefetch_factor': 2
        }

    return config_benchmark
