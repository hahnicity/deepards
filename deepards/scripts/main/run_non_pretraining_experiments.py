import argparse
import subprocess


def run_experiment(dry_run, experiment_name_prefix, cuda_arg, config_override, cuda_devices):
    experiment_name = "{}_{}".format(experiment_name_prefix, config_override.split('.yml')[0].replace('/', '_'))
    commands = [[str(i) for i in [
        'ts', 'python', 'train_ards_detector.py', '-co', config_override,
        '--no-print-progress', '-exp', experiment_name,
        '--oversample', '--clip-grad', cuda_arg,
        '--cuda-device', dev,
    ]] for dev in cuda_devices.split('+')]

    if dry_run:
        print('\nDry Runnings:\n')

    n_times_each_experiment = 10
    i = 0
    while i < n_times_each_experiment:
        to_run = commands[i % len(commands)]
        if dry_run:
            print("{}\n".format(" ".join(to_run)))
        else:
            proc = subprocess.Popen(to_run)
            proc.communicate()
        i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-e', '--experiment-name', required=True)
    parser.add_argument('--cuda-devices', help="cuda device you wish to use. Can use + to split work across different devices. E.g. 0+1", required=True)
    parser.add_argument('-co', '--config-override', required=True, help='Path to config override file you set for the experiment')
    args = parser.parse_args()

    run_experiment(args.dry_run, args.experiment_name, '--cuda-no-dp', args.config_override, args.cuda_devices)


if __name__ == "__main__":
    main()
