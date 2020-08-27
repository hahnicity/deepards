import argparse
import os
import subprocess


def run_experiment(dry_run, cuda_arg, config_override, cuda_devices, n_times_each_experiment):
    experiment_name = config_override.split('.yml')[0].replace('/', '_')
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train_ards_detector.py'))
    commands = [[str(i) for i in [
        'ts', 'python', script_path, '-co', config_override,
        '--no-print-progress', '-exp', experiment_name,
        '--clip-grad', cuda_arg,
        '--cuda-device', dev,
    ]] for dev in cuda_devices.split('+')]

    if dry_run:
        print('\nDry Runnings:\n')

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
    parser.add_argument('--cuda-devices', help="cuda device you wish to use. Can use + to split work across different devices. E.g. 0+1", required=True)
    parser.add_argument('-co', '--config-override', required=True, help='Path to config override file you set for the experiment')
    parser.add_argument('--n-runs', type=int, help='Times to run each experiment', default=10)
    args = parser.parse_args()

    run_experiment(args.dry_run, '--cuda-no-dp', args.config_override, args.cuda_devices, args.n_runs)


if __name__ == "__main__":
    main()
