import argparse
import subprocess


def run_experiment(dry_run, experiment_name_prefix, cuda_arg, config_override, cuda_device):
    experiment_name = "{}_{}".format(experiment_name_prefix, config_override.split('.yml')[0].replace('/', '_'))
    command = [str(i) for i in [
        'ts', 'python', 'train_ards_detector.py', '-co', config_override,
        '--no-print-progress', '-exp', experiment_name,
        '--oversample', '--clip-grad', cuda_arg,
        '--cuda-device', cuda_device,
    ]]

    if dry_run:
        print("Running:\n\n{}".format(" ".join(command)))
        return

    n_times_each_experiment = 10
    for i in range(n_times_each_experiment):
        proc = subprocess.Popen(command)
        proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-e', '--experiment-name', required=True)
    parser.add_argument('--cuda-device', type=int, default=0)
    parser.add_argument('-co', '--config-override', required=True, help='Path to config override file you set for the experiment')
    args = parser.parse_args()

    run_experiment(args.dry_run, args.experiment_name, '--cuda-no-dp', args.config_override, args.cuda_device)


if __name__ == "__main__":
    main()
