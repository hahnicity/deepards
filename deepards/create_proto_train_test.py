import argparse
import os
import subprocess

ards_train =  ['0008RPI0120150227', '0015RPI0320150401', '0021RPI0420150513', '0026RPI1020150523', '0027RPI0620150525', '0093RPI0920151212', '0098RPI1420151218', '0099RPI0120151219', '0102RPI0120151225', '0120RPI1820160118', '0129RPI1620160126', '0147RPI1220160213', '0148RPI0120160214', '0149RPI1820160212', '0153RPI0720160217', '0194RPI0320160317', '0209RPI1920160408', '0224RPI3020160414', '0243RPI0720160512', '0245RPI1420160512', '0253RPI1220160606', '0260RPI2420160617', '0265RPI2920160622', '0266RPI1720160622', '0268RPI1220160624', '0271RPI1220160630', '0372RPI2220161211', '0381RPI2320161212', '0390RPI2220161230', '0412RPI5520170121', '0484RPI4220170630', '0506RPI3720170807', '0511RPI5220170831', '0514RPI5420170905', '0527RPI0420171028', '0546RPI5120171216', '0549RPI4420171213', '0551RPI0720180102', '0569RPI0420180116', '0640RPI2820180822']

other_train = ['0033RPI0520150603', '0108RPI0120160101', '0111RPI1520160101', '0112RPI1620160105', '0124RPI1220160123', '0125RPI1120160123', '0133RPI0920160127', '0144RPI0920160212', '0145RPI1120160212', '0157RPI0920160218', '0163RPI0720160222', '0166RPI2220160227', '0170RPI2120160301', '0173RPI1920160303', '0257RPI1220160615', '0304RPI1620160829', '0306RPI3520160830', '0317RPI3220160910', '0336RPI3920161006', '0343RPI3920161016', '0347RPI4220161016', '0354RPI5820161029', '0356RPI2220161101', '0361RPI4620161115', '0365RPI5820161125', '0387RPI3920161224', '0398RPI4220170104', '0423RPI3220170205', '0434RPI4520170224', '0460RPI2220170518', '0463RPI3220170522', '0544RPI2420171204', '0545RPI0520171214', '0552RPI2520180101', '0585RPI2720180206', '0593RPI1920180226', '0624RPI0320180708', '0624RPI1920180702', '0625RPI2820180628', '0705RPI5020190318']

ards_test = ['0127RPI0120160124', '0411RPI5820170119', '0261RPI1220160617',
       '0235RPI1320160426', '0160RPI1420160220', '0122RPI1320160120',
       '0251RPI1820160609', '0139RPI1620160205', '0357RPI3520161101',
       '0558RPI0820180104']

other_test = ['0443RPI1620170319', '0410RPI4120170118', '0380RPI3920161212',
       '0606RPI1920180416', '0135RPI1420160203', '0231RPI1220160424',
       '0137RPI1920160202', '0315RPI2720160910', '0132RPI1720160127',
       '0225RPI2520160416']


parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--dataset-path', required=True)
args = parser.parse_args()

all_data_dir = os.path.join(args.dataset_path, 'experiment1/all_data')
train_dir = os.path.join(args.dataset_path, 'experiment1/prototrain')
test_dir = os.path.join(args.dataset_path, 'experiment1/prototest')
os.mkdir(train_dir)
os.mkdir(test_dir)
all_data_raw_dir = os.path.join(all_data_dir, 'raw')
train_raw_dir = os.path.join(train_dir, 'raw')
train_meta_dir = os.path.join(train_dir, 'meta')
all_data_meta_dir = os.path.join(all_data_dir, 'meta')
test_raw_dir = os.path.join(test_dir, 'raw')
test_meta_dir = os.path.join(test_dir, 'meta')
os.mkdir(train_raw_dir)
os.mkdir(train_meta_dir)
os.mkdir(test_raw_dir)
os.mkdir(test_meta_dir)

for pt in ards_train + other_train:
    proc = subprocess.Popen(['ln', '-s', os.path.join(all_data_raw_dir, pt), train_raw_dir])
    proc.communicate()
    proc = subprocess.Popen(['ln', '-s', os.path.join(all_data_meta_dir, pt), train_meta_dir])
    proc.communicate()

for pt in ards_test + other_test:
    proc = subprocess.Popen(['ln', '-s', os.path.join(all_data_raw_dir, pt), test_raw_dir])
    proc.communicate()
    proc = subprocess.Popen(['ln', '-s', os.path.join(all_data_meta_dir, pt), test_meta_dir])
    proc.communicate()
