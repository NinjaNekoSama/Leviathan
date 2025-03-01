import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--config', default='DFOCRP_KkHK0R2F-NML1.csv', type=str,
                    help='Path to csv file housing the metadata')
parser.add_argument('--data_dir', default='POC', type=str, metavar='PATH',
                    help='path to training data')


def main():
    args = parser.parse_args()
    data = pd.read_csv(args.config)
    print(data.columns)



if __name__ == '__main__':
    main()  