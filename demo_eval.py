from openbackdoor.utils.evaluator import Evaluator
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poison_file_path', type=str, default='./poison_data/sst-2/1/mix/badnet/train-poison.csv')
    parser.add_argument('--orig_file_path', type=str, default='./poison_data/sst-2/1/mix/badnet/train-clean.csv')
    parser.add_argument('--eval_type', type=str, default='ppl', choices=['ppl', 'use', 'grammar'])
    args = parser.parse_args()
    return args

def read_data(file_path):
    data = [item[1] for item in pd.read_csv(file_path).values.tolist()]
    return data


if __name__ == '__main__':
    args = parse_args()
    poison_file_path = args.poison_file_path
    orig_file_path = args.orig_file_path
    eval_type = args.eval_type

    evaluator = Evaluator()

    poison_data = read_data(poison_file_path)
    clean_data = read_data(orig_file_path)
    assert len(poison_data) == len(clean_data)

    if eval_type == 'ppl':
        print(evaluator.evaluate_ppl(clean_data, poison_data))
    elif eval_type == 'grammar':
        print(evaluator.evaluate_grammar(clean_data, poison_data))
    elif eval_type == 'use':
        print(evaluator.evaluate_use(clean_data, poison_data))

