import argparse
import normalize_results
import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./celeba/draw')

args, unknown = parser.parse_known_args()

args = vars(args)

normalize_results.normalize_data(args['dir'], 5, 'Attack on VAE')
m = metrics.calc_from_normalized(args['dir'])

print(m)
