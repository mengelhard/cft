import json
import numpy as np
import pandas as pd
import sys, os
import argparse
import glob


def main(args):

	ufn = args.filename

	u_dict, sr_dict = count_users_and_subreddits(
		ufn, max_lines=args.maxlines, print_freq=args.vfreq)

	udf = pd.DataFrame.from_dict(u_dict,
		orient='index')
	udf.columns = ['count']

	srdf = pd.DataFrame.from_dict(sr_dict,
		orient='index')
	srdf.columns = ['count']

	udf.sort_values('count', ascending=False).to_csv(
		ufn + '_uname_counts.csv')

	srdf.sort_values('count', ascending=False).to_csv(
		ufn + '_sr_counts.csv')


def list_to_df(l, elt_name='element', count_name='count', min_count=1):
	elts, counts = np.unique(l, return_counts=True)
	df = pd.DataFrame({elt_name: elts, count_name: counts}).sort_values(
		count_name, ascending=False)
	return df[df[count_name] >= min_count]


def count_users_and_subreddits(
	fn, max_lines=0, print_freq=1000000,
	ignore=['[deleted]', 'AutoModerator']):

	uname_dict = dict()
	sr_dict = dict()

	with open(fn) as f:
		for i, line in enumerate(f):
			if (max_lines > 0) and (i > max_lines):
				break
			j = json.loads(line)
			username = j['author']
			subreddit = j['subreddit']
			if username not in ignore:
				uname_dict[username] = uname_dict.get(username, 0) + 1
				sr_dict[subreddit] = sr_dict.get(subreddit, 0) + 1
			if i % print_freq == 0:
				print('Read %i total comments from %s' % (i, fn))

	return uname_dict, sr_dict


def without(d, keys):
	new_d = d.copy()
	for key in keys:
		new_d.pop(key)
	return new_d


if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='Count usernames and subreddits from archive')
	parser.add_argument(
		'-f', '--filename', type=str, default=None,
		help='file to process')
	parser.add_argument(
		'-c', '--min_count', type=int, default=1,
		help='minimum comment count to save')
	parser.add_argument(
		'-m', '--maxlines', type=int, default=0,
		help='max lines to parse per file')
	parser.add_argument(
		'-v', '--vfreq', type=int, default=1000000,
		help='print frequency')
	args = parser.parse_args()

	main(args)
