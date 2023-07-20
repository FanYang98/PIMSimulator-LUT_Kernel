import argparse
import os

parser = argparse.ArgumentParser(description='A script to run PIMsimulator kernel test.')
parser.add_argument('--num_codebooks', type=str, required=True, help='The number of codebooks.')
parser.add_argument('--num_tokens', type=str, required=True, help='The number of tokens.')
parser.add_argument('--num_centroids', type=str, required=True, help='The number of centroids.')
parser.add_argument('--feature_lens', type=str, required=True, help='The length of feature.')

args = parser.parse_args()

cmd = "python3 ./data/gemv/auto_gen_gemv_whole_fast.py "+ \
    " --num_codebooks " + args.num_codebooks + \
    " --num_tokens " + args.num_tokens + \
    " --num_centroids " + args.num_centroids + \
    " --feature_lens " + args.feature_lens
os.system(cmd)

args = args.num_codebooks + " " + args.num_tokens + " " + args.num_centroids + " " + args.feature_lens
cmd = "echo '"+args+"' | ./sim --gtest_filter=PIMKernelFixture.gemv"
print(cmd)
os.system(cmd)
