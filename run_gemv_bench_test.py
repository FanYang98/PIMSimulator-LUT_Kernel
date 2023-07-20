import argparse
import os

parser = argparse.ArgumentParser(description='A script to run PIMsimulator bench tests.')
parser.add_argument('--num_codebooks', type=str, required=True, help='The number of codebooks.')
parser.add_argument('--num_tokens', type=str, required=True, help='The number of tokens.')
parser.add_argument('--num_centroids', type=str, required=True, help='The number of centroids.')
parser.add_argument('--feature_lens', type=str, required=True, help='The length of feature.')

args = parser.parse_args()

args = args.num_codebooks + " " + args.num_tokens + " " + args.num_centroids + " " + args.feature_lens
cmd = "echo '"+args+"' | ./sim --gtest_filter=PIMBenchFixture.gemv"
print(cmd)

os.system(cmd)
    
