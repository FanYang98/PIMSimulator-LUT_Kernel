import numpy as np
import argparse

parser = argparse.ArgumentParser(description='A script to generate data file to drive PIMsimulator.')
parser.add_argument('--num_codebooks', type=int, required=True, help='The number of codebooks.')
parser.add_argument('--num_tokens', type=int, required=True, help='The number of tokens.')
parser.add_argument('--num_centroids', type=int, required=True, help='The number of centroids.')
parser.add_argument('--feature_lens', type=int, required=True, help='The length of feature.')

args = parser.parse_args()

print("num_codebooks:", args.num_codebooks)
print("num_tokens:", args.num_tokens)
print("num_centroids:", args.num_centroids)
print("feature_lens:", args.feature_lens)

nc = args.num_codebooks
n = args.num_tokens 
num_centroids = args.num_centroids
out_features = args.feature_lens

BATCH = n
REAL_DIM_IN = nc*num_centroids
DIM_IN = nc*num_centroids
DIM_OUT = out_features

batch_in = np.zeros((DIM_IN, BATCH)).astype('float16')
data_w = np.zeros((DIM_OUT, DIM_IN)).astype('float16')
batch_out = np.zeros((DIM_OUT, BATCH)).astype('float16')
batch_out2 = np.zeros((DIM_OUT, BATCH)).astype('float16')

batch_in = batch_in.T.copy()
batch_out = batch_out.T.copy()
batch_out2 = batch_out2.T.copy()

np.save("./data/gemv/gemv_input_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_in)
np.save("./data/gemv/gemv_weight_" + str(DIM_OUT) + "x" + str(DIM_IN), data_w)
np.save("./data/gemv/gemv_output_" + str(DIM_OUT) + "x" + str(DIM_IN), batch_out)

print(batch_in)
print(batch_out)
print(batch_out2)
print(batch_in.shape)
print(batch_out.shape)
