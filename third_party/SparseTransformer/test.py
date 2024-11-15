import sptr
import torch
import numpy as np 
# Define module
dim = 48
num_heads = 3
indice_key = 'sptr_0'
window_size = np.array([0.4, 0.4, 0.4])  # can also be integers for voxel-based methods
shift_win = False  # whether to adopt shifted window
attn = sptr.VarLengthMultiheadSA(
    dim, 
    num_heads, 
    indice_key, 
    window_size, 
    shift_win
).cuda()

feats = torch.randn((2,48)).cuda()
indices = torch.randn((2,4)).cuda()

# Wrap the input features and indices into SparseTrTensor. Note: indices can be either intergers for voxel-based methods or floats (i.e., xyz) for point-based methods
# feats: [N, C], indices: [N, 4] with batch indices in the 0-th column
input_tensor = sptr.SparseTrTensor(feats, indices, spatial_shape=None, batch_size=None)
output_tensor = attn(input_tensor)

# Extract features from output tensor
output_feats = output_tensor.query_feats
print(output_feats)
