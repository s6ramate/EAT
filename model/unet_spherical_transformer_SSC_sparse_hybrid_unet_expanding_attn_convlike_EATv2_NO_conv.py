import functools
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
from spconv.core import ConvAlgo
from collections import OrderedDict
import torch.nn.functional
from torch_scatter import scatter_mean
from model.spherical_transformer import SphereFormer
from model.expanding_attention_transformer_convlike_eatv2_full_no_conv import ExpandingTransformer as ExpandingTransformer


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size
        )
        output = self.conv_branch(input)
        output = output.replace_feature(
            output.features + self.i_branch(identity).features
        )
        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()
        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=indice_key,
            ),
        )

    def forward(self, input):
        return self.conv_layers(input)


def get_downsample_info(xyz, batch, indice_pairs):
    pair_in, pair_out = indice_pairs[0], indice_pairs[1]
    valid_mask = pair_in != -1
    valid_pair_in, valid_pair_out = (
        pair_in[valid_mask].long(),
        pair_out[valid_mask].long(),
    )
    xyz_next = scatter_mean(xyz[valid_pair_in], index=valid_pair_out, dim=0)
    batch_next = scatter_mean(batch.float()[valid_pair_in], index=valid_pair_out, dim=0)
    return xyz_next, batch_next

class SparseResidual(spconv.SparseModule):
    def __init__(self,*inner):
        super(SparseResidual, self).__init__()
        self.inner = spconv.SparseSequential(*inner)
        
    def forward(self, x):
        residual = x
        x = self.inner(x)
        x = spconv.functional.sparse_add(x, residual)
        return x
    
class ExpandingTransformerSpconv(spconv.SparseModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mod = ExpandingTransformer(*args, **kwargs)
        # self.mod2 = ExpandingTransformer2(*args, **kwargs)
        
    def forward(self, spconv_tensor:spconv.SparseConvTensor):
        coords = spconv_tensor.indices
        xyz = coords[:,1:]
        batch = coords[:, 0]
        feats = spconv_tensor.features
        feats, coords, pred = self.mod(feats,xyz,batch, spconv_tensor.spatial_shape)
        return spconv.SparseConvTensor(feats,coords,[int(self.mod.scale * dim) for dim in spconv_tensor.spatial_shape], spconv_tensor.batch_size)
    
    
class SparseBlock(spconv.SparseModule):
    def __init__(self,dim_in, dim_out, dim_down=None, dim_up=None, scale = 1,scale_with_conv = False,subm=nn.Identity(),drop_path=0.0,outer_drop=0.0,**kwargs):
        super(SparseBlock, self).__init__(subm)
        dim_down = dim_out if dim_down is None else dim_down
        dim_up = dim_out if dim_up is None else dim_up

        from math import ceil
        self.mod = spconv.SparseSequential(
            SparseResidual(
            ExpandingTransformerSpconv(dim_in,dim_in,dim_down, expand=True,scale=0.5, prune=False, **kwargs) if not scale_with_conv else spconv.SparseConv3d(dim_in, dim_down, 2*scale + 1, scale, scale, bias = False),
            nn.LayerNorm(dim_down, elementwise_affine=False),
            nn.Dropout(outer_drop),
            nn.ReLU(),
            subm or torch.nn.Linear(dim_down,dim_up),
            ExpandingTransformerSpconv(dim_up,dim_up,dim_out,drop_path=drop_path,shift = True, expand=True, scale=2.0, prune=False, **kwargs) if not scale_with_conv else nn.Identity(),
            nn.LayerNorm(dim_up if scale_with_conv else dim_out, elementwise_affine=False),
            nn.ReLU(),
            spconv.SparseConvTranspose3d(dim_up, dim_out, 2*scale, scale, scale//2, bias = False) if scale_with_conv else nn.Identity(),
            ),
            nn.Dropout(outer_drop),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.mod(x)
        return x

class testModel(nn.Module):
    def __init__(self,**kwargs):
        super(testModel, self).__init__()
        
        block = SparseBlock(256,256,512,512,scale=2, drop_path=0*(0.3/4),**kwargs)
        block = SparseBlock(128,128,256,256,scale=2,subm=block, drop_path=1*(0.3/4),**kwargs)
        block = SparseBlock(64,64,128,128,scale=2,subm=block,drop_path=2*(0.3/4),**kwargs)
        block = SparseBlock(51,51,64,64,scale=2,subm=block,scale_with_conv=True, drop_path=3*(0.3/4),**kwargs)
        self.model = block
        
    def forward(self, x):
        return self.model(x)
    
    
class UBlock(nn.Module):
    def __init__(
        self,
        nPlanes,
        norm_fn,
        block_reps,
        block,
        window_size,
        window_size_sphere,
        quant_size,
        quant_size_sphere,
        head_dim=16,
        window_size_scale=[2.0, 2.0],
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path=0.0,
        indice_key_id=1,
        grad_checkpoint_layers=[],
        sphere_layers=[1, 2, 3, 4, 5],
        a=0.05 * 0.25,
    ):
        super().__init__()

        self.nPlanes = nPlanes
        self.indice_key_id = indice_key_id
        self.grad_checkpoint_layers = grad_checkpoint_layers
        self.sphere_layers = sphere_layers

        blocks = {
            "block{}".format(i): block(
                nPlanes[0],
                nPlanes[0],
                norm_fn,
                indice_key="subm{}".format(indice_key_id),
            )
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if indice_key_id in sphere_layers:
            self.window_size = window_size
            self.window_size_sphere = window_size_sphere
            num_heads = nPlanes[0] // head_dim
            self.transformer_block = SphereFormer(
                nPlanes[0],
                num_heads,
                window_size,
                window_size_sphere,
                quant_size,
                quant_size_sphere,
                indice_key="sphereformer{}".format(indice_key_id),
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[0],
                a=a,
            )

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(
                    nPlanes[0],
                    nPlanes[1],
                    kernel_size=2,
                    stride=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                    algo=ConvAlgo.Native,
                ),
            )

            window_size_scale_cubic, window_size_scale_sphere = window_size_scale
            window_size_next = np.array(
                [
                    window_size[0] * window_size_scale_cubic,
                    window_size[1] * window_size_scale_cubic,
                    window_size[2] * window_size_scale_cubic,
                ]
            )
            quant_size_next = np.array(
                [
                    quant_size[0] * window_size_scale_cubic,
                    quant_size[1] * window_size_scale_cubic,
                    quant_size[2] * window_size_scale_cubic,
                ]
            )
            window_size_sphere_next = np.array(
                [
                    window_size_sphere[0] * window_size_scale_sphere,
                    window_size_sphere[1] * window_size_scale_sphere,
                    window_size_sphere[2],
                ]
            )
            quant_size_sphere_next = np.array(
                [
                    quant_size_sphere[0] * window_size_scale_sphere,
                    quant_size_sphere[1] * window_size_scale_sphere,
                    quant_size_sphere[2],
                ]
            )
            self.u = UBlock(
                nPlanes[1:],
                norm_fn,
                block_reps,
                block,
                window_size_next,
                window_size_sphere_next,
                quant_size_next,
                quant_size_sphere_next,
                window_size_scale=window_size_scale,
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                drop_path=drop_path[1:],
                indice_key_id=indice_key_id + 1,
                grad_checkpoint_layers=grad_checkpoint_layers,
                sphere_layers=sphere_layers,
                a=a,
            )

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(
                    nPlanes[1],
                    nPlanes[0],
                    kernel_size=2,
                    bias=False,
                    indice_key="spconv{}".format(indice_key_id),
                    algo=ConvAlgo.Native,
                ),
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail["block{}".format(i)] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key="subm{}".format(indice_key_id),
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, inp, xyz, batch):
        assert (inp.indices[:, 0] == batch).all()

        output = self.blocks(inp)

        # transformer
        if self.indice_key_id in self.sphere_layers:
            if self.indice_key_id in self.grad_checkpoint_layers:

                def run(feats_, xyz_, batch_):
                    return self.transformer_block(feats_, xyz_, batch_)

                transformer_features = torch.utils.checkpoint.checkpoint(
                    run, output.features, xyz, batch
                )
            else:
                transformer_features = self.transformer_block(
                    output.features, xyz, batch
                )
            output = output.replace_feature(transformer_features)

        identity = spconv.SparseConvTensor(
            output.features, output.indices, output.spatial_shape, output.batch_size
        )

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)

            # downsample
            indice_pairs = output_decoder.indice_dict[
                "spconv{}".format(self.indice_key_id)
            ].indice_pairs
            xyz_next, batch_next = get_downsample_info(xyz, batch, indice_pairs)

            output_decoder = self.u(output_decoder, xyz_next, batch_next.long())
            output_decoder = self.deconv(output_decoder)
            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1)
            )
            output = self.blocks_tail(output)

        return output    

class Semantic(nn.Module):
    def __init__(
        self,
        input_c,
        m,
        classes,
        block_reps,
        block_residual,
        layers,
        window_size,
        window_size_sphere,
        quant_size,
        quant_size_sphere,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path_rate=0.0,
        window_size_scale=2.0,
        grad_checkpoint_layers=[],
        sphere_layers=[1, 2, 3, 4, 5],
        a=0.05 * 0.25,
    ):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1"
            )
        )

        self.unet = UBlock(
            layers,
            norm_fn,
            block_reps,
            block,
            window_size,
            window_size_sphere,
            quant_size,
            quant_size_sphere,
            window_size_scale=window_size_scale,
            rel_query=rel_query,
            rel_key=rel_key,
            rel_value=rel_value,
            drop_path=dpr,
            indice_key_id=1,
            grad_checkpoint_layers=grad_checkpoint_layers,
            sphere_layers=sphere_layers,
            a=a,
        )
        
        self.output_layer = spconv.SparseSequential(norm_fn(m), nn.ReLU())

        #### semantic segmentation
        self.linear = nn.Linear(m, classes)  # bias(default): True

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, xyz, batch):
        """
        :param input_map: (N), int, cuda
        """

        output = self.input_conv(input)
        output = self.unet(output, xyz, batch)
        output = self.output_layer(output)

        #### semantic segmentation
        semantic_scores = self.linear(output.features)  # (N, nClass), float
        return semantic_scores


class Semantic_for_SSC(nn.Module):
    def __init__(
        self,
        input_c,
        m,
        classes,
        block_reps,
        block_residual,
        layers,
        window_size,
        window_size_sphere,
        quant_size,
        quant_size_sphere,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path_rate=0.0,
        window_size_scale=2.0,
        grad_checkpoint_layers=[],
        sphere_layers=[1, 2, 3, 4, 5],
        a=0.05 * 0.25,
    ):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 7)]

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_c, m, kernel_size=3, padding=1, bias=False, indice_key="subm1"
            )
        )

        self.unet = UBlock(
            layers,
            norm_fn,
            block_reps,
            block,
            window_size,
            window_size_sphere,
            quant_size,
            quant_size_sphere,
            window_size_scale=window_size_scale,
            rel_query=rel_query,
            rel_key=rel_key,
            rel_value=rel_value,
            drop_path=dpr,
            indice_key_id=1,
            grad_checkpoint_layers=grad_checkpoint_layers,
            sphere_layers=sphere_layers,
            a=a,
        )
    

        self.output_layer = spconv.SparseSequential(norm_fn(m), nn.ReLU())

        #### semantic segmentation
        self.linear = nn.Linear(m, classes)  # bias(default): True

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, xyz, batch):
        """
        :param input_map: (N), int, cuda
        """

        output = self.input_conv(input)
        output = self.unet(output, xyz, batch)
        output = self.output_layer(output)

        # ### semantic segmentation
        # semantic_scores = self.linear(output.features)   # (N, nClass), float
        # return semantic_scores

        semantic_features = self.linear(output.features)  # (N, nClass), float
        output = output.replace_feature(
            torch.cat((semantic_features, output.features), 1)
        )
        return output
  
class SSC(nn.Module):
    def __init__(
        self,
        segmentor,
        input_c,
        m,
        classes,
        block_reps,
        block_residual,
        layers,
        window_size,
        window_size_sphere,
        quant_size,
        quant_size_sphere,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        drop_path_rate=0.0,
        window_size_scale=2.0,
        grad_checkpoint_layers=[],
        sphere_layers=[1, 2, 3, 4, 5],
        a=0.05 * 0.25,
        **kwargs
    ):
        super().__init__()

        self.segmentor = segmentor
        
        self.model = testModel(**kwargs)
        
        self.linout = nn.Linear(51,20)

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, input, xyz, batch, image):
        """
        :param input_map: (N), int, cuda
        """

        output = self.segmentor(input, xyz, batch)
        feats = output.features
        coords = output.indices

        coords = torch.cat((coords[:,0:1],torch.floor((coords[:,1:]) * 0.05 / 0.2)), dim = 1)
        
        xyz0 = xyz[:,[1,0,2]]
        mask = torch.all(
            (xyz0 > torch.tensor([-25.6, 0, -2.0]).reshape(1,3).cuda()) & (xyz0 < torch.tensor([25.6, 51.2, 4.4]).reshape(1,3).cuda()),axis=1)

        xyz=xyz[mask, :]
        feats=feats[mask, :]
        
        coords = torch.floor(xyz / 0.2)

        batch = batch[mask]
        coords = torch.cat((batch.reshape(-1,1), coords), dim=1).to(torch.int32)
        
        coords[:,2] = coords[:,2] + 128
        coords[:,3] = coords[:,3] + 10
        x = spconv.SparseConvTensor(feats,coords.to(torch.int32),(256,256,32),batch_size=1)
        
        output = self.model(x)
        
        output = output.replace_feature(self.linout(output.features))
        output = output.dense().permute(0,1,2,4,3)


        return {
            "seg": None,
            "1_1": output,
        }
