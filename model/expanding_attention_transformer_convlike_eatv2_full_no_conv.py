import torch
from torch import nn

from einops.layers.torch import Rearrange
import spconv.pytorch as spconv

def compute_hull(inds,shape, scale, expand, feat_dim = None):
    with torch.cuda.amp.autocast(enabled=False):
        kern = torch.zeros(27,27, device="cuda")
        kern[torch.arange(27, dtype=torch.long,device="cuda"),torch.arange(27, dtype=torch.long,device="cuda")] = 1
        kern = kern.reshape(1,27,3,3,3)


        dense = torch.zeros((1,1,*shape), device="cuda") 
        dense[(0,0,*inds.unbind(dim=1))]= torch.arange(1,inds.shape[0]+1, dtype=torch.float,device="cuda")

        if scale == 1:
            feats = [torch.nn.functional.conv3d(dense,kern[:,i:i+1,:,:,:],padding=1, stride=1) for i in range(27)]
        if scale == 0.5:
            feats = [torch.nn.functional.conv3d(dense,kern[:,i:i+1,:,:,:],padding=1, stride=2) for i in range(27)]
        elif scale == 2:
            feats = [torch.nn.functional.conv_transpose3d(dense,kern[:,i:i+1,:,:,:], stride=2, padding=1, output_padding=1) for i in range(27)]
            
        feats = torch.cat(feats,dim=1).long()
       
        if expand:
            keep_inds = (feats > 0).any(dim=1,keepdim=True).nonzero()
        else:
            keep_inds = (feats[:,13:14] > 0).any(dim=1,keepdim=True).nonzero()
            
            
        hit_feats = feats[:,:,keep_inds[:,2],keep_inds[:,3],keep_inds[:,4]]
        hit_feats = hit_feats.squeeze().permute(1,0)-1

        return hit_feats, torch.cat((torch.zeros_like(keep_inds[:,2:3]), keep_inds[:,2:]), dim = 1).to(torch.int32)

class ExpandingAttention(nn.Module):
    def __init__(self,feat_dim,num_heads,dropout, shape=(256,256,32), scale = 1, expand=False,positional_queries=False, positional_factors=False, ternary_dot=False, hull_feats=False,inner_drop=0.1, spatial_embs = False):
        super().__init__()
        self.scale = scale
        self.shape = shape
        self.feat_dim = feat_dim
        self.expand = expand
        self.num_heads = num_heads
        self.positional_queries = positional_queries
        self.positional_factors = positional_factors
        self.ternary_dot = ternary_dot
        self.hull_feats = hull_feats
        self.use_spatial_embs = spatial_embs
        
        
        if self.hull_feats:
            self.pos_embed = nn.Linear(3, feat_dim)
            self.hull_norm = nn.LayerNorm(feat_dim)
        
        self.norm1 = nn.LayerNorm(feat_dim)
        self.mlp = MLP(feat_dim, 4*feat_dim, drop=inner_drop) # might as well be sparse conv block        
        self.norm2 = nn.LayerNorm(feat_dim)
        if (not self.positional_queries) or self.ternary_dot:
            self.to_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.to_k = nn.Sequential(Rearrange("l c -> 1 c l"), nn.Conv1d(feat_dim,feat_dim,1,1,0,groups=num_heads) ,Rearrange("1 c l -> l c"))
        self.to_v = nn.Sequential(Rearrange("l c -> 1 c l"), nn.Conv1d(feat_dim,feat_dim,1,1,0,groups=num_heads) ,Rearrange("1 c l -> l c"))
        
        if positional_queries:
            self.spatial_heads = torch.nn.parameter.Parameter(torch.randn(27 * num_heads,feat_dim //num_heads ))
        if positional_factors:
            self.spatial_factors = torch.nn.parameter.Parameter(torch.randn(27 * num_heads))
        if self.use_spatial_embs:
            self.spatial_embs = torch.nn.parameter.Parameter(torch.randn(27 * num_heads,feat_dim //num_heads,feat_dim //num_heads))
            self.spatial_bias = torch.nn.parameter.Parameter(torch.randn(27 * num_heads,feat_dim //num_heads))
        
    def forward(self, coords,feats):
        feat_dim = self.feat_dim
        device = feats.device
        dtype = feats.dtype
        
        shape = self.shape
        scale = self.scale

        relevant_points = coords
        relevant_attendee_feats = feats

        if self.expand:
            hit_feats, new_inds = compute_hull(relevant_points.long(),shape,scale, self.expand,feat_dim )
        else:
            hit_feats, new_inds = compute_hull(relevant_points.long(),shape,scale, self.expand,feat_dim )

        vals,sort_inds = (hit_feats >= 0).sum(dim=1).sort()

        hit_indices_grouped_by_attendee_count, end_indices = vals.unique(return_counts=True)


        split_indices = end_indices.cumsum(dim=0)[:-1]

        hit_feats = hit_feats[sort_inds]
        new_inds = new_inds[sort_inds]
        shortcut_indices = hit_feats[:,13]      
        
        
        
        if self.positional_queries and not self.ternary_dot:
            k,v = self.to_k(relevant_attendee_feats),self.to_v(relevant_attendee_feats)
        else:
            q,k,v = self.to_q(relevant_attendee_feats),self.to_k(relevant_attendee_feats),self.to_v(relevant_attendee_feats)

        
        if self.hull_feats:
            to_emb = new_inds[shortcut_indices==-1,1:]
            to_emb = to_emb.to(q.dtype)
            hull_feats = self.pos_embed(to_emb)
            hull_feats = self.hull_norm(hull_feats)
            shortcut_indices[shortcut_indices == -1] = -(torch.arange((shortcut_indices == -1).sum(),device=device) + relevant_attendee_feats.shape[0])
            
            q = torch.cat((q,hull_feats))


        grouped_shortcut_indices = shortcut_indices.tensor_split(split_indices.cpu())
        grouped_hit_feats = hit_feats.tensor_split(split_indices.cpu())

        grouped_hit_positions = list(map(lambda x: (x >=0).nonzero(as_tuple=True), grouped_hit_feats))
        grouped_hit_feats = list(map(lambda x: x[(x >=0).nonzero(as_tuple=True)], grouped_hit_feats))

        # !remaining potential centers <=> next feature map tokens
        outs = []
        for count_per_window, indices, position_tuples, _shortcut_indices in list(zip(hit_indices_grouped_by_attendee_count,grouped_hit_feats, grouped_hit_positions, grouped_shortcut_indices)):
            embedded_attendees = k[indices]
            embedded_attendees2 = v[indices]
            
            #!
            from einops import rearrange
            windows_to_process = rearrange(embedded_attendees, "(b seq) c -> b seq c",seq=count_per_window)
            windows_to_process2 = rearrange(embedded_attendees2, "(b seq) c -> b seq c",seq=count_per_window)

            windows_to_process = self.norm1(windows_to_process)
            windows_to_process2 = self.norm1(windows_to_process2)
            
            
            position_indices = position_tuples[1].reshape(-1,1,count_per_window).repeat_interleave(self.num_heads, dim=0)
            indices_for_select = (position_indices + (torch.arange(self.num_heads).cuda() *27).repeat(position_indices.shape[0]//self.num_heads).reshape(-1,1,1)).flatten()
            if self.positional_queries or self.ternary_dot:
                relevant_spatial_heads = self.spatial_heads.index_select(0, indices_for_select).reshape(-1,count_per_window,self.feat_dim//self.num_heads)
            if self.positional_factors:
                relevant_spatial_factors = self.spatial_factors.index_select(0, indices_for_select).reshape(-1,1, count_per_window)
                
            
            if self.hull_feats:     
                _shortcut_indices = _shortcut_indices.abs()               
            
            if self.ternary_dot:
                if self.hull_feats:                    
                    queries_to_process = relevant_spatial_heads, rearrange(q.index_select(0,_shortcut_indices), "b (heads rest) -> (b heads) 1 rest",heads=self.num_heads)
                else:
                    q = torch.cat((torch.zeros_like(q[0:1,:]),q))
                    queries_to_process = relevant_spatial_heads, rearrange(q.index_select(0,_shortcut_indices + 1), "b (heads rest) -> (b heads) 1 rest",heads=self.num_heads)
            elif self.positional_queries:
                queries_to_process = relevant_spatial_heads
            else:
                if self.hull_feats:                    
                    queries_to_process = rearrange(q.index_select(0,_shortcut_indices), "b (heads rest) -> (b heads) 1 rest",heads=self.num_heads)
                else:
                    q = torch.cat((torch.zeros_like(q[0:1,:]),q))
                    queries_to_process = rearrange(q.index_select(0,_shortcut_indices + 1), "b (heads rest) -> (b heads) 1 rest",heads=self.num_heads)
                
            windows_to_process = rearrange(windows_to_process, "b seq (heads rest) -> (b heads) seq rest",heads=self.num_heads)
            windows_to_process2 = rearrange(windows_to_process2, "b seq (heads rest) -> (b heads) seq rest",heads=self.num_heads)
        
            if self.use_spatial_embs:
                relevant_spatial_embs = self.spatial_embs.index_select(0, indices_for_select).reshape(-1,count_per_window,self.feat_dim//self.num_heads,self.feat_dim//self.num_heads)
                relevant_spatial_bias = self.spatial_bias.index_select(0, indices_for_select).reshape(-1,count_per_window,self.feat_dim//self.num_heads)
                
                windows_to_process = torch.einsum("bij,bikj -> bik",windows_to_process, relevant_spatial_embs) + relevant_spatial_bias

                
            if self.ternary_dot:
                attn_weights = torch.einsum("bkj,bij,bkj -> bik",*queries_to_process, windows_to_process)
            elif self.positional_queries:
                attn_weights = torch.einsum("bkj,bkj -> bk",queries_to_process, windows_to_process).unsqueeze(1)
            else:
                attn_weights = torch.einsum("bij,bkj -> bik",queries_to_process, windows_to_process)
            attn_weights = (attn_weights * (int(feat_dim) ** (-0.5)))
            attn_weights = attn_weights.softmax(dim=2)
            if self.positional_factors:
                attn_weights = attn_weights * relevant_spatial_factors
        
            out = torch.einsum("bji, bkj -> bik",windows_to_process2,attn_weights).squeeze(2)
            out = rearrange(out, "(b heads) rest -> b (heads rest)",heads=self.num_heads)
            outs.append(out)
            
        
        out = torch.cat(outs,dim=0).to(dtype=relevant_attendee_feats.dtype)
        mask = torch.logical_and(shortcut_indices>=0, shortcut_indices< relevant_attendee_feats.shape[0])
        out[mask] = out[mask] + relevant_attendee_feats[shortcut_indices[mask]]

        if self.hull_feats or self.ternary_dot:
            mask2 = shortcut_indices>= relevant_attendee_feats.shape[0]
            out[mask2] = out[mask2] + q[shortcut_indices[mask2]].to(out.dtype)
            
        out = self.mlp(self.norm2(out))+ out

        return out, new_inds
        
class Pruning(nn.Module):
    def __init__(self,dim_in = 32, dim_out = 20):
        super().__init__()

        self.mod = nn.Linear(dim_in, dim_out)
            
    def forward(self, x):
        prediction = self.mod(x)
        mask = prediction.max(1)[1]
        mask = mask != 0
        
        return mask, prediction

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, act = nn.GELU, drop = 0.1):
        super().__init__()
        self.dim = dim
        self.hiddem_dim = hidden_dim
        self.act = act()
        self.drop = drop
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop,inplace=True)
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.drop(x)
        return x
     
class ExpandingTransformer(nn.Module):
    def __init__(self, dim_in,dim, dim_out, num_heads = 8,dropout=0.1,scale=1.0,expand=True,prune=True,shape=(256,256,32),positional_queries=False,positional_factors=False,ternary_dot=False, hull_feats=False, inner_drop=0.1,spatial_embs=False, *args, **kwargs):
        super().__init__()
        self.prune = prune
        self.scale = scale
        self.num_heads = num_heads
        self.attn = ExpandingAttention(dim,num_heads,dropout, shape=shape,positional_queries=positional_queries, scale=scale,expand=expand, positional_factors=positional_factors, ternary_dot=ternary_dot, hull_feats=hull_feats, inner_drop=inner_drop, spatial_embs=spatial_embs)

        
        self.pruning = Pruning(dim) if prune else None
        if kwargs.get("use_out_proj", False):
            self.out_projection = nn.Linear(dim, dim_out)
        else:
            self.out_projection = None
        
    def forward(self, feats, coords, batch, spatial_shape):
        self.attn.shape = spatial_shape
        coords = torch.cat((batch.reshape(-1,1), coords), dim = 1)

        feats, coords = self.attn(coords[:,1:], feats)
        
        if self.prune:
            mask, prediction = self.pruning(feats)
            prediction_dense = spconv.SparseConvTensor(prediction,coords,spatial_shape=(256,256,32),batch_size=1).dense()
            feats = feats[mask]
            coords = coords[mask]
        else:
            prediction_dense = None
        if self.out_projection is not None:
            feats = self.out_projection(feats)
        return feats, coords, prediction_dense
