import torch
import torch.nn as nn
from attention import TransformerBlock

class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_dim=768,
        max_patches=256,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        pos_embed_type='learned',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.max_patches = max_patches
        self.pos_embed_type = pos_embed_type

        self.patch_proj = nn.Linear(patch_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)

        if pos_embed_type == 'learned':
            self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        elif pos_embed_type == 'coordinate':
            self.pos_embed_mlp = nn.Sequential(
                nn.Linear(2, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_pos_embed, std=.02)
        else:
            raise ValueError(f"Unknown position embedding type: {pos_embed_type}")

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, patch_coords=None):
        B, N, _ = x.shape
        if N > self.max_patches:
            raise ValueError(f"Got {N} patches, but max_patches={self.max_patches}.")
        
        x = self.patch_proj(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if self.pos_embed_type == 'learned':
            x = x + self.pos_embed[:, :N+1]
        elif self.pos_embed_type == 'coordinate':
            patch_pos_embed = self.pos_embed_mlp(patch_coords)
            
            x = torch.cat([
                x[:, 0:1] + self.cls_pos_embed,
                x[:, 1:] + patch_pos_embed
            ], dim=1)
        
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits