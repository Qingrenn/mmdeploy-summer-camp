
1.Backbone
```Python
def forward(self, x):
    """Model forward function."""
    # x: (32, 3, 256, 256)
    inter_feat = self.stem(x)
    out_feats = []

    for ind in range(self.num_stacks):
        single_hourglass = self.hourglass_modules[ind] 
        out_conv = self.out_convs[ind]

        hourglass_feat = single_hourglass(inter_feat) # (32, 256, 64, 64)
        out_feat = out_conv(hourglass_feat) # (32, 256, 64, 64)
        out_feats.append(out_feat)

        if ind < self.num_stacks - 1:
            inter_feat = self.conv1x1s[ind](
                        inter_feat) + self.remap_convs[ind](out_feat)
            inter_feat = self.inters[ind](self.relu(inter_feat))

    return out_feats
```

2.Head
```python
def forward(self, x):
    """Forward function.

    Returns:
        out (list[Tensor]): a list of heatmaps from multiple stages.
    """
    import pdb; pdb.set_trace()
    out = []
    assert isinstance(x, list)
    for i in range(self.num_stages):
        y = self.multi_deconv_layers[i](x[i]) # (32, 256, 64, 64)
        y = self.multi_final_layers[i](y) # (32, 17, 64, 64)
        out.append(y)
    return out
```

3.Postprocess
调用decode方法