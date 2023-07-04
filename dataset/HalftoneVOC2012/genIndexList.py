import json
from os.path import join
from glob import glob

# Gen Special List
colors = [join('special', x) for x in glob(join('val/target_c', '*.png'))]
halfs = [x.replace('target_c', 'raw_ov') for x in colors]
valSet = {'inputs': colors, 'labels': halfs}
print(f"numbers of special test data: {len(colors)}")

colors = [join('special', x) for x in glob(join('train/target_c', '*.png'))]
halfs = [x.replace('target_c', 'raw_ov') for x in colors]
trainSet = {'inputs': colors, 'labels': halfs}
print(f"numbers of sepcial training data: {len(colors)}")

with open('../special.json', 'w') as f:
    json.dump({'train': trainSet, 'val': valSet}, f)

# Gn images list
colors = [join('HalftoneVOC2012', x) for x in glob(join('val/target_c', '*.png'))]
halfs = [x.replace('target_c', 'raw_ov') for x in colors]
valSet = {'inputs': colors, 'labels': halfs}
print(f"numbers of image test data: {len(colors)}")

colors = [join('HalftoneVOC2012', x) for x in glob(join('train/target_c', '*.png'))]
halfs = [x.replace('target_c', 'raw_ov') for x in colors]
trainSet = {'inputs': colors, 'labels': halfs}
print(f"numbers of image training data: {len(colors)}")

with open('../HalftoneVOC2012.json', 'w') as f:
    json.dump({'train': trainSet, 'val': valSet}, f)
