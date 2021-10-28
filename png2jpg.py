# -*- coding: utf-8 -*-

import os
from PIL import Image

save_path = 'qualitative_evaluation/'
for root, dirs, files in os.walk('r=140, eps=1e-4/'):
    print('root:', root)
    if not os.path.exists(os.path.join(save_path, root)):
        os.makedirs(os.path.join(save_path, root))
    
    for dir in dirs:
        print('dir:', dir)
        if not os.path.exists(os.path.join(save_path, root, dir)):
            os.makedirs(os.path.join(save_path, root, dir))
    
    for name in files:
        if '.png' in name:
            print(os.path.join(root, name))
            Image.open(
                os.path.join(root, name)
            ).convert('RGB').save(
                os.path.join(save_path, root, name.replace('.png', '.jpg'))
            )
        if '.jpg' in name:
            print(
                os.path.join(root, name)
            )
            Image.open(
                os.path.join(root, name)
            ).convert('RGB').save(
                os.path.join(save_path, root, name)
            )
    