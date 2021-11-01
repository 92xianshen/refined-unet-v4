# -*- coding: utf-8 -*-

import os
from PIL import Image

task = 'a=140, b=.0625, r=3'

result_path = os.path.join('../result/', task)
save_path = os.path.join('../qualitative_evaluation/', task)

for root, dirs, files in os.walk(result_path):
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
    