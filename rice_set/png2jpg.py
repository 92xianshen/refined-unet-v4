# -*- coding: utf-8 -*-

import os
from PIL import Image

input_path = '../../rice_result/'
output_path = '../../rice_jpg/'

for root, dirs, files in os.walk(input_path):
    print('root:', root)
    if not os.path.exists(os.path.join(output_path, root)):
        os.makedirs(os.path.join(output_path, root))
    
    for dir in dirs:
        print('dir:', dir)
        if not os.path.exists(os.path.join(output_path, root, dir)):
            os.makedirs(os.path.join(output_path, root, dir))
    
    for name in files:
        if '.png' in name:
            print(os.path.join(root, name))
            Image.open(
                os.path.join(root, name)
            ).convert('RGB').save(
                os.path.join(output_path, root, name.replace('.png', '.jpg'))
            )
        if '.jpg' in name:
            print(
                os.path.join(root, name)
            )
            Image.open(
                os.path.join(root, name)
            ).convert('RGB').save(
                os.path.join(output_path, root, name)
            )
    