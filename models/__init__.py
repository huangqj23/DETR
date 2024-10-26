'''
Author: huangqj23
Date: 2024-10-25 11:07:36
LastEditors: huangqj23
LastEditTime: 2024-10-26 17:35:23
FilePath: /DETR/models/__init__.py
Description: 

Copyright (c) 2024 by FutureAI, All Rights Reserved. 
'''
from .detr import build

def build_model(args):
    return build(args)