#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

def write_tensor_to_libsvm(tensor, path: str, label: int=1, precision: int=6):        
    s = time.time()
    with open(path, "w") as f:
        for row in tensor:
            line = f"{label} "
            for i, el in enumerate(row):
                line += f"{i+1}: {round(el.item(), precision)}"
            line = line.strip()
            f.write(line + "\n")
    print(f"{time.time()-s}s")