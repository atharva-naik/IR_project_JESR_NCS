#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import ast
import _ast
import random
import string
import pprint
import tempfile
from typing import *
from ast_unparse37 import Unparser

def details(varname: str, obj: object):
    """details of a particular variable given the variable name and object/reference corresponding to it.

    Args:
        varname (str): name of the vairable as a string
        obj (object): the actual variable.
    """
    def attrs(obj: object) -> List[str]:
        useful_attrs = []
        for attr in dir(obj):
            # ignore dunder fields
            if not (attr.startswith("__") and attr.endswith("__")):
                useful_attrs.append(attr)

        return useful_attrs
    try: length = len(obj)
    except TypeError as e: 
        length = e
    print(f"""\x1b[34;1m{varname}\x1b[0m
obj: {obj}
\x1b[33mtype: {type(obj)}\x1b[0m
\x1b[32mlen: {length}\x1b[0m
attrs: {attrs(obj)}
""")
# INFO:
# 1. root node of an ast is always an object of type _ast.Module.
# 2. 
# CODE = """
# import os, re
# from torch.utils.data import *
# import tensorflow as tf, torch as pt
# import stuff as other_stuff
# from subprocess import Popen

# def xparse(x, *_args, y=-1, **kw_args) -> int:
#     eg = Example(var=1)
#     y = eg.tree(x)
    
#     return y.tree"""
CODE = 'y = max(torch.np.func(x))'
def rand_str(k: int=10) -> str:
    """sample random string of length k.
    Args:
        k (int, optional): _description_. Defaults to 10.
    Returns:
        str: random string output.
    """
    randstr = ""
    for char in random.sample("_"+string.ascii_letters+string.digits, k=k):
        randstr += char
    return randstr

def hasanyattr(obj, attr_list: Union[List[str], Tuple[str]]) ->  bool:
    for attr_name in attr_list:
        if hasattr(obj, attr_name):
            return True
    return False

# locate library functions and record there offsets.
class LibFunctionFinder(ast.NodeVisitor):
    def visit_Call(self, node):
        attr_call = False
        if type(node.func) == _ast.Attribute:
            attr_call = True
        if not hasattr(self, "lib_fn_list"):
            self.lib_fn_ctr = 0
            self.lib_fn_dist = {}
            self.lib_fn_list = []
        if not hasattr(self, "builtin_fns"):
            self.builtin_fns = [
                "abs",
                "aiter",
                "all",
                "any",
                "anext",
                "ascii",
                "bin",
                "bool",
                "breakpoint",
                "bytearray",
                "bytes",
                "callable",
                "chr",
                "classmethod",
                "compile",
                "complex",
                "delattr",
                "dict",
                "dir",
                "divmod",
                "enumerate",
                "eval",
                "exec",
                "filter",
                "float",
                "format",
                "frozenset",
                "getattr",
                "globals",
                "hasattr",
                "hash",
                "help",
                "hex",
                "id",
                "input",
                "int",
                "isinstance",
                "issubclass",
                "iter",
                "len",
                "list",
                "locals",
                "map",
                "max",
                "memoryview",
                "min",
                "next",
                "object",
                "oct",
                "open",
                "ord",
                "pow",
                "print",
                "property",
                "range",
                "repr",
                "reversed",
                "round",
                "set",
                "setattr",
                "slice",
                "sorted",
                "staticmethod",
                "str",
                "sum",
                "super",
                "tuple",
                "type",
                "vars",
                "zip",
                "__import__",
            ]
        if attr_call:
            fn_name = node.func.attr
            full_name = node.func.value.id + "." + node.func.attr
        else:
            fn_name = node.func.id
            full_name = fn_name
        # print(f"end: {node.f}")
        self.lib_fn_list.append({
            "id": self.lib_fn_ctr,
            "node": node,
            "name": fn_name,
            "full_name": full_name,
            "line": node.lineno,
            "start": node.col_offset,
            "isbuiltin": fn_name in self.builtin_fns,
            "isattrcall": attr_call, 
            # "end": node.end_col_offset,
        })
        try: self.lib_fn_dist[fn_name] += 1
        except KeyError: 
            self.lib_fn_dist[fn_name] = 1
        self.lib_fn_ctr += 1
        print(f"\x1b[32;1mvisited \x1b[34;1m'{fn_name}'\x1b[0m")
        pprint.pprint(self.lib_fn_list)
        if hasattr(super(), "visit_Call"):
            return super().visit_Call(node)
        else: return super().generic_visit(node)

class ImportedModuleFinder(ast.NodeVisitor):
    def visit_Import(self, node) -> Any:
        return super().visit_Import(node)

# perturbation class.
class PerturbAst:
    def __init__(self, tree):
        ast.walk(tree)

tree = ast.parse(bytes(CODE,"utf8"))
lff = LibFunctionFinder()
lff.visit(tree)
"""
tree = ast.parse(bytes(CODE, "utf8"))
content = ""
fname = f"{rand_str(16)}.py"
with open(fname, "w") as f:
    Unparser(tree, file=f)
with open(fname, "r") as f:
    content = f.read()
os.remove(fname)
print("\x1b[34;1munparsed:\x1b[0m")
print(content)
"""