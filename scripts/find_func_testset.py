#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import tqdm
import pprint
import argparse
import os, ast, _ast

modules_signatures = json.load(open("ast_perturb/module_signatures.json"))
signatures = modules_signatures['signatures']
fn_names = set()
for key, value in signatures.items():
    fn_names.add(key)
    for rec in value:
        fn_names.add(rec["qualified_name"])
fn_names = list(fn_names)
builtin_fn_names = ["abs", "aiter","all", "any", "anext", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes", "callable", "chr", "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len", "list", "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip", "__import__"]
fn_names.extend(builtin_fn_names)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--path', type=str, default="candidate_snippets.json",
                    help='path to test data candidates (default: candidate_snippets.json)')
args = parser.parse_args()

# find library function calls.
class LibFunctionFinder(ast.NodeVisitor):
    def init(self):
        global fn_names
        # `lib_fn` attrs.
        self.lib_fn_ctr = 0
        self.lib_fn_dist = {}
        self.lib_fn_list = []
        self.lib_fn_names = fn_names
        
        self.matched_fns = []
        self.has_lib_fn = False
        
    def reset(self):
        matched_fns = self.matched_fns
        has_lib_fn = self.has_lib_fn
        self.init()
    
        return has_lib_fn, matched_fns
    
    def compare_fn_names(self, f1, f2):
        f1_list = f1.split(".")
        f2_list = f2.split(".")
        if (f1_list[-1] == f2_list[-1]) and f2.endswith(f1):
            # print("compare_fn_names:", f1, f2, f1.endswith(f2), f2.endswith(f1))
            return True
        return False
    
    def check_user_defined(self, fn_name):
        for lib_fn_name in self.lib_fn_names:
            if self.compare_fn_names(fn_name, lib_fn_name):
                self.matched_fns.append(lib_fn_name)
                self.has_lib_fn = True
                return True
        return False
    
    def get_full_name(self, value, attr):
        """recursively get full name of function from a call with attributed prefix."""
        if isinstance(value, _ast.Attribute):
            return self.get_full_name(value.value, value.attr+"."+attr)
        # for cases like open("file.txt").read(), "".join(), Entry.objects.filter()[:1].get()
        # and (datetime.datetime.now() - datetime.timedelta(days=7)).date()
        elif isinstance(value, (_ast.Call, _ast.Str, _ast.Subscript, _ast.BinOp, _ast.UnaryOp)): 
            return attr
        # base case.
        else: return value.id+"."+attr
    
    def visit_Call(self, node):
        attr_call = False
        if type(node.func) == _ast.Attribute:
            attr_call = True
        if attr_call:
            value = node.func.value
            fn_name = node.func.attr
            full_name = self.get_full_name(value, fn_name)
            # print(full_name)
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
            "is_attr_call": attr_call,
            "is_user_defined": self.check_user_defined(full_name),
        })
        try: self.lib_fn_dist[fn_name] += 1
        except KeyError: 
            self.lib_fn_dist[fn_name] = 1
        self.lib_fn_ctr += 1
        # print(f"\x1b[32;1mvisited \x1b[34;1m'{fn_name}'\x1b[0m")
        if hasattr(super(LibFunctionFinder, self), "visit_Call"):
            return super(LibFunctionFinder, self).visit_Call(node)
        else: return super(LibFunctionFinder, self).generic_visit(node)
        
def test_lib_fn_finder():
    CODE = """self.model1 = lib.function_base.flip(x)
self.model2 = abs(x)
self.model3 = iter(tuple(x))
self.model4 = abs(lib.function_base.flip(x))"""
    CODE = "numpy.flip(x).wow(y)"
    tree = ast.parse(bytes(CODE,"utf8"))
    lff = LibFunctionFinder()
    lff.init()
    lff.visit(tree)
    print(lff.reset())
    
def find_fn_test_subset(snippets):
    fn_snippet_ids = []
    lff = LibFunctionFinder()
    lff.init()
    pbar = tqdm.tqdm(
        enumerate(snippets), 
        total=len(snippets),
    )
    syntax_err_count = 0
    attr_err_count = 0
    for i, snippet in pbar:
        try:
            tree = ast.parse(bytes(snippet, "utf8"))
        except SyntaxError as e:
            pbar.set_description(f"SyntaxError: {e}")
            syntax_err_count += 1
            continue
        try: lff.visit(tree)
        except AttributeError as e:
            attr_err_count += 1
            # print("—"*20)
            # print(f"{e} for:", snippet)
            # print("—"*20)
        has_lib_fn, _ = lff.reset()
        if has_lib_fn: fn_snippet_ids.append(i)
    print(f"syntax error encountered for: {syntax_err_count} ({100*syntax_err_count/len(snippets):.2f}%)")
    print(f"attribute error encountered for: {attr_err_count} ({100*attr_err_count/len(snippets):.2f}%)")
            
    return fn_snippet_ids

# main loop.
if __name__ == "__main__":
    # test_lib_fn_finder()
    print(f"finding `lib_fn` test set for: \x1b[34;1m{args.path}\x1b[0m")
    data = json.load(open(args.path))
    snippets = data["snippets"]
    fn_snippet_ids = find_fn_test_subset(snippets)
    ratio = round(100*len(fn_snippet_ids)/len(snippets), 2)
    print(f"found \x1b[34;1m{len(fn_snippet_ids)}/{len(snippets)}\x1b[0m instances with lib functions ({ratio}% of the data)")
    data["lib_fn_snippet_ids"] = fn_snippet_ids
    with open(args.path, "w") as f:
        json.dump(data, f, indent=4)