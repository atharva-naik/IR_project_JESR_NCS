#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import ast
import _ast
import copy
import json
import random
import string
import pprint
import numpy as np
from typing import *
from fuzzywuzzy import fuzz
from dataclasses import dataclass
from ast_unparse37 import Unparser
from sortedcontainers import SortedSet

# show details of an object.
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
CODES = ['y = max(torch.nn.functional.tanh(x))', "[i for i in x]", "[i+1 for i in [1/j for j in range(5)]]", "[max(torch.nn.functional.tanh(x)) for x in range(y)]", "{i+1 for i in {1/j for j in range(5)}}", "x+'3.0'", "y=x+3.0", '''x = x + "hello" +'there'+ z''']
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

modules_signatures = json.load(open("module_signatures.json"))
signatures = modules_signatures['signatures']
fn_names = set()
for key, value in signatures.items():
    fn_names.add(key)
    for rec in value:
        fn_names.add(rec["qualified_name"])
fn_names = list(fn_names)
builtin_fn_names = ["abs", "aiter","all", "any", "anext", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes", "callable", "chr", "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len", "list", "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip", "__import__"]
fn_names.extend(builtin_fn_names)
# perturbation class.
@dataclass(frozen=False)
class RuleFilter:
    rule1:bool=False
    rule2:bool=False
    rule3:bool=False
    rule4:bool=False
    rule1_metadata:str="Library function substitution"
    rule2_metadata:str="List comprehension to set comprehension"
    rule3_metadata:str="Set comprehension to list comprehension"
    rule4_metadata:str="Change type of constant `int`/`float` to `str` and vice-versa"
    random_fn_sub:bool=True
    recursive_sub:bool=False

    def state(self):
        return [getattr(self, f"rule{i+1}") for i in range(len(self))]

    def setState(self, state: List[bool]):
        for i, val in enumerate(state):
            setattr(self, f"rule{i+1}", val)

    def getMaskFromNames(self, names: List[str]):
        rules_mask = []
        for i in range(len(self)):
            if f"rule{i+1}" in names:
                rules_mask.append(1)
            else: rules_mask.append(0)

        return rules_mask

    def getRuleCount(self):
        ctr = 0
        try:
            getattr(self, f"rule{ctr+1}")
            ctr += 1
        except AttributeError: 
            return ctr

    def __len__(self):
        return 4

    def allowByName(self, name: str):
        validNames = [f'rule{i+1}' for i in range(len(self))]
        assert name in validNames, f"invalid rule name. Must be in {validNames}"
        setattr(self, name, True)

    def blockByName(self, name: str):
        validNames = [f'rule{i+1}' for i in range(len(self))]
        assert name in validNames, f"invalid rule name. Must be in {validNames}"
        setattr(self, name, False)

    def allowAll(self):
        for i in range(len(self)):
            setattr(self, f"rule{i+1}", True)
    
    def blockAll(self):
        for i in range(len(self)):
            setattr(self, f"rule{i+1}", False)

    @classmethod
    def AllowAll(cls):
        return RuleFilter(
            rule1=True,
            rule2=True,
            rule3=True, 
            rule4=True,
        )

    @classmethod
    def BlockAll(cls):
        return RuleFilter(
            rule1=False,
            rule2=False,
            rule3=False,
            rule4=False,
        )

    def __call__(self, index: int):
        return getattr(self, f"rule{index}")

    def metadata(self, index: int):
        return getattr(self, f"rule_metadata{index}")

    def show(self):
        print("\x1b[34;1mPerturbing ast with following rules:\x1b[0m")
        for i in range(len(self)): 
            print(f"{i+1}. "+getattr(self, f"rule{i+1}_metadata"))

# class for perturbing AST parse tree.
class PerturbAst(ast.NodeTransformer):
    """Perturb AST using various rules:
    1. Library function substitution.
        a) Randomly replace function names of known library function with other known library functions.
        b) Find function with closest signature and function name and substitute it.
    2. Replace list comprehension with set comprehension.

    Args:
        ast (_type_): _description_
    """
    def __init__(self, *args, rule_filter: Union[RuleFilter, None]=None, **kwargs):
        self.visit_sequence = []
        super(PerturbAst, self).__init__(*args, **kwargs)
        if rule_filter is None:
            self.rule_filter = RuleFilter.allowAll()
        else: self.rule_filter = rule_filter
        self.use_rules = {}

    def init(self):
        global fn_names
        global signatures
        
        self.applied_rules = SortedSet()
        self.signatures = signatures
        self.lib_fn_names = fn_names
        
    def reset(self):
        # clear visit sequence.
        self.visit_sequence = []
        self.init()
    
    def compare_fn_names(self, f1, f2):
        f1_list = f1.split(".")
        f2_list = f2.split(".")
        if (f1_list[-1] == f2_list[-1]) and f2.endswith(f1):
            return True
        return False

    def choice(self):
        import random
        return random.choice(self.applied_rules)

    def fn_name_sim_score(self, target_name: str, candidate_name: str):
        # use levenshtein distance. (it is from 0 to 100, so norm by dividing by 100)
        lev_score = fuzz.token_sort_ratio(target_name, candidate_name)/100
        # penalize a bit for length mismatch. (clamp minimum value at 0.5)
        len_score = 1-min((abs(len(candidate_name)-len(target_name))/len(target_name)), 0.5)

        return lev_score*len_score

    def fn_sig_sim_score(self, p1: dict, p2: dict, r1: str, r2: str):
        # return type similarity score.
        # stupid binary version
        ret_sim_score = 1 if r1 == r2 else 0
        p_len_score = 1 if len(p1) == len(p2) else 0
        cmp_iter = zip()
        p_sim_z = 0
        p_sim_score = 0
        for pn1, pn2 in cmp_iter:
            if p1[pn1]["kind"] == "KEYWORD_ONLY":
                if pn1 == pn2:
                    p_sim_score += int(pn1 == pn2)
                p_sim_z += 1
            if p1[pn1]["has_default_value"]:
                if p1[pn1]["default"] == p2[pn2]["default"]:
                    p_sim_score += 1
                p_sim_z += 1
            # TODO: add argument type comparison here.
        p_sim_score /= p_sim_z

        return ret_sim_score + p_len_score + p_sim_score

    def fn_def_sim_score(self, f1_dict: dict, f2_dict: dict):
        name_sim_score = self.fn_name_sim_score(f1_dict["name"], f2_dict["name"])
        sig_sim_score = self.fn_sig_sim_score(
            f1_dict["parameters"], f2_dict["parameters"], 
            f1_dict["return_type"], f2_dict["return_type"]
        )

        return name_sim_score + sig_sim_score
    
    def is_user_defined(self, fn_name):
        for lib_fn_name in self.lib_fn_names:
            if self.compare_fn_names(fn_name, lib_fn_name):
                return False
        return True
    
    def get_full_name(self, value, attr):
        """recursively get full name of function from a call with attributed prefix."""
        if isinstance(value, _ast.Attribute):
            return self.get_full_name(value.value, value.attr+"."+attr)
        # for cases like open("file.txt").read(), "".join(), Entry.objects.filter()[:1].get()
        # and (datetime.datetime.now() - datetime.timedelta(days=7)).date()
        elif isinstance(value, (_ast.Call, _ast.Str, _ast.Subscript, _ast.BinOp)): 
            return attr
        # base case.
        else: return value.id+"."+attr
    
    def sample_neg_fn(self, fn: str) -> str:
        """return the first random function that has a different qualified name
        than the reference function `fn`.
        Args:
            fn (str): reference function name to make sure a different function is sampled.
        Returns:
            (str): name of the sampled function.
        """
        shuffled = random.sample(
            list(self.signatures.keys()), 
            k=len(self.signatures)
        )
        for k in shuffled:
            rec = self.signatures[k][0]
            name = rec["name"]
            qualified_name = rec["qualified_name"]
            if not self.compare_fn_names(fn, qualified_name):
                return name

    def rules_applied(self):
        return list(self.applied_rules)

    def applicable_rules(self, tree: _ast.Module) -> list:
        test_tree = copy.deepcopy(tree)
        # get original state of rule filter.
        # the state of the rule filter is the value of all the `rule` attrs.
        state: Dict[str, bool] = self.rule_filter.state()
        self.rule_filter.allowAll() # enable/allow all rules.
        self.visit(test_tree) # traverse tree to enumerate all applicable rules and make changes on the copied `test_tree`
        rules = self.applied_rules # list of all applicable rules.
        self.rule_filter.setState(state) # restore original state of the rule filter.
        self.reset() # reset traversal specific attrs.

        return list(rules)

    def __call__(self, code: str, rule_probs: Union[List[float], None]=None, verbose=False) -> Tuple[_ast.Module, dict]:
        # get parsed AST.
        tree: _ast.Module = ast.parse(bytes(code, "utf8"))
        # find list of applicable rules.
        rules: List[str] = self.applicable_rules(tree)
        rules_mask: List[int]= self.rule_filter.getMaskFromNames(rules)
        # pick a random rule.
        N: int = len(self.rule_filter)
        if rule_probs is None:
            rule_probs = np.ones(N)
        rule_probs = np.array(rule_probs)*rules_mask
        rule_probs /= rule_probs.sum()
        sampled_rule = np.random.choice(
            [f"rule{i+1}" for i in range(N)], 
            1, p=rule_probs
        )[0]
        # set rule filter to block mode and enable sampled filter by name.
        self.rule_filter.blockAll()
        self.rule_filter.allowByName(sampled_rule)
        # get the perturbed tree.
        perturbed_tree: _ast.Module = self.visit(tree)
        if verbose:
            print(f"original code: {code}")
            print(f"`PerturbAst.visit` returned code as `{type(perturbed_tree)}` object")
            print(f"new code: {serialize_tree(perturbed_tree)}")
            print(f"visit sequence: {self.visit_sequence}")
        # reset perturber.
        self.reset()
        
        return perturbed_tree, {
            "original_code": code,
            "perturbed_code": serialize_tree(perturbed_tree),
            "rule_applied": sampled_rule,
        }

    def is_rule1_applicable(self, func):
        return not(self.is_user_defined(func))

    def apply_rule1_rand(self, func):
        if isinstance(func, _ast.Name):
            neg_fn = self.sample_neg_fn(func.id)
            func.id = neg_fn
        elif isinstance(func, _ast.Attribute):
            neg_fn = self.sample_neg_fn(func.attr) # print(neg_fn)
            func.attr = neg_fn

    def apply_rule1_smart(self, func):
        if isinstance(func, _ast.Name):
            neg_fn = self.sample_neg_fn(func.id)
            func.id = neg_fn
        elif isinstance(func, _ast.Attribute):
            neg_fn = self.sample_neg_fn(func.attr) # print(neg_fn)
            func.attr = neg_fn

    def visit_Num(self, node):
        if self.rule_filter(4):
            self.applied_rules.add("rule4")
            node = _ast.Str(
                s=str(node.n),
                lineno=node.lineno,
                col_offset=node.col_offset
            )
            return super(PerturbAst, self).generic_visit(node)
        else: return super(PerturbAst, self).generic_visit(node)

    def visit_Str(self, node):
        if self.rule_filter(4):
            self.applied_rules.add("rule4")
            # check if int conversion is allowed.
            try: n = int(node.s)
            except ValueError:
                # check if float conversion is allowed.
                try: n = float(node.s)
                # pick a random float or int.
                except ValueError: 
                    subs_type = random.choice([int,float])
                    n = subs_type(len(node.s))
            node = _ast.Num(
                n=n,
                lineno=node.lineno,
                col_offset=node.col_offset
            )
            return super(PerturbAst, self).generic_visit(node)
        else: return super(PerturbAst, self).generic_visit(node)

    def visit_Call(self, node):
        attr_call = False
        if type(node.func) == _ast.Attribute:
            attr_call = True
        if attr_call:
            value = node.func.value
            fn_name = node.func.attr
            full_name = self.get_full_name(value, fn_name)
        else:
            fn_name = node.func.id
            full_name = fn_name
        # check if rule1 is applicable
        if self.is_rule1_applicable(full_name) and self.rule_filter(1):
            self.applied_rules.add("rule1")
            if self.rule_filter.random_fn_sub:
                self.apply_rule1_rand(node.func)
            else:
                self.apply_rule1_smart(node.func)

        return super(PerturbAst, self).generic_visit(node)

    def visit_ListComp(self, node: _ast.ListComp) -> Any:
        if self.rule_filter(2):
            self.applied_rules.add("rule2")
            if self.rule_filter.recursive_sub: 
                node = _ast.SetComp(
                    elt=node.elt,
                    generators=node.generators,
                )
                return super(PerturbAst, self).generic_visit(node)
            else:
                return _ast.SetComp(
                    elt=node.elt,
                    generators=node.generators,
                )
        else: return super(PerturbAst, self).generic_visit(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        self.visit_sequence.append(node)
        return super(PerturbAst, self).generic_visit(node)

    def visit_SetComp(self, node: _ast.ListComp) -> Any:
        if self.rule_filter(3):
            self.applied_rules.add("rule3")
            if self.rule_filter.recursive_sub: 
                node = _ast.ListComp(
                    elt=node.elt,
                    generators=node.generators,
                )
                return super(PerturbAst, self).generic_visit(node)
            else:
                return _ast.ListComp(
                    elt=node.elt,
                    generators=node.generators,
                )
        else: return super(PerturbAst, self).generic_visit(node)

def serialize_tree(tree):
    # convert tree back to code block.
    content = ""
    fname = f"{rand_str(16)}.py"
    with open(fname, "w") as f:
        Unparser(tree, file=f)
    with open(fname, "r") as f:
        content = f.read()
    os.remove(fname)
    
    return content.strip("\n")

def perturb_codes(CODES: List[str], verbose: bool=False) -> List[dict]:
    rule_filter = RuleFilter.AllowAll()
    rule_filter.recursive_sub = True
    rule_filter.show()
    data_gen = PerturbAst(rule_filter=rule_filter)
    data_gen.init()
    ops = []
    for CODE in CODES:
        perturbed_tree, op_dict = data_gen(CODE, verbose=verbose)
        ops.append(op_dict)

    return ops

if __name__ == "__main__":
    pprint.pprint(perturb_codes(CODES, verbose=False))