#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import pprint
import pathlib
import inspect
import importlib
import numpy as np
import pickle as pkl
from typing import *

### uncomment the code below to target other python dists. This one is for py3.7

### CODE TO FIND STANDARD PYTHON MODULES ###

# # lib dist-packages path:
# std_module_list = []
# DIST_PACKAGES_PATH = os.path.expanduser("~/anaconda3/lib/python3.7")
# for fname in os.listdir(DIST_PACKAGES_PATH):
#     if fname != "site-packages":
#         std_module_list.append(pathlib.Path(fname).stem)
#         # std_module_list.append(os.path.join(
#         #     DIST_PACKAGES_PATH, fname
#         # ))
# print(std_module_list)

### END OF CODE TO FIND STANDARD PYTHON MODULES ###

std_module_list = ['socket', 'fnmatch', 'netrc', 'decimal', 'ssl', 'getpass', 'tracemalloc', 'webbrowser', 'imaplib', 'chunk', 'lzma', 'stat', 'formatter', '_sysconfigdata_powerpc64le_conda_cos7_linux_gnu', 'encodings', 'pathlib', 'bz2', 'optparse', '_sysconfigdata_i686_conda_cos6_linux_gnu', 'asynchat', 'urllib', '_sitebuiltins', 'uu', 'test', 'io', '_py_abc', 'socketserver', 'pipes', 'html', 'ipaddress', 'code', 'genericpath', 'xmlrpc', 'asyncore', 'random', 'struct', 'argparse', 'difflib', 'dis', '__future__', 'tarfile', 'concurrent', 'getopt', 'sched', '_sysconfigdata_aarch64_conda_cos7_linux_gnu', 'locale', '_markupbase', 'dummy_threading', 'dataclasses', 'cgi', 'cmd', 'xml', 'importlib', 'sre_constants', 'wave', 'imghdr', 'threading', 'gettext', 'pstats', 'this', 'pty', 'wsgiref', 'zipfile', '__phello__.foo', 'sqlite3', 'http', 'runpy', '_osx_support', 'contextvars', 'antigravity', 'lib-dynload', 'turtle', 'tree_sitter-0.20.0', 'codeop', 'mimetypes', 'enum', 'tempfile', 'asyncio', 'rlcompleter', 'keyword', '_strptime', '_weakrefset', 'selectors', 'pyclbr', '_pydecimal', 'cgitb', 'telnetlib', 'csv', '_bootlocale', 'imp', 'tree_sitter', 'reprlib', 'typing', '_collections_abc', 'glob', 'macpath', 'ntpath', 'venv', 'nturl2path', '_pyio', 'textwrap', 'ctypes', 'hmac', 'pydoc_data', 'hashlib', 'crypt', 'py_compile', 'cProfile', 'sre_parse', 'uuid', 'functools', 'traceback', 'config-3', 'stringprep', 'compileall', 'contextlib', 'distutils', 'bdb', 'symtable', 'shutil', 'smtplib', 'pydoc', 'numbers', 'symbol', 'json', 'statistics', 'logging', 'shlex', 'doctest', 'token', 'codecs', 'queue', 'copyreg', 'collections', 'trace', '_sysconfigdata_x86_64_conda_cos6_linux_gnu', 'lib2to3', 'plistlib', '_compat_pickle', 'modulefinder', 'ast', 'fractions', 'copy', 'pickle', 'linecache', 'sndhdr', 'gzip', 'mailcap', 'smtpd', 'bisect', 'aifc', 'quopri', 'pprint', 'string', 'weakref', 'inspect', 'site', 'sunau', 'heapq', 'nntplib', 'opcode', 'turtledemo', 'curses', '_sysconfigdata_m_linux_x86_64-linux-gnu', 'ensurepip', 'pdb', 'subprocess', 'mailbox', 'configparser', 'types', 'binhex', 'shelve', 'timeit', 'LICENSE', 'datetime', 'unittest', 'pkgutil', 'profile', 'pickletools', 'abc', 'fileinput', 'warnings', 'operator', 'posixpath', 'multiprocessing', 'email', 'xdrlib', 'calendar', '_sysconfigdata_x86_64_apple_darwin13_4_0', 'struct', 'tkinter', 'ftplib', 'idlelib', 'platform', 'sre_compile', '_dummy_thread', 'poplib', 'secrets', 'signal', 'colorsys', '_compression', 'filecmp', 'tabnanny', '__pycache__', 'sysconfig', 'tty', 'base64', '_threading_local', 'zipapp', 'tokenize', 're', 'dbm', 'os']

# serialize the default value of a parameter.
def serialize_default_value(default) -> Union[str, bool, int, float, None]:
    if isinstance(default, np.bool_): 
        return bool(default)
    elif isinstance(default, bool):
        return bool(default)
    elif isinstance(default, (int, float, str)): 
        return default
    elif default is None: return default
    else: return str(default)
# convert inspect.Parameter to regular dictionary (for serialization as a json.).
def get_parameter_dict(p: inspect.Parameter) -> dict:
    # incorporate annotation value.
    return {
        "kind": str(p.kind),
        "has_default_value": p.default != p.empty,
        "default": serialize_default_value(p.default),
    }
# convert OrderedDict of inspect.Parameter objects to dict of dicts.
def get_parameters_dict(params: OrderedDict[str, inspect.Parameter]) -> dict:
    d = {}
    for k,v in params.items():
        d[k] = get_parameter_dict(v) 

    return d
# get dictionary from inspect.Signature object.
def get_signature_dict(name: str, module: str, 
                       signature: inspect.Signature):
    qualified_name = f"{module}.{name}"
    d = {
        "name": name, "module": module,
        "qualified_name": qualified_name
    }
    d["return_type"] = str(signature.return_annotation)
    d["parameters"] = get_parameters_dict(signature.parameters)

    return d
# get all the signatures for a module (non recursively).
def get_module_signatures(module_name) -> Tuple[dict, dict, int, set]:
    module = importlib.import_module(module_name)
    signatures = {}
    error_cases = {}
    skipped_attr_count = 0
    skipped_attr_set = set()
    for attr in dir(module):
        func = getattr(module, attr)
        # <class 'function'> or <class 'builtin_function_or_method'>
        if type(func) in [type(os.path.expanduser), type(os.remove)]:
            try: 
                signature = inspect.signature(func)
                signature_dict = get_signature_dict(
                    attr, module_name, signature
                )
            except ValueError as e:
                print(f"Encountered error: '{e}' for \x1b[1m{attr}\x1b[0m")
                try: error_cases[attr].append((attr, e))
                except KeyError: 
                    error_cases[attr] = [(attr, e)]
                continue
            # add to signatures map.
            try: signatures[attr].append(signature_dict)
            except KeyError: signatures[attr] = [signature_dict]
        else:
            skipped_attr_count += 1
            skipped_attr_set.add(type(func))
            print(f"\x1b[34mskipping {attr}: {type(func)}\x1b[0m")

    return signatures, error_cases, skipped_attr_count, skipped_attr_set

def merge_dicts(d1: Dict[str, list], d2: Dict[str, list]) -> Dict[str, list]:
    dm = {}
    # cases when one of the two dicts is empty.
    if d1 == {}: return d2
    elif d2 == {}: return d1
    # find the union of all keys.
    all_keys = set(d1.keys())
    all_keys = all_keys.union(d2.keys())
    # iterate over the union of all keys.
    for k in all_keys:
        dm[k] = d1.get(k,[]) + d2.get(k,[])

    return dm

# recursively get function signatures from a module (process children that are modules themselves.).
def get_module_signatures_r(module_name) -> Tuple[dict, dict, int, set]:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # this error case is triggered when something like the os module is one of the attributes of a submodule/module
        # i.e. reference to a standard python library or some other library within a module.
        # the ideal behaviour is to ignore these, which is followed here.
        # print(e)
        return {}, {}, 0, set()
    signatures = {}
    error_cases = {}
    skipped_attr_count = 0
    skipped_attr_set = set()
    for attr in dir(module):
        func = getattr(module, attr)
        # if 'func' is of type <class 'module'>
        if type(func) == type(os):
            sub_signatures, sub_error_cases, sub_skipped_attr_count, sub_skipped_attr_set = get_module_signatures_r(f"{module_name}.{attr}")
            signatures = merge_dicts(signatures, sub_signatures)
            error_cases = merge_dicts(error_cases, sub_error_cases)
            skipped_attr_count += sub_skipped_attr_count
            skipped_attr_set = skipped_attr_set.union(sub_skipped_attr_set)
        # <class 'function'> or <class 'builtin_function_or_method'>
        elif type(func) in [type(os.path.expanduser), type(os.remove)]:
            try: 
                signature = inspect.signature(func)
                signature_dict = get_signature_dict(
                    attr, module_name, signature
                )
            except ValueError as e:
                # print(f"Encountered error: '{e}' for \x1b[1m{attr}\x1b[0m")
                try: error_cases[attr].append((attr, e))
                except KeyError: 
                    error_cases[attr] = [(attr, e)]
                continue
            # add to signatures map.
            try: signatures[attr].append(signature_dict)
            except KeyError: signatures[attr] = [signature_dict]
        else:
            skipped_attr_count += 1
            skipped_attr_set.add(type(func))
            # print(f"\x1b[34mskipping {attr}: {type(func)}\x1b[0m")
    return signatures, error_cases, skipped_attr_count, skipped_attr_set
# list of modules whose function signatures are to be extracted.
module_list = ["torch", "numpy", "pandas", "matplotlib.pyplot"]+std_module_list
# all module signature data.
all_module_signatures = {
    "module_list": module_list,
    "signatures": {}
}
# iterate over list of modules to be explored.
all_signatures = {}
for module in module_list:
    signatures, error_cases, skipped_attr_count, skipped_attrs = get_module_signatures_r(module)
    # all_module_signatures["signatures"][module] = signatures
    all_signatures = merge_dicts(all_signatures, signatures)
    # pprint.pprint(signatures)
    # pprint.pprint(error_cases)
    print(f"\x1b[34;1m{module} statistics:\x1b[0m")
    print(f"found signatures for \x1b[32;1m{len(signatures)}\x1b[0m functions")
    print(f"errored out for \x1b[31;1m{len(error_cases)}\x1b[0m functions")
    # pprint.pprint(skipped_attrs)
    print(f"skipped \x1b[34;1m{skipped_attr_count}\x1b[0m attributes")
all_module_signatures["signatures"] = all_signatures

class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)
# serialize/save all module, function signatures.
with open("module_signatures.json", "w") as f:
    json.dump(all_module_signatures, f, 
              cls=CustomJSONizer, indent=4)
# with open("module_signatures.pkl", "wb") as f:
#     pkl.dump(all_module_signatures, f)