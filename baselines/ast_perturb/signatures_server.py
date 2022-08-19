import json
import jinja2
from typing import *
from flask import g, Flask, jsonify, request

# list of standard python modules (py3.7)
std_module_list = ['socket', 'fnmatch', 'netrc', 'decimal', 'ssl', 'getpass', 'tracemalloc', 'webbrowser', 'imaplib', 'chunk', 'lzma', 'stat', 'formatter', '_sysconfigdata_powerpc64le_conda_cos7_linux_gnu', 'encodings', 'pathlib', 'bz2', 'optparse', '_sysconfigdata_i686_conda_cos6_linux_gnu', 'asynchat', 'urllib', '_sitebuiltins', 'uu', 'test', 'io', '_py_abc', 'socketserver', 'pipes', 'html', 'ipaddress', 'code', 'genericpath', 'xmlrpc', 'asyncore', 'random', 'struct', 'argparse', 'difflib', 'dis', '__future__', 'tarfile', 'concurrent', 'getopt', 'sched', '_sysconfigdata_aarch64_conda_cos7_linux_gnu', 'locale', '_markupbase', 'dummy_threading', 'dataclasses', 'cgi', 'cmd', 'xml', 'importlib', 'sre_constants', 'wave', 'imghdr', 'threading', 'gettext', 'pstats', 'this', 'pty', 'wsgiref', 'zipfile', '__phello__.foo', 'sqlite3', 'http', 'runpy', '_osx_support', 'contextvars', 'antigravity', 'lib-dynload', 'turtle', 'tree_sitter-0.20.0', 'codeop', 'mimetypes', 'enum', 'tempfile', 'asyncio', 'rlcompleter', 'keyword', '_strptime', '_weakrefset', 'selectors', 'pyclbr', '_pydecimal', 'cgitb', 'telnetlib', 'csv', '_bootlocale', 'imp', 'tree_sitter', 'reprlib', 'typing', '_collections_abc', 'glob', 'macpath', 'ntpath', 'venv', 'nturl2path', '_pyio', 'textwrap', 'ctypes', 'hmac', 'pydoc_data', 'hashlib', 'crypt', 'py_compile', 'cProfile', 'sre_parse', 'uuid', 'functools', 'traceback', 'config-3', 'stringprep', 'compileall', 'contextlib', 'distutils', 'bdb', 'symtable', 'shutil', 'smtplib', 'pydoc', 'numbers', 'symbol', 'json', 'statistics', 'logging', 'shlex', 'doctest', 'token', 'codecs', 'queue', 'copyreg', 'collections', 'trace', '_sysconfigdata_x86_64_conda_cos6_linux_gnu', 'lib2to3', 'plistlib', '_compat_pickle', 'modulefinder', 'ast', 'fractions', 'copy', 'pickle', 'linecache', 'sndhdr', 'gzip', 'mailcap', 'smtpd', 'bisect', 'aifc', 'quopri', 'pprint', 'string', 'weakref', 'inspect', 'site', 'sunau', 'heapq', 'nntplib', 'opcode', 'turtledemo', 'curses', '_sysconfigdata_m_linux_x86_64-linux-gnu', 'ensurepip', 'pdb', 'subprocess', 'mailbox', 'configparser', 'types', 'binhex', 'shelve', 'timeit', 'LICENSE', 'datetime', 'unittest', 'pkgutil', 'profile', 'pickletools', 'abc', 'fileinput', 'warnings', 'operator', 'posixpath', 'multiprocessing', 'email', 'xdrlib', 'calendar', '_sysconfigdata_x86_64_apple_darwin13_4_0', 'struct', 'tkinter', 'ftplib', 'idlelib', 'platform', 'sre_compile', '_dummy_thread', 'poplib', 'secrets', 'signal', 'colorsys', '_compression', 'filecmp', 'tabnanny', '__pycache__', 'sysconfig', 'tty', 'base64', '_threading_local', 'zipapp', 'tokenize', 're', 'dbm', 'os']

with open("module_signatures.json") as f:
    all_module_signatures = json.load(f)
app = Flask(__name__)

# home page (has search function):
@app.route("/")
def home():
    PY_FUNC_LIST = list(all_module_signatures["signatures"].keys())
    return jinja2.Template(r"""
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Search Python Functions</title>
        <style>
            * { box-sizing: border-box; }
            body {
            font: 16px Arial;
            }
            .autocomplete {
            /*the container must be positioned relative:*/
            position: relative;
            display: inline-block;
            }
            input {
            border: 1px solid transparent;
            background-color: #f1f1f1;
            padding: 10px;
            font-size: 16px;
            }
            input[type=text] {
            background-color: #f1f1f1;
            width: 100%;
            }
            input[type=submit] {
            background-color: DodgerBlue;
            color: #fff;
            }
            .autocomplete-items {
            position: absolute;
            border: 1px solid #d4d4d4;
            border-bottom: none;
            border-top: none;
            z-index: 99;
            /*position the autocomplete items to be the same width as the container:*/
            top: 100%;
            left: 0;
            right: 0;
            }
            .autocomplete-items div {
            padding: 10px;
            cursor: pointer;
            background-color: #fff;
            border-bottom: 1px solid #d4d4d4;
            }
            .autocomplete-items div:hover {
            /*when hovering an item:*/
            background-color: #e9e9e9;
            }
            .autocomplete-active {
            /*when navigating through the items using the arrow keys:*/
            background-color: DodgerBlue !important;
            color: #ffffff;
            }
        </style>
    </head>
    <body>
    <!--Make sure the form has the autocomplete function switched off:-->
    <form autocomplete="off" action="/search">
    <div class="autocomplete" style="width:300px;">
        <input id="pyFunctionSearch" type="text" name="q" placeholder="round">
    </div>
    <input type="submit">
    </form>
    </body>
    <script>
        var py_func_list = {{ PY_FUNC_LIST }}
        function autocomplete(inp, arr) {
        /*the autocomplete function takes two arguments,
        the text field element and an array of possible autocompleted values:*/
        var currentFocus;
        /*execute a function when someone writes in the text field:*/
        inp.addEventListener("input", function(e) {
            var a, b, i, val = this.value;
            /*close any already open lists of autocompleted values*/
            closeAllLists();
            if (!val) { return false;}
            currentFocus = -1;
            /*create a DIV element that will contain the items (values):*/
            a = document.createElement("DIV");
            a.setAttribute("id", this.id + "autocomplete-list");
            a.setAttribute("class", "autocomplete-items");
            /*append the DIV element as a child of the autocomplete container:*/
            this.parentNode.appendChild(a);
            /*for each item in the array...*/
            for (i = 0; i < arr.length; i++) {
                /*check if the item starts with the same letters as the text field value:*/
                if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
                /*create a DIV element for each matching element:*/
                b = document.createElement("DIV");
                /*make the matching letters bold:*/
                b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
                b.innerHTML += arr[i].substr(val.length);
                /*insert a input field that will hold the current array item's value:*/
                b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
                /*execute a function when someone clicks on the item value (DIV element):*/
                    b.addEventListener("click", function(e) {
                    /*insert the value for the autocomplete text field:*/
                    inp.value = this.getElementsByTagName("input")[0].value;
                    /*close the list of autocompleted values,
                    (or any other open lists of autocompleted values:*/
                    closeAllLists();
                });
                a.appendChild(b);
                }
            }
        });
        /*execute a function presses a key on the keyboard:*/
        inp.addEventListener("keydown", function(e) {
            var x = document.getElementById(this.id + "autocomplete-list");
            if (x) x = x.getElementsByTagName("div");
            if (e.keyCode == 40) {
                /*If the arrow DOWN key is pressed,
                increase the currentFocus variable:*/
                currentFocus++;
                /*and and make the current item more visible:*/
                addActive(x);
            } else if (e.keyCode == 38) { //up
                /*If the arrow UP key is pressed,
                decrease the currentFocus variable:*/
                currentFocus--;
                /*and and make the current item more visible:*/
                addActive(x);
            } else if (e.keyCode == 13) {
                /*If the ENTER key is pressed, prevent the form from being submitted,*/
                e.preventDefault();
                if (currentFocus > -1) {
                /*and simulate a click on the "active" item:*/
                if (x) x[currentFocus].click();
                }
            }
        });
        function addActive(x) {
            /*a function to classify an item as "active":*/
            if (!x) return false;
            /*start by removing the "active" class on all items:*/
            removeActive(x);
            if (currentFocus >= x.length) currentFocus = 0;
            if (currentFocus < 0) currentFocus = (x.length - 1);
            /*add class "autocomplete-active":*/
            x[currentFocus].classList.add("autocomplete-active");
        }
        function removeActive(x) {
            /*a function to remove the "active" class from all autocomplete items:*/
            for (var i = 0; i < x.length; i++) {
            x[i].classList.remove("autocomplete-active");
            }
        }
        function closeAllLists(elmnt) {
            /*close all autocomplete lists in the document,
            except the one passed as an argument:*/
            var x = document.getElementsByClassName("autocomplete-items");
            for (var i = 0; i < x.length; i++) {
            if (elmnt != x[i] && elmnt != inp) {
            x[i].parentNode.removeChild(x[i]);
            }
        }
        }
        /*execute a function when someone clicks in the document:*/
        document.addEventListener("click", function (e) {
            closeAllLists(e.target);
        });
        }
        autocomplete(document.getElementById("pyFunctionSearch"), py_func_list);
    </script>
</html>
""").render(PY_FUNC_LIST=PY_FUNC_LIST)

@app.route("/module-list")
def show_module_list1():
    return jsonify(all_module_signatures["module_list"])
@app.route("/module_list")
def show_module_list2():
    return jsonify(all_module_signatures["module_list"])

@app.route("/std-module-list")
def show_std_module_list1():
    return jsonify(std_module_list)
@app.route("/std_module_list")
def show_std_module_list2():
    return jsonify(std_module_list)
@app.route("/std-module_list")
def show_std_module_list3():
    return jsonify(std_module_list)
@app.route("/std_module-list")
def show_std_module_list4():
    return jsonify(std_module_list)

# search for functions.
@app.route("/search")
def search():
    query = request.args.get("q")
    return jsonify(all_module_signatures["signatures"].get(query, {}))

app.run(debug=True, port=9090)