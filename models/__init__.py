# This package contains main model files, for our "Universal Joint/Shared Space Encoder"
# some common utilites.
import os
def get_tok_path(model_name: str) -> str:
    assert model_name in ["codebert", "graphcodebert"]
    if model_name == "codebert":
        tok_path = os.path.expanduser("~/codebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/codebert-base"
    elif model_name == "graphcodebert":
        tok_path = os.path.expanduser("~/graphcodebert-base-tok")
        if not os.path.exists(tok_path):
            tok_path = "microsoft/grapcodebert-base"
            
    return tok_path