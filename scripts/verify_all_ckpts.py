# check if all checkpoints are present
import os
import json

# main
if __name__ == "__main__":
    ckpt_info = json.load(open("./checkpoint_info.json"))
    ctr = 0
    tot = len(ckpt_info)
    for path, value in ckpt_info:
        if not value: continue
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path):
            print(f"\x1b[32;1mcheckpoint exists for {path}\x1b[0m")
        else: 
            ctr += 1
            print(f"\x1b[31;1mcheckpoint missing for {path}\x1b[0m")
    print(f"missing {ctr}/{tot} checkpoints!")