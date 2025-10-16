import os, json
from pathlib import Path
import torch
from dataset import SpeechCommands        
from preprocessing import MFCCPreprocessor    


def main():
    out_root = Path("features_mfcc")  
    out_root.mkdir(parents=True, exist_ok=True)

    pre = MFCCPreprocessor(sample_rate=16000, n_mfcc=40, n_mels=64,fixed_frames=100)
    ds = SpeechCommands(root="./data", preprocessor=pre)


    classes_path = out_root / "classes.json"
    classes_path.write_text(json.dumps(ds.classes, indent=2))
    print(f"writing class list to {classes_path.resolve()}")

    for classname in ds.classes:
        (out_root / classname).mkdir(parents=True, exist_ok=True)

    for i in range(len(ds)):
        mfcc, y = ds[i]                               
        classname = ds.classes[y]

        rel = ds.base._walker[ds.indices[i]]            
        stem = Path(rel).stem # stem as i want file name without extension                       
        out_file = out_root / classname / f"{stem}.pt"

        torch.save({"mfcc": mfcc, "label": int(y)}, out_file)

        if (i + 1) % 1000 == 0:
            print(f"Saved {i+1}/{len(ds)}")

    print("Done!")
    


if __name__ == "__main__":
    main()

