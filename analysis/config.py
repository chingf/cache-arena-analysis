import os
import pickle
import numpy as np
from pathlib import Path

# Data configuration variables
pickle_dir = Path('/home/chingf/Code/cache-arena-analysis/pickles')
engram_dir = Path('/home/chingf/engram/Emily')
neural_data_dir = engram_dir / 'NeuralData/Gcamp'
behav_data_dir = engram_dir / 'BehavioralData'
birds = ['RBY45', 'LMN73']

# Arena configuration variables
with open(pickle_dir / "arena_params.p", "rb") as f:
    arena_params = pickle.load(f)

# Collects filepaths to data
auto_collect = True # Directory structure is not yet consistent for auto-collect
h5_path_dict = {}
h5_path_dict['RBY45'] = [
    f for f in (neural_data_dir / 'ForChing').iterdir() if f.suffix == '.mat'
    ]
if auto_collect:
    for bird in ['LMN73']:
        h5_path_dict[bird] = []
        bird_dir = neural_data_dir / bird
        for date_dir in [d for d in bird_dir.iterdir() if d.is_dir()]:
            for processing_dir in [d for d in date_dir.iterdir() if d.is_dir()]:
                results_dir = processing_dir / 'Results'
                if not results_dir.exists(): continue
                if not np.any(["DLC" in f.name for f in results_dir.iterdir()]):
                    continue
                extracted_files = [
                    f for f in results_dir.iterdir() if f.stem.startswith(
                        "ExtractedWith"
                        )
                    ]
                if len(extracted_files) == 0: continue
                extracted_files.sort(key=os.path.getmtime, reverse=True)
                h5_path_dict[bird].append(extracted_files[0])
else:
    lmn_dir = neural_data_dir / 'LMN73'
    h5_path_dict['LMN73'] = [
        lmn_dir / '20200908/2020-09-08-12-02-54_video_SplitIntoBatches_29_250_10_200/Results/ExtractedWithXY_Cleaned182657_09182020.mat',
        lmn_dir / '20200910/2020-09-10-12-00-48_video_SplitIntoBatches_29_250_10_200/Results/ExtractedWithXY_Cleaned181248_09182020.mat',
        lmn_dir / '20200918/2020-09-18-11-50-00_video_SplitIntoBatches_29_250_10_200/Results/ExtractedWithDLCAndAnnotations153159_10222020.mat'
        ]

