import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import os
import os.path as osp
import random
import numpy as np
import pandas as pd
import time
import torch
from tqdm import tqdm
import warnings
import logging
import time

from flashvggt.models.flash_vggt import FlashVGGT

# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"

# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# # Set computation precision
# torch.set_float32_matmul_precision('highest')
# torch.backends.cudnn.allow_tf32 = False

@hydra.main(version_base=None, config_path="./eval_configs", config_name="depth.yaml")
def eval(cfg: DictConfig):
    seed_everything(cfg.seed)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # create logging directory
    os.makedirs(cfg.log_dir, exist_ok=True)

    columns = [
        "Name", # This should be empty
        "Method", "Dataset", "Frames", 
        "Depth-Rel.", "Depth-τ", 
        "Point-Acc.", "Point-Comp.", "Point-CD", "Point-NC",
        "Cam-APE", "Cam-ARE", "Cam-RPE-Trans", "Cam-RPE-Rot",
        "Time (s)", "Mem. (GB)"
    ]

    eval_df = pd.DataFrame(columns=columns)
    # if not osp.exists(cfg.save_path):
    #     eval_df = pd.DataFrame(columns=columns)
    # else:
    #     eval_df = pd.read_csv(cfg.save_path)
    #     assert set(columns) == set(eval_df.columns), "Columns do not match"

    # set up models
    model = FlashVGGT(kv_downfactor=cfg.kv_downfactor)
    model.load_ckpt(cfg.ckpt_path)

    model.eval()
    model.to(device)

    # set up metric
    metric = instantiate(cfg.metric)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # set up data
    datasets = [instantiate(dataset, **cfg.data.common, num_frames=cfg.num_frames) for dataset in cfg.data.datasets]
    for dataset in datasets:
        results = {}
        desc = f"[{dataset.name}] [{cfg.num_frames} Frames]".ljust(25)

        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=desc):
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    images = data["images"].to(device)
                    start_event.record()
                    model_preds = model(images)
                    end_event.record()

            torch.cuda.synchronize()
            infer_time = start_event.elapsed_time(end_event) / 1000.0
            
            for key in data:
                data[key] = data[key].to("cpu")
            for key in model_preds:
                if isinstance(model_preds[key], torch.Tensor):
                    model_preds[key] = model_preds[key].to("cpu")
            results = dict_add(results, metric(model_preds, data))
            results = dict_add(results, {"Time (s)": infer_time})

            del model_preds, data, images
            torch.cuda.empty_cache()

        results = dict_div(results, len(dataset))
        max_memory = torch.cuda.max_memory_allocated(0)/1024/1024/1024
        torch.cuda.reset_peak_memory_stats()

        eval_df.loc[len(eval_df)] = ['', cfg.model, dataset.name, cfg.num_frames, results["Depth-Rel."], results["Depth-τ"], results["Point-Acc."], results["Point-Comp."], results["Point-CD"], results["Point-NC"], results["Cam-APE"], results["Cam-ARE"], results["Cam-RPE-Trans"], results["Cam-RPE-Rot"], results["Time (s)"], max_memory]

    if osp.exists(osp.join(cfg.log_dir, f"eval/{cfg.save_name}.csv")):
        eval_df_prev = pd.read_csv(osp.join(cfg.log_dir, f"eval/{cfg.save_name}.csv"))
        assert set(eval_df.columns) == set(eval_df_prev.columns), "Columns do not match"
        eval_df = pd.concat([eval_df_prev, eval_df])

    # sort by dataset then model
    eval_df = eval_df.sort_values(by=["Dataset", "Frames", "Method"])
    # round to 4 decimal places
    eval_df = eval_df.round(6)

    # columns_except_name_and_dataset = [col for col in eval_df.columns if col not in ["Name", "Dataset"]]
    # mean_df = eval_df[columns_except_name_and_dataset]
    # mean_df = mean_df.groupby(["Method", "Frames"]).mean().reset_index()
    # mean_df = mean_df.round(4)
    # mean_df["Name"] = ""
    # mean_df["Frames"] = "Mean"
    # mean_df = mean_df[eval_df.columns]

    if not osp.exists(osp.join(cfg.log_dir, f"eval")):
        os.makedirs(osp.join(cfg.log_dir, f"eval"), exist_ok=True)
    eval_df.to_csv(osp.join(cfg.log_dir, f"eval/{cfg.save_name}.csv"), index=False)
    # mean_df.to_csv(cfg.save_path.replace(".csv", "_mean.csv"), index=False)

def dict_add(dict1, dict2):
    for key in dict2:
        if key not in dict1:
            dict1[key] = dict2[key]
        else:
            dict1[key] += dict2[key]
    return dict1

def dict_div(dict, count):
    assert count > 0
    for key in dict:
        dict[key] /= count
    return dict

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    eval()