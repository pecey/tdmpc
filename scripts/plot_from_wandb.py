import pandas as pd
import wandb
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

# color_mapping = {"Single": "#1f77b4", 
#                  "Separate": "#2ca02c"}

color_mapping = {"PV": "#1f77b4", 
                 "P": "#ff0318",
                 "V": "#ff7f03",
                 "None": "#2ca02c",
                 "AR": "#1f77b4",
                 "TEA": "#9467bd", 
                 "TEA (d=10)": "#8f561d"
                 }

column_mapping = {"episode_reward": "Score", 
                  "episode": "Episode",
                  "idx": "Training iterations"}

def running_average(data, n):
    weights = [1/n]*n
    moving_average = np.convolve(data, weights, "valid")
    spare = len(data) - len(moving_average)
    spare_mean = [data[i:].mean() for i in range(-spare, 0, 1)]
    return np.concatenate([moving_average, spare_mean])

def plot_grouped_run_with_smoothing(results_path, data, col):
    # n = smoothing window
    for n in [5, 10]:
        if ADD_LEGEND:
            path = f"{results_path}/grouped/{col}_grouped_smooth_{n}_legend.pdf"
        else:
            path = f"{results_path}/grouped/{col}_grouped_smooth_{n}.pdf"
        plot_grouped_runs(data, True, col, path, n)

def plot_grouped_runs_without_smoothing(results_path, data, col):
    path = f"{results_path}/grouped/{col}_grouped.pdf"
    plot_grouped_runs(data, False, col, path, n = None)

def plot_grouped_runs(data, smoothing, col, path, n):
    figure = plt.figure()
    print(f"Plotting {data.keys()} against baseline. Col : {col}")
    def _plot_group(d, label, x = None, color=None):
        seeds = list(d.keys())
        x_len = np.min([len(d[s][col]) for s in seeds])
        x = np.array(d[seeds[0]]['episode'][:x_len])

        # First smooth all runs individually
        if smoothing:
            y = np.array([running_average(np.array(d[s][col][:x_len]), n=n) for s in seeds])
        else:
            y = np.array([d[s][col][:x_len] for s in seeds])

        print(f"{label} has {len(seeds)} data points : {seeds}. X: {x.shape}, Y: {y.shape}")
        
        # For log scale
        # y = np.log10(y)

        # Compute mean and standard deviation of smoothed runs
        y_mean = np.mean(y, axis=0) 
        y_std = np.std(y, axis=0)

        if color is not None:
            plt.plot(x, y_mean, label=label, linewidth=2.5, color=color)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
        else:
            plt.plot(x, y_mean, label=label, linewidth=2.5)
            plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
        return x, y_mean[-1]
    
    for k, group in data.items():
        x, y_mean = _plot_group(group, k, color = color_mapping[k])
       
    # plt.setp(axes.get_xticklabels(), visible=False)
    # plt.setp(axes.get_xticklabels()[::5], visible=True)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=6)
    
    # # For adding extra tick
    # if col == "mean_rollout_length":
    #     print(plt.yticks()[0])
    #     # y_ticks = list(plt.yticks()[0])
    #     # y_ticks = sorted(y_ticks + [150])
    #     # print(y_ticks)
    #     # plt.yticks(y_ticks)
    
    plt.autoscale(axis='x', tight="True")
    plt.xlabel(column_mapping["episode"])
    plt.ylabel(column_mapping[col])
    # plt.ylabel(r"$\log_{10}\text{(action repeats)}$")
    # plt.xlim(right=XLIM)
    # plt.ylim(bottom=1, top=800)
    plt.grid(True, alpha=0.6, linestyle='--', color='lightgrey')
    plt.tight_layout()
    if ADD_LEGEND:
        plt.legend()

    plt.title(title)
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_individual_runs(data, smoothing, col, path):
    figure = plt.figure()
    
    def _plot_col(d, label):
        y = np.array(d[col])
        x = d['episode']
        if smoothing:
            y = running_average(y, n=5)
        else:
            y = y
        plt.plot(x, y, label=f"seed {label}")
    
    for k, group in data.items():
        _plot_col(group, k)
    plt.grid("on")
    plt.autoscale(axis='x', tight="True")
    plt.xlabel(column_mapping["episode"])
    plt.ylabel(column_mapping[col])
    plt.tight_layout()
    plt.legend()
    plt.title(title)
    plt.savefig(path, format='pdf', bbox_inches='tight')
    plt.close()
    

def get_data_from_run(api, project, run_name, use_cached=False):
    raw_data_path = f"{results_path}/raw_data/{project}_{run_name}.json"
    if os.path.isfile(raw_data_path) and use_cached:
        with open(raw_data_path) as f:
            data = json.load(f)
    else:
        runs = api.runs(f"pecey/{project}", filters={"displayName": run_name})
        data = {}
        for run in runs:
            try:
                seed = run.config["seed"]
                step = run.history()['eval/step'].to_list()
                episode_reward = run.history()['eval/episode_reward'].to_list()
                episode = run.history()['eval/episode'].to_list()
            
                data[seed] = {"step": step,
                              "episode_reward": episode_reward,
                              "episode": episode
                              }
            except Exception as e:
                print(f"Failed for {run}. Error: {e}")
            
        with open(raw_data_path, 'w') as f:
            json.dump(data, f)
    return data

def get_data_from_group(api, project, group_name, use_cached=False):

    def remove_na(lst):
        return [x for x in lst if str(x) != 'nan']

    raw_data_path = f"{results_path}/raw_data/{project}_{group_name}.json"
    if os.path.isfile(raw_data_path) and use_cached:
        with open(raw_data_path) as f:
            data = json.load(f)
    else:
        runs = api.runs(f"pecey/{project}", filters={"group": group_name})
        data = {}
        for run in runs:
            try:
                seed = run.config["seed"]
                step = remove_na(run.history()['eval/step'].to_list())
                episode_reward = remove_na(run.history()['eval/episode_reward'].to_list())
                episode = remove_na(run.history()['eval/episode'].to_list())
       
                print(f"Run {run.name} Seed {seed} has {len(episode_reward)} episode rewards and {len(episode)} episodes.")
                
                data[seed] = {"step": step,
                              "episode_reward": episode_reward,
                              "episode": episode
                              }
            except Exception as e:
                print(f"Failed for {run}. Error: {e}")
            
        with open(raw_data_path, 'w') as f:
            json.dump(data, f)
    return data

api = wandb.Api()
entity = "pecey"

project = "tdmpc-hopper-hop"
groups = {"AR": "hopper-hop-state-using-ac-repeats-without-policy-and-value-1768876764",
          "TEA": "hopper-hop-state-using-tea-without-policy-and-value-1768940730",
          "TEA (d=10)" : "hopper-hop-state-using-tea-without-policy-and-value-d-10-1769024551",
          }
results_dir = "hopper-hop-using-teas-wo-policy-and-value-function-1768876764"
title="Hopper Hop"

DATA_FROM_GROUP = True
ADD_LEGEND = False
force_download = False

results_path = f"/N/u/palchatt/BigRed200/tdmpc/results/{project}/{results_dir}"
os.makedirs(f"{results_path}/individual", exist_ok=True)
os.makedirs(f"{results_path}/grouped", exist_ok=True)
os.makedirs(f"{results_path}/data", exist_ok=True)
os.makedirs(f"{results_path}/raw_data", exist_ok=True)
os.makedirs(f"{results_path}/models", exist_ok=True)

# data = {k: get_data_from_group(api, project, v) for k,v in groups.items()}
# baseline_data = get_data_from_group(api, project.replace("_long_horizon",""), baseline_group)
# baseline_data = get_data_from_group(api, project, baseline_group)

def get_data(api, project, wandb_data_identifiers, path, DATA_FROM_GROUP, force_download):
    """
    wandb_data_identifiers: dict of label and run name or group name.
    """
    get_data_fn = get_data_from_group if DATA_FROM_GROUP else get_data_from_run

    if os.path.exists(path) and not force_download:
        print(f"Data exists locally. Reading from {path}")
        with open(path) as f:
            data = json.load(f)
    else:
        print("Getting data from WandB server.")
        if type(wandb_data_identifiers) == dict:
            data = {k: get_data_fn(api, project, v) for k,v in wandb_data_identifiers.items()}
        else:
            data = get_data_fn(api, project, wandb_data_identifiers)
        with open(path, 'w') as f:
            json.dump(data, f)
    return data

data = get_data(api, project, groups, f"{results_path}/data/data.json", 
                DATA_FROM_GROUP, 
                force_download)

cols = ["episode_reward"]
for col in cols:
    plot_grouped_run_with_smoothing(results_path, data, col)    
    # plot_grouped_runs_without_smoothing(results_path, data, baseline_data, col)   

print(f"Graphs in {results_path}")