#!/usr/bin/env python3
import pandas as pd
from mpi4py import MPI
import numpy as np
import adios2
#import os
#import glob
#from multiprocessing import Pool
#import time
import argparse
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import operator
from operator import add
from matplotlib.font_manager import FontProperties
import json

def SetupArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instream", "-i", help="Name of the input stream", required=True)
    parser.add_argument("--config", "-c", help="Name of the config JSON file", default="charts.json")
    parser.add_argument("--nompi", "-nompi", help="ADIOS was installed without MPI", action="store_true")
    args = parser.parse_args()

    return args

def get_num_hosts(attr_info):
    names = {}
    # Iterate over the metadata, and get our hostnames.
    # Build a dictionary of unique values, if the value is
    # already there overwrite it.
    for key in attr_info:
        if "Hostname" in key:
            names[(attr_info[key]['Value'])] = 1
    return len(names)

# Build a dataframe that has per-node data for this timestep of the output data

def build_per_host_dataframe(fr_step, step, num_hosts, variables, columns, filename):
    # Read the number of ranks - check for the new method first
    num_ranks = 1
    if len(fr_step.read('num_ranks')) == 0:
        num_ranks = len(fr_step.read('num_threads'))
    else:
        num_ranks = fr_step.read('num_ranks')[0]

    # Find out how many ranks per node we have
    ranks_per_node = num_ranks / num_hosts
    rows = []
    # For each variable, get each MPI rank's data, some will be bogus (they didn't write it)
    for name in variables:
        rows.append(fr_step.read(name))
    print("Processing dataframe...")
    # Now, transpose the matrix so that the rows are each rank, and the variables are columns
    df = pd.DataFrame(rows).transpose()
    # Add a name for each column
    df.columns = columns
    # Add the MPI rank column (each row is unique)
    df['mpi_rank'] = range(0, len(df))
    # Add the step column, all with the same value
    df['step']=step
    # Filter out the rows that don't have valid data (keep only the lowest rank on each host)
    # This will filter out the bogus data
    df_trimmed = df[df['mpi_rank']%ranks_per_node == 0]
    #print(df_trimmed)
    print("Plotting...")
    df_trimmed[columns].plot(kind='bar', stacked=True)
    imgfile = filename+"_"+"{0:0>5}".format(step)+".svg"
    print("Writing...")
    plt.savefig(imgfile)
    plt.close()
    print("done.")

# Build a dataframe that has per-rank data for this timestep of the output data

def build_per_rank_dataframe(fr_step, step, variables, columns, filename):
    rows = []
    # For each variable, get each MPI rank's data
    for name in variables:
    #    print(name)
    #    print(fr_step.read(name))
        rows.append(fr_step.read(name))
    print("Processing dataframe...")
    # print(rows)
    # Now, transpose the matrix so that the rows are each rank, and the variables are columns
    df = pd.DataFrame(rows).transpose()
    # Add a name for each column
    df.columns = columns
    # Add the MPI rank column (each row is unique)
    df['mpi_rank'] = range(0, len(df))
    # Add the step column, all with the same value
    df['step']=step
    #if (filename == "io_usage"):
    #    print(df)
    print("Plotting...")
    df[columns].plot(logy=True)
    imgfile = filename+"_"+"{0:0>5}".format(step)+".svg"
    print("Writing...")
    plt.savefig(imgfile)
    plt.close()
    print("done.")

# Build a dataframe for the top X timers

def build_topX_timers_dataframe(fr_step, step, num_timers, filename):
    variables = fr_step.available_variables()
    num_threads = fr_step.read('num_threads')[0]
    timer_data = {}
    # Get all timers
    for name in variables:
        if ".TAU application" in name:
            continue
        if "addr=" in name:
            continue
        if "Exclusive TIME" in name:
            shape_str = variables[name]['Shape'].split(',')
            shape = list(map(int,shape_str))
            shortname = name.replace(" / Exclusive TIME", "")
            timer_data[shortname] = []
            temp_vals = fr_step.read(name)
            timer_data[shortname].append(temp_vals[0])
            index = num_threads
            while index < shape[0]:
                timer_data[shortname].append(temp_vals[index])
                index += num_threads
    df = pd.DataFrame(timer_data)
    # Get the mean of each column 
    mean_series = df.mean()
    # Get top X timers
    sorted_series = mean_series.sort_values(ascending=False)
    topX_series = sorted_series[:num_timers].axes[0].tolist()
    # Plot the DataFrame
    print("Plotting...")
    df[topX_series].plot(kind='bar', stacked=True)
    imgfile = filename+"_"+"{0:0>5}".format(step)+".svg"
    print("Writing...")
    plt.savefig(imgfile)
    plt.close()
    print("done.")

    
# Process the ADIOS2 file

def process_file(args):
    config_data = open(args.config)
    config = json.load(config_data)
    print(config)
    filename = args.instream
    print ("Opening:", filename)
    if not args.nompi:
        fr = adios2.open(filename, "r", MPI.COMM_SELF, "adios2.xml", "TAUProfileOutput")
    else:
        fr = adios2.open(filename, "r", "adios2.xml", "TAUProfileOutput")
    # Get the attributes (simple name/value pairs)
    attr_info = fr.available_attributes()
    # Get the unique host names from the attributes
    num_hosts = get_num_hosts(attr_info)
    cur_step = 0
    # Iterate over the steps
    for fr_step in fr:
        # track current step
        cur_step = fr_step.current_step()
        print(filename, "Step = ", cur_step)
        for f in config["figures"]:
            print(f["name"])
            if "Timer" in f["name"]:
                build_topX_timers_dataframe(fr_step, cur_step, int(f["granularity"]), f["filename"])
            elif f["granularity"] == "node":
                build_per_host_dataframe(fr_step, cur_step, num_hosts, f["components"], f["labels"], f["filename"])
            else:
                build_per_rank_dataframe(fr_step, cur_step, f["components"], f["labels"], f["filename"])


if __name__ == '__main__':
    args = SetupArgs()
    print(args)
    process_file(args)

