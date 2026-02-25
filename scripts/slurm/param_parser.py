"""param_parser.py.

This script is used to parse a JSON configuration file that defines a list of parameters for a Slurm array job. The configuration file should have the following format: 
{
    "separator": "=",  # Optional, default is "="
    "entries": [
        {
            "keys": ["x", "y"],
            "values": [
                ["a1", "a2", "a3"],
                ["b1", "b2", "b3"]
            ]
        },
        {
            "keys": ["z"],
            "values": [
                ["c1", "c2"]
            ]
        }  
    ]
}
This will generate the following parameter combinations:
(
 "x=a1 y=b1 z=c1",
 "x=a1 y=b1 z=c2",
 "x=a2 y=b2 z=c1",
 "x=a2 y=b2 z=c2",
 "x=a3 y=b3 z=c1",
 "x=a3 y=b3 z=c2",
)
Those under the same entry are joined together, and the different entries are combined with a Cartesian product. The script can be used to either print the total number of parameter combinations or to print a specific combination based on a task ID.

Arguments:
- config_path: The path to the JSON configuration file.
- --task_id (-i): The index of the parameter combination to print (0-based).
- --size (-s): If set, the script will print the total number of parameter combinations instead of a specific combination.
- --prefix (-p): An optional prefix to add to each argument (default is an empty string).
"""


import itertools as it
import argparse
import json
from pathlib import Path
from typing import List


def combine_joined_parameters(
    keys: List[str], values: List[List[str]], arg_prefix: str = "", sep: str = "="
) -> List[str]:
    joined_params = []
    for i_arg in range(len(values[0])):
        params: List[str] = []
        for key, val_list in zip(keys, values):
            params.append(f"{arg_prefix}{key}{sep}{str(val_list[i_arg])}")
        joined_params.append(" ".join(params))
    return joined_params


def parse_param_list(config: dict, arg_prefix: str = "") -> List[str]:
    sep = config.get("separator", "=")
    all_params = []
    for entry in config["entries"]:
        keys = entry["keys"]
        values = entry["values"]

        if len(keys) != len(values):
            raise ValueError("The number of keys and values in an entry must be the same")
        if len(values) == 0:
            raise ValueError("An entry must have at least one value")
        if not all(len(v) == len(values[0]) for v in values):
            raise ValueError("All value lists in an entry must have the same length")

        joined = combine_joined_parameters(keys, values, arg_prefix, sep)
        all_params.append(joined)

    combined = list(it.product(*all_params))
    return [" ".join(c) for c in combined]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("-i", "--task_id", type=int, required=False)
    parser.add_argument("-s", "--size", action="store_true")
    parser.add_argument("-p", "--prefix", default="")
    parser.add_argument("-l", "--list", action="store_true", help="Print all parameter combinations")
    parser.add_argument("--separator", default=None, help="The separator to use between keys and values (default is '=')")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    if args.separator is not None:
        config["separator"] = args.separator

    param_list = parse_param_list(config, args.prefix)

    if args.size:
        print(len(param_list))
    elif args.task_id is not None:
        if args.task_id < len(param_list):
            print(param_list[args.task_id])
    elif args.list:
        for params in param_list:
            print(params)
    else:
        raise ValueError("Pass either --task_id (-i), --size (-s), or --list (-l)")

if __name__ == "__main__":
    main()
