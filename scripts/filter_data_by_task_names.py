#!/usr/bin/env python

import os
import json
import shutil

root_dir = "/home/group_25b505/group_6/workspace/user_00085_25b505/projects/Isaac-GR00T/demo_data"


def create_data(
    data_name: str,
    loc_names: list,
):
    """Creates a dataset filtered by location names."""

    print(f"Target data: {data_name}")
    print(f"Target Tasks: {loc_names}")

    # set data name and directory to be used
    new_data_name = "_".join([
        data_name,
        "-".join(loc_names),
    ])
    new_data_dir = os.path.join(root_dir, new_data_name)

    os.makedirs(
        new_data_dir,
        exist_ok=True,
    )

    # create symbolic links for data and videos
    for subdir in ("data", "videos"):
        if not os.path.exists(os.path.join(root_dir, new_data_name, subdir)):
            os.symlink(
                os.path.join(root_dir, data_name, subdir),
                os.path.join(root_dir, new_data_name, subdir),
            )

    # copy meta recursively
    shutil.copytree(
        os.path.join(root_dir, data_name, "meta"),
        os.path.join(root_dir, new_data_name, "meta"),
        dirs_exist_ok=True,
    )

    # load original meta/episodes.jsonl
    json_data_path = os.path.join(root_dir, new_data_name, "meta/episodes.jsonl")
    new_json_data = []

    num_orig_eps = len(open(json_data_path).readlines())

    # extract episodes of specified location names
    for ep in open(json_data_path).readlines():
        ep = json.loads(ep)

        if any(n in ep["tasks"][0].lower() for n in map(str.lower, loc_names)):
            new_json_data += [ep]

    # save filtered episodes 
    with open(json_data_path, "w") as f:
        f.writelines([
            json.dumps(ep) + "\n"
            for ep in new_json_data
        ])

    num_created_eps = len(new_json_data)
    print(f"The filtered data was saved to {new_data_dir}")
    print(f"{num_created_eps / num_orig_eps * 100:.2f}% ({num_created_eps} / {num_orig_eps}) episodes were used.")
    print()


# 2025-05-v3.0-success-only
data_name = "2025-05-v3.0-success-only"
loc_names_list = [
    ["the oven"],
    ["bottle"],
    ["the oven", "bottle"],
]
# loc_names_list = [
#     ["pick", "place", "put", "grab", "grasp"],
#     ["open" , "close"],
#     ["pick", "place", "put", "grab", "grasp","open", "close"],
# ]

for loc_names in loc_names_list:
    create_data(
        data_name,
        loc_names,
    )

# 2025-06-v3.0-success-only
data_name = "2025-06-v3.0-success-only"
loc_names_list = [
    ["the oven"],
    ["bottle"],
    ["the oven", "bottle"],
]
# loc_names_list = [
#     ["pick", "place", "put", "grab", "grasp"],
#     ["open" , "close"],
#     ["pick", "place", "put", "grab", "grasp","open", "close"],
# ]

for loc_names in loc_names_list:
    create_data(
        data_name,
        loc_names,
    )

# 2025-07-v3.0-success-only
data_name = "2025-07-v3.0-success-only"
loc_names_list = [
    ["the oven"],
    ["bottle"],
    ["the oven", "bottle"],
]
# loc_names_list = [
#     ["pick", "place", "put", "grab", "grasp"],
#     ["open" , "close"],
#     ["pick", "place", "put", "grab", "grasp","open", "close"],
# ]

for loc_names in loc_names_list:
    create_data(
        data_name,
        loc_names,
    )

# 2025-08-v3.0-success-only
# data_name = "2025-08-v3.0-success-only"
# loc_names_list = [
#     ["aist"],
# ]

# for loc_names in loc_names_list:
#     create_data(
#         data_name,
#         loc_names,
#     )


