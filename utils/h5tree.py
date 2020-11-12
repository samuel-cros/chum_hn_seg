#!/usr/bin/env python

import os
import h5py as h5
import numpy as np
from pathlib import Path
from treelib import Node, Tree
from argparse import ArgumentParser

def attrs(group):
    a = {}
    for k, v in group.attrs.items():
        try:
            v = v.round(2)
        except (TypeError, AttributeError):
            pass
        if isinstance(v, np.ndarray):
            v = list(v)
        a[k] = v
    if a:
        return str(a).replace("'", "")
    return ""

def h5tree(path, with_attr=False):
    tree = Tree()
    def h5t(value, group, parent="root"):
        if type(group) is h5.Group:
            name = value
            if with_attr:
                name += " "+attrs(group)
            tree.create_node(name, value, parent=parent)
            for v, g in group.items():
                h5t(v, g, parent=value)
        if type(group) is h5.Dataset:
            desc = " ".join([str(group.shape).replace(",", ""), group.dtype.name])
            if with_attr:
                desc += " "+attrs(group)
            tree.create_node(value+": "+desc, value, parent=parent)
            
    with h5.File(path, "r") as group:
        name = path.stem
        if with_attr:
            name += ": "+attrs(group)
        tree.create_node(name, "root")
        for v, g in group.items():
            h5t(v, g, "root")
    return tree

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

if __name__ == "__main__":
    parser = ArgumentParser("View the contents of an hdf5 file")
    parser.add_argument("file", type=Path)
    parser.add_argument("--attrs", "-a", action="store_true")
    args = parser.parse_args()
    print(h5tree(args.file, args.attrs))
    print(file_size(args.file))
