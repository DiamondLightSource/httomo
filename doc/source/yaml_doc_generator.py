#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
"""Script that appends yaml information to documentation pages"""

import os

def add_function_summary(dirs, root, files):
    rst_name = root.split("/")[-1]
    doc_rst_file = f"{doc_source_dir}/method_api/{rst_name}.rst"
    with open(doc_rst_file, "a") as edit_doc:
        if os.stat(doc_rst_file).st_size == 0:
            add_title(edit_doc, rst_name)
        else:
            edit_doc.write(f"\n\n   .. rubric:: **Functions**")
            edit_doc.write(f"\n\n   .. autosummary::\n")
            for fi in files:
                yml_title = fi.split(".yaml")[0]
                edit_doc.write(f"\n       {yml_title}")


def create_module_doc(doc_source_dir, root, files):
    rst_name = root.split("/")[-1]
    doc_rst_file = f"{doc_source_dir}/method_api/{rst_name}.rst"
    for fi in files:
        t_name = root.split("source")[-1]
        t_name = f"{t_name}/{fi}"
        with open(doc_rst_file, "a") as edit_doc:
            edit_doc.write(f"\n\n.. dropdown:: {fi}")
            edit_doc.write(f"\n\n    :download:`Download <{t_name}>`")
            edit_doc.write(f"\n\n    .. literalinclude:: {t_name}")


def add_title(edit_doc, rst_name):
    """Add a title to rst file"""
    title = f":mod:`{rst_name}`"
    edit_doc.write(f"{title}\n")
    underline = len(title) * "="
    edit_doc.write(f"{underline}\n")


if __name__ == "__main__":
    # Create a file for each tomopy
    doc_source_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_templates = doc_source_dir + '/../../templates/'
    for root, dirs, files in os.walk(path_to_templates, topdown=True):
        dirs[:] = [d for d in dirs]
        files[:] = [fi for fi in files]
        if files:
            add_function_summary(doc_source_dir, root, files)
            create_module_doc(doc_source_dir, root, files)
