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

import os


def add_function_summary(doc_dir, root, files):
    """Append title and function summary to documentation file.

    Parameters
    ----------
    doc_dir : Path
        Documentation path
    root : Path
        Path to the yaml template directory.
    files : List
        List of yaml function templates for each module.
    """
    rst_name = root.split("/")[-1]
    doc_rst_file = f"{doc_dir}/api/{rst_name}.rst"
    with open(doc_rst_file, "a") as edit_doc:
        if os.stat(doc_rst_file).st_size == 0:
            rst_name = root.split("/")[-1]
            add_title(edit_doc, rst_name)
            add_backend_link(edit_doc, rst_name)
        else:
            edit_doc.write(f"\n\n   .. rubric:: **Functions**")
            edit_doc.write(f"\n\n   .. autosummary::\n")
            for fi in files:
                yml_title = fi.split(".yaml")[0]
                edit_doc.write(f"\n       {yml_title}")


def create_yaml_dropdown(doc_dir, root, files):
    """Create dropdown panels to allow yaml functions to be downloaded.

    Parameters
    ----------
    doc_dir : Path
        Documentation path
    root : Path
        Path to the yaml template directory.
    files : List
        List of functions for each module.
    """
    rst_name = root.split("/")[-1]
    doc_rst_file = f"{doc_dir}/api/{rst_name}.rst"
    template_dir = root.split("source")[-1]
    download_all_button(template_dir, doc_rst_file)

    for fi in files:
        t_name = f"{template_dir}/{fi}"
        f_name = fi.split(".yaml")[0]
        url = (
            f"https://tomopy.readthedocs.io/en/stable/api/{rst_name}"
            f".html#{rst_name}.{f_name}"
        )
        with open(doc_rst_file, "a") as edit_doc:
            edit_doc.write(f"\n\n.. dropdown:: {fi}")
            edit_doc.write(f"\n\n    :download:`Download <{t_name}>`\n\n")
            if "tomopy" in t_name:
                edit_doc.write(
                    f"    |link_icon| `Link to {f_name}"
                    f" function description <{url}>`_"
                )
            edit_doc.write(f"\n\n    .. literalinclude:: {t_name}")


def download_all_button(template_dir, doc_rst_file):
    """Download all yaml function text as one file.

    Parameters
    ----------
    template_dir : str
        Directory to template files.
    doc_rst_file : File
        rst file to write to.
    """
    download_all_path = f"{template_dir}/download_all.yaml"
    # Only include download all when more than one yaml file exists
    if len(files) > 1:
        with open(doc_rst_file, "a") as edit_doc:
            download_str = "\n\n:download:`Download all yaml templates<"
            edit_doc.write(f"{download_str}{download_all_path}>`")


def add_title(edit_doc, rst_name):
    """Add a title to rst file.
    Parameters
    ----------
    edit_doc : File
        Document to write to.
    rst_name : str
        name of rst file.
    """
    edit_doc.write(f".. |link_icon| unicode:: U+1F517\n\n")
    title = f":mod:`{rst_name}`"
    edit_doc.write(f"{title}\n")
    underline = len(title) * "="
    edit_doc.write(f"{underline}\n")


def add_backend_link(edit_doc, rst_name):
    """Links to backends documentation.

    Parameters
    ----------
    edit_doc : File
        Document to write to.
    rst_name : str
        name of rst file.
    """
    if "tomopy" in rst_name:
        # If it is a tomopy module, insert a link.
        url = "https://tomopy.readthedocs.io/en/stable/api/"
    elif "httomolibgpu" in rst_name:
        url = "https://diamondlightsource.github.io/httomolibgpu/api/"
    elif "httomolib" in rst_name:
        url = "https://diamondlightsource.github.io/httomolib/api/"
    else:
        raise ValueError("The name of the backend package is not recognised")
    edit_doc.write(f"\n{url}{rst_name}.html\n\n")


def save_all_yaml_functions(tmp_dir, yaml_files):
    """Create a file including all yaml function definitions.

    Parameters
    ----------
    tmp_dir : Path
        Path to the yaml template directory.
    yaml_files : list
        List of individual yaml template files.
    """
    file_path = f"{tmp_dir}/download_all.yaml"
    # Only merge files when more than one yaml file exists
    if len(yaml_files) > 1:
        with open(file_path, "w") as outfile:
            for f in yaml_files:
                with open(f"{tmp_dir}/{f}") as infile:
                    outfile.write(infile.read())


if __name__ == "__main__":
    """Create documentation for modules from httomo, tomopy, httomolib and httomolibgpu.
    Append the yaml information to the documentation pages.
    """
    doc_source_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_templates = doc_source_dir + "/../build/yaml_templates/"
    for root, dirs, files in os.walk(path_to_templates, topdown=True):
        dirs[:] = [d for d in dirs]
        files[:] = [fi for fi in files if ".yaml" in fi]
        if files:
            add_function_summary(doc_source_dir, root, files)
            save_all_yaml_functions(root, files)
            create_yaml_dropdown(doc_source_dir, root, files)
