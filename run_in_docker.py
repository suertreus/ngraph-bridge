#!/usr/bin/env python3
# ==============================================================================
#  Copyright 2018-2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import errno
import os
from subprocess import check_output, call
import sys
import shutil
import glob
import platform

from tools.build_utils import *
import argparse
from tools.test_utils import *
from tools.build_utils import download_repo


def main():
    '''
    Tests nGraph-TensorFlow Python 3. This script needs to be run after 
    running build_ngtf.py which builds the ngraph-tensorflow-bridge
    and installs it to a virtual environment that would be used by this script.
    '''
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    arguments = parser.parse_args()

    #-------------------------------
    # Recipe
    #-------------------------------

    root_pwd = os.getcwd()

    if os.getenv("IN_DOCKER") == None:
        if check_container() == True:
            stop_container()
        start_container("/ngtf")
        run_in_docker("/ngtf/run_in_docker.py", arguments)
        return

    # Constants
    build_dir = 'build_cmake'
    venv_dir = 'build_cmake/venv-tf-py3'
    tf_src_dir = 'build_cmake/tensorflow'

    os.environ['NGRAPH_TF_BACKEND'] = 'CPU'

    load_venv(venv_dir)

    # Finally run Resnet50 based training and inferences
    run_resnet50(build_dir)

    os.chdir(root_pwd)


if __name__ == '__main__':
    main()
