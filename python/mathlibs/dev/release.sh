#!/usr/bin/env bash
#
# Copyright 2016 The BigDL Authors.
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
#

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR
BIGDL_DIR="$(cd ${RUN_SCRIPT_DIR}/../../..; pwd)"
echo $BIGDL_DIR
BIGDL_PYTHON_DIR="$(cd ${BIGDL_DIR}/python/mathlibs; pwd)"
echo $BIGDL_PYTHON_DIR


if (( $# < 4)); then
  echo "Usage: release.sh platform version quick_build upload mvn_parameters"
  echo "Usage example: bash release.sh linux default false true"
  echo "Usage example: bash release.sh linux 0.14.0.dev1 true true"
  echo "If needed, you can also add other profiles such as: -Dspark.version=2.4.6 -P spark_2.x"
  exit -1
fi

platform=$1
version=$2
quick=$3  # Whether to rebuild the jar; quick=true means not rebuilding the jar
upload=$4  # Whether to upload the whl to pypi
profiles=${*:5}

if [ "${version}" != "default" ]; then
    echo "User specified version: ${version}"
    echo $version > $BIGDL_DIR/python/version.txt
fi

bigdl_version=$(cat $BIGDL_DIR/python/version.txt | head -1)
echo "The effective version is: ${bigdl_version}"

if [ "$platform" ==  "mac" ]; then
    echo "Building bigdl for mac system"
    dist_profile="-P mac $profiles"
    verbose_pname="macosx_10_11_x86_64"
elif [ "$platform" == "linux" ]; then
    echo "Building bigdl for linux system"
    dist_profile="-P linux $profiles"
    verbose_pname="manylinux2010_x86_64"
else
    echo "Unsupported platform"
fi

cd $BIGDL_PYTHON_DIR
wheel_command="python setup.py bdist_wheel --plat-name ${verbose_pname}"
echo "Packing python distribution:   $wheel_command"
${wheel_command}
cd -

if [ ${upload} == true ]; then
    upload_command="twine upload python/mathlibs/dist/bigdl_math-${bigdl_version}-py3-none-${verbose_pname}.whl"
    echo "Please manually upload with this command:  $upload_command"
    $upload_command
fi
