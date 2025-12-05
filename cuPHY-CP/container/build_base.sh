#!/bin/bash -e
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

IMAGE_NAME=aerial_base


# Switch to SCRIPT_DIR directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
echo $SCRIPT starting...
cd $SCRIPT_DIR
source ./setup.sh

if [[ "$AERIAL_VERSION_TAG" == "$(unset AERIAL_VERSION_TAG && source ./setup.sh && echo $AERIAL_VERSION_TAG)" ]]
then
    echo "Error: Do not run this script without setting the AERIAL_VERSION_TAG variable"
    exit 1
fi

TARGETARCH=$(basename $AERIAL_PLATFORM)
case "$TARGETARCH" in
    "amd64")
        CPU_TARGET=x86_64
        ;;
    "arm64")
        CPU_TARGET=aarch64
        ;;
    *)
        echo "Unsupported target architecture"
        exit 1
        ;;
esac


NGC_ORG_TEAM=${NGC_ORG_TEAM:-nvidia/team/aerial}
DOCA_VERSION="3.0.0058-1"
LDPC_DECODER_CUBIN_VERSION="2f9dcb" # ldpc hash {SHA:0:6}

CURL_AUTH_ARGS=()
if [[ -n "${NGC_API_KEY:-}" ]]; then
    # The NGC_API_KEY not needed for general release builds.
    # If needed for something else, set NGC_API_KEY and modify curl commands
    # https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html#download-resources-via-wget-authenticated-access
    CURL_AUTH_ARGS=(-H "Authorization: Bearer ${NGC_API_KEY}")

    # Path when using AUTH
    DOCA_CURL_VERSION_PATH="org/${NGC_ORG_TEAM}/resources/doca-for-aerial/versions/${DOCA_VERSION}/files/doca-for-aerial.zip"
    LDPC_CURL_VERSION_PATH="org/${NGC_ORG_TEAM}/resources/ldpc-decoder-cubin/versions/${LDPC_DECODER_CUBIN_VERSION}/files/ldpc_decoder_cubin.zip"
else
    # Path when using public artifact - no AUTH
    DOCA_CURL_VERSION_PATH="resources/org/${NGC_ORG_TEAM}/doca-for-aerial/${DOCA_VERSION}/files?redirect=true&path=doca-for-aerial.zip"
    LDPC_CURL_VERSION_PATH="resources/org/${NGC_ORG_TEAM}/ldpc-decoder-cubin/${LDPC_DECODER_CUBIN_VERSION}/files?redirect=true&path=ldpc_decoder_cubin.zip"
fi

echo "Downloading doca-for-aerial.zip..."
curl --fail "${CURL_AUTH_ARGS[@]}" -H "Content-Type: application/json" -L -o doca-for-aerial.zip "https://api.ngc.nvidia.com/v2/${DOCA_CURL_VERSION_PATH}"
unzip -o doca-for-aerial.zip

echo "Downloading ldpc_decoder_cubin.zip..."
curl --fail "${CURL_AUTH_ARGS[@]}" -H 'Content-Type: application/json' -L -o ldpc_decoder_cubin.zip "https://api.ngc.nvidia.com/v2/${LDPC_CURL_VERSION_PATH}"
unzip -o ldpc_decoder_cubin.zip

hpccm --recipe aerial_base_recipe.py --cpu-target $CPU_TARGET --userarg AERIAL_GPU_TYPE=${AERIAL_GPU_TYPE} --format docker > Dockerfile_tmp
if [[ -n "$AERIAL_BUILDER" ]]
then
    docker buildx build --builder $AERIAL_BUILDER --load --platform $AERIAL_PLATFORM -t $AERIAL_REPO$IMAGE_NAME:${AERIAL_VERSION_TAG} -f Dockerfile_tmp .
else
    DOCKER_BUILDKIT=1 docker build --network host --no-cache --platform $AERIAL_PLATFORM -t $AERIAL_REPO$IMAGE_NAME:${AERIAL_VERSION_TAG} -f Dockerfile_tmp .
fi
rm Dockerfile_tmp
