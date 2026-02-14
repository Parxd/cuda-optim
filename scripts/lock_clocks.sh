#!/bin/bash

# Credit to salykova: (https://github.com/salykova/sgemm.cu)
base_clock=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep "Graphics" | grep -om1 "[0-9]\+")
memory_clock=$(nvidia-smi -q -d SUPPORTED_CLOCKS | grep "Memory" | grep -om1 "[0-9]\+")
sudo nvidia-smi --lock-gpu-clocks=${base_clock}
sudo nvidia-smi --lock-memory-clocks=${memory_clock}