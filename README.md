# Optimzing CUDA Kernels
A "worklog" of kernels and notes on optimizing CUDA kernels for common ops. in machine learning

Currently written in:
- raw CUDA
- CUTLASS / CuTe

Ops. attempted:
- (unary / binary) elementwise op.
- float32 GEMM

To build and run an operation:
```bash
$ mkdir build && cd build
$ cmake ..
$ make <OP_NAME> && ./<OP_NAME>
```

To register an operation (for ref.):
- create dir. under `csrc/<OP_NAME>`
- create driver file `<OP_NAME>.cpp`
- register op. in CMake: `add_op(<OP_NAME> csrc/<OP_NAME>)`