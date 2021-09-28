# Example C++ Zenoh Flow Operator with OpenCV

[![Join the chat at https://gitter.im/atolab/zenoh-flow](https://badges.gitter.im/atolab/zenoh-flow.svg)](https://gitter.im/atolab/zenoh-flow?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[Zenoh Flow](https://github.com/eclipse-zenoh/zenoh-flow) provides a Zenoh-based dataflow programming framework for computations that span from the cloud to the device.

:warning: **This software is still in alpha status and should _not_ be used in production. Breaking changes are likely to happen and the API is not stable.**

-----------
## Description

:warning: This example works only on Linux and it require OpenCV with CUDA enabled to be installed, please follow the instruction on [this gits](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7) to install it.

:warning: This example works only on Linux and it require a **CUDA** capable **NVIDIA GPU**, as well as NVIDIA CUDA and CuDNN to be installed, please follow [CUDA instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuDNN instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

## Compiling

### Requirements

- Rust: see the [installation page](https://www.rust-lang.org/tools/install)
- cxxbridge

  ```sh
  cargo install cxxbridge-cmd
  ```

### Generating the shared library

On Unix-based machines.

```sh
mkdir build && cd build
cmake ..
make
```

This will:
1. "patch" the header file `include/wrapper.hpp`,
2. compile the Rust code located under the `vendor/wrapper` folder and generate a static library `libwrapper.a`,
3. compile the C++ wrapper code,
4. compile the operator,
5. link everything together producing `build/libcxx_operator.so`.

The `libcxx_operator` library can then be loaded by Zenoh Flow!

