# Local llama.cpp runtime

`llama-b10666-rocm/` is a local, Git-ignored binary distribution of llama.cpp
build 10666 (`4e97ac86e`). Run its programs with the directory on
`LD_LIBRARY_PATH`, for example:

```bash
runtime="$PWD/tools/llama-b10666-rocm"
LD_LIBRARY_PATH="$runtime" "$runtime/llama-quantize" --help
```

The matching clean source checkout will be recorded separately after M1 setup.
