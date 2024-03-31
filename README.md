# CSE3150 - Lab 6

## Compilation

`make all` to compile both testing and `main()` binaries.

`make check` to compile and run tests.

`make run` to compile and execute `./lab6.out` with default filenames.

### Usage

`./lab6.out <file1> <file2> <file3>` to use input files other than the default.

### Note

The available SIMD intrinsics will vary based on what CPU your compiler targets.

In particular, AVX512 may not be supported on your platform.

Therefore you may encounter errors with the default settings. Use the following
options to disable certain SIMD instruction sets and complile without errors.

To compile **without** support for AVX512, append `NOAVX512=1` to the
`make` command. 

**e.g.** `make run NOAVX512=1`

To compile **without** support for AVX2, append `NOAVX2=1` to the
`make` command. This will also disable support for AVX512.

**e.g.** `make all NOAVX2=1`

If your platform does not support AVX1 intrinsics, I am afraid you are compiling
on a potato.

