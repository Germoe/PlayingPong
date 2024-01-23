In order to run the code we need to install CMake, the dependencies and run `autorom` to initialize the directory for the emulator files.

---

On Linux:

    ```bash
    wget https://cmake.org/files/v3.4/cmake-3.4.1-Linux-x86_64.tar.gz
    tar xf cmake-3.4.1-Linux-x86_64.tar.gz
    export PATH="`pwd`/cmake-3.4.1-Linux-x86_64/bin:$PATH" # save it in .bashrc if needed
    ```

On Mac:

    ```bash
    brew install cmake
    ```

---

Install the dependencies:

```bash
pip install 'pettingzoo[atari,accept-rom-license]' opencv-python tensorboard supersuit torchrl autorom
```

---

Initialize AutoROM

```bash
AutoROM
```
