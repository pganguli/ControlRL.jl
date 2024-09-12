# ControlRL.jl
Starter code for applying reinforcement learning methods to control problems.

## Getting started

First clone this repo and `cd` into this repo. Then follow the following steps.

### Installing Julia
Follow the instructions from [julialang.org](https://julialang.org/downloads/). For Mac and Linux systems, this is done by
```bash
curl -fsSL https://install.julialang.org | sh
```

For Windows, you can either install it through Microsoft Store, or use the Linux instruction above with [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install).
```
winget install julia -s msstore
```

### Set up Julia environment
Once Julia is installed, type `julia` in terminal to open the Julia REPL. You should see something like 
```
julia> 
```
Press `]` to enter the built-in package manager. The prompt becomes
```
(@v1.10) pkg>
```
Install the Pluto.jl package to use interactive notebooks:
```
(@v1.10) pkg> add Pluto
```
Once it is installed, press backspace to exit the package manager, and type
```
julia> using Pluto; Pluto.run()
```
To launch the Pluto notebook.

### Use the example notebook
In the newly opened browser tab (Pluto should automatically open if for you; if not, copy and paste the URL from the terminal to your favorite browser) you should see the welcome page of Pluto. In "Open a notebook", select "examples/example.jl". The rest of the work will be done over there. Have fun!
