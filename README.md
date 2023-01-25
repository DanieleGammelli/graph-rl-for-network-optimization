# Graph Reinforcement Learning for Network Control via Bi-Level Optimization
Official implementation of "Graph Reinforcement Learning for Network Control via Bi-Level Optimization"

## Prerequisites

You will need to have a working IBM CPLEX installation. If you are a student or academic, IBM is releasing CPLEX Optimization Studio for free. You can find more info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students)

To install all required dependencies, run
```
pip install -r requirements.txt
```
**Important**: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```

## Contents

* `supplychain/`: Code to reimplement Supply Chain Inventory Management experiments (Section 5.2).
* `mobility/`: Code to reimplement Dynamic Vehicle Routing experiments (Section 5.3).

Please refer to the respective README.md files for further details about the usage of both sub-directories.
