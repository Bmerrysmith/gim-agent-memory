# Open GIM in Visual Studio

Open `gim-agent-memory.sln` in Visual Studio
Community 2026. Python development support is already installed on this computer.
The solution uses the existing checkout and includes the Python source, tests, and benchmarks.

The project selects `GIM Python 3.12.14 (.venv)`, whose executable is
`.venv\Scripts\python.exe`. Its working
directory is the project root and its Python search path includes `src`.

Press **Ctrl+F5** to run the workshop demo or **F5** to debug it. The startup file
is `main.py`; project Properties > Debug contains the workshop arguments. The
default command runs 12 episodes with seed 2026 and writes
`runs/workshop-g0-g1-demo.json`.

Right-click the project in Solution Explorer and open its **Python** menu:

| Command | Runs |
| --- | --- |
| GIM - Workshop demo | The same reproducible workshop as Ctrl+F5 |
| GIM - Iteration suite | `benchmarks/iteration_suite.py --label <timestamp> --profile` |
| GIM - All tests | `main.py check` |
| GIM - Status | `main.py status` |
| GIM - Storage benchmark | `benchmarks/benchmark_storage.py --output runs/storage-benchmark.json` |

These commands send their results to Visual Studio's Output window. The Test
Explorer settings also specify unittest, the `tests` directory, and `test_*.py`.
The existing `main.py check` command remains the verified full-suite entry point.

The project explicitly lists the Python source, tests, benchmark and research
documents. Virtual-environment files, IDE caches and run output are not source
items. When adding a new source file outside Visual Studio, use **Add > Existing
Item** to include it in the project. The project home remains this repository.

The migration adds IDE metadata and this guide. The existing code, hypotheses,
architecture, Git history and Python environment remain in the same checkout.
CLion metadata remains available for rollback. `.vs/` and per-user Visual Studio
settings are excluded from Git.

If this checkout moves to another computer, recreate `.venv` with Python 3.11 or
newer and select that environment in Visual Studio. The current `.venv` is tied
to its existing base-interpreter installation.

Microsoft documentation: [Python project environments](https://learn.microsoft.com/en-us/visualstudio/python/selecting-a-python-environment-for-a-project)
and [custom Python project commands](https://learn.microsoft.com/en-us/visualstudio/python/defining-custom-python-project-commands).
