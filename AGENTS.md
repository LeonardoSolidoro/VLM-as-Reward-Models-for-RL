# Google Anigravity Agent Instructions (AGENTS.md)

This file contains the core directives, environment constraints, and coding standards for the AI agent working on this project. 

## 1. Environment & Dependency Management
- **Target Environment:** All code execution and package installation must occur within the Miniconda environment named `VLM_RM`.
- **Primary Package Manager:** Use `pip` for dependency management.
- **Dependency Tracking:** Update `environment.yml` or `requirements.txt` implicitly when adding new dependencies to maintain environment reproducibility.

## 2. Error Handling & Code Paradigms (STRICT)
- **Fail-Fast Philosophy:** Code must be designed to fail loudly and immediately when encountering unexpected states. Do not attempt to silently swallow errors.
- **No Dict Fallbacks:** NEVER use the `.get()` method on dictionaries to provide silent fallbacks for missing data. Access keys directly (e.g., `config['hyperparameter']`) so that a `KeyError` is immediately raised if a required key is missing.
- **Explicit Exception Handling:** NEVER use `try...except` blocks without printing or logging the exact error message and traceback. Bare `except:` or `except Exception: pass` is strictly prohibited. If an error is caught, it must be printed, logged, or re-raised.

## 3. General Deep Learning Best Practices
- **Global Hyperparameter Exposure:** Always expose useful hyperparameters (e.g., learning rate, batch size, hidden dimensions, epochs) globally. Utilize a configuration file (like YAML, JSON, or Hydra) if the project is configured for it, or use `argparse` for command-line arguments in standalone `.py` scripts. Never hardcode hyperparameters deep within functions or classes.
- **Device Agnosticism:** Never hardcode `.cuda()` or `.cpu()`. Always use dynamic device allocation (e.g., `device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')`) and apply it consistently to models and tensors.
- **Deterministic Operations (Reproducibility):** Always include and call a global seed-setting function at the beginning of training scripts. This must set seeds for `torch`, `numpy`, and Python's built-in `random` module, and configure deterministic algorithms in PyTorch if requested.
- **Proper Checkpointing:** When saving models, do not just save the model structure. Always save a comprehensive checkpoint dictionary containing the model's `state_dict`, the `optimizer`'s `state_dict`, the current epoch, and the latest loss/metric.
- **Tensor Shape Documentation:** When writing custom PyTorch `nn.Module` classes or complex tensor operations, include inline comments or docstrings specifying the expected tensor shapes at each step (e.g., `# x shape: (Batch, Channels, Height, Width)`).
- **Efficient Data Loading:** Utilize PyTorch `DataLoader` with an appropriate number of `num_workers`. Ensure transformations are optimized and that data is pinned to memory (`pin_memory=True`) when training on GPUs to prevent data-loading bottlenecks.
- **Type Hinting:** Enforce strict Python type hinting for function arguments and return types to improve code readability and static analysis, especially for data pipelines and utility functions.
