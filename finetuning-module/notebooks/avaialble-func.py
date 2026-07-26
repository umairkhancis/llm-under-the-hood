import sys
import gpt_training_module

# 1. Force reload the module from disk
import importlib
importlib.reload(gpt_training_module)

# 2. Print out where Python is actually loading this folder from
print("Package location:", gpt_training_module.__file__)

# 3. Print everything inside the package to see if plot_losses is visible
print("Available functions:", dir(gpt_training_module))
