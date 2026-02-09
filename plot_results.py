import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open("results_fullgpu.json", "r") as f:
    data = json.load(f)

plt.figure(figsize=(10, 6))

# Plot Float64 results
res = data["results"]["Float64"]

# No Preconditioner (usually slow)
plt.semilogy(res["noprecond_history"], 
             label="No Preconditioner", 
             linestyle="--", color="gray", alpha=0.7)

# With GPU ILU (should be much faster)
plt.semilogy(res["ilu_history"], 
             label=f"Fully GPU ILU (ω={data['metadata']['omega']})", 
             linewidth=2, color="blue")

plt.grid(True, which="both", alpha=0.3)
plt.xlabel("GMRES Iterations")
plt.ylabel("Relative Residual (Log Scale)")
plt.title(f"Convergence: {data['metadata']['grid_size']} Grid")
plt.legend()
plt.tight_layout()
plt.show()
