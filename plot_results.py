import json
import matplotlib
# Force a non-interactive backend to fix libEGL/screen warnings on servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

# 1. Load the data
json_filename = "results_fullgpu.json"

if not os.path.exists(json_filename):
    print(f"Error: {json_filename} not found.")
    exit(1)

with open(json_filename, "r") as f:
    data = json.load(f)

# 2. Extract metadata for the title
meta = data.get("metadata", {})
grid_info = meta.get("grid_size", "Unknown Grid")
omega_val = meta.get("omega", "?")
jacobi_iters = meta.get("jacobi_iters", "?")

# 3. Setup the plot
plt.figure(figsize=(10, 6))
colors = {"Float64": "blue", "Float32": "green", "Float16": "red"}
linestyles = {"Float64": "-", "Float32": "--", "Float16": ":"}

# 4. Loop dynamically through whatever precisions exist in the file
results_dict = data.get("results", {})
if not results_dict:
    print("Error: No results found in JSON file.")
    exit(1)

print(f"Found results for: {list(results_dict.keys())}")

for precision, res in results_dict.items():
    # Get style based on precision (default to black/solid if unknown)
    c = colors.get(precision, "black")
    ls = linestyles.get(precision, "-")
    
    # Check if history exists (it might not if you didn't update the Julia code yet!)
    if "noprecond_history" in res and "ilu_history" in res:
        # Plot No Preconditioner (only once, or differentiate by alpha)
        # We use a thin line for the non-preconditioned baseline
        plt.semilogy(res["noprecond_history"], 
                     linestyle=":", color=c, alpha=0.4, 
                     label=f"{precision} (No Precond)")

        # Plot With ILU (Thick line)
        plt.semilogy(res["ilu_history"], 
                     linestyle=ls, color=c, linewidth=2, 
                     label=f"{precision} (GPU ILU)")
    else:
        print(f"Warning: Convergence history missing for {precision}. Did you update the Julia script?")

# 5. Finalize and Save
plt.grid(True, which="both", alpha=0.3)
plt.xlabel("GMRES Iterations")
plt.ylabel("Relative Residual (Log Scale)")
plt.title(f"Convergence: {grid_info} | Jacobi iters: {jacobi_iters} | $\omega$={omega_val}")
plt.legend()
plt.tight_layout()

output_file = "convergence_plot.png"
plt.savefig(output_file, dpi=300)
print(f"Plot saved to {output_file}")
