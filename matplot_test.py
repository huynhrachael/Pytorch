# %%
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Print the active backend being used for rendering
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Using backend: {matplotlib.get_backend()}")

try:
    # Generate test data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label="$\sin(x)$", color="blue")
    ax.set_title("Matplotlib Installation Test")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.legend()
    ax.grid(True)

    # Save the file to verify file-writing works
    output_filename = "matplotlib_test_output.png"
    plt.savefig(output_filename, dpi=100)
    
    if os.path.exists(output_filename):
        print(f"Success! Test image saved to: {os.path.abspath(output_filename)}")
    
    # Try to open the interactive window (will skip if headless/no GUI available)
    print("Attempting to display interactive window...")
    plt.show()
    print("Test complete.")

except Exception as e:
    print(f"An error occurred during the test: {e}")

# %%
