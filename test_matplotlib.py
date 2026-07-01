import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create the line plot
plt.plot(x, y, label="Linear Growth")

# Add titles and labels
plt.title("Matplotlib in VS Code")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend()

# CRITICAL STEP: Display the plot window
plt.show()



