import matplotlib.pyplot as plt
import os

# Create a loop for generating and saving figures
for i in range(5):  # Replace '5' with the desired number of figures
    # Your code to create and customize the plot goes here
    x = [i for i in range(10)]
    y = [i * (j + 1) for j in range(10)]
    a = [i for i in range(20)]
    b = [i * (j + 1) for j in range(20)]

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Figure {i + 1}')

    # Create a folder to save the figures if it doesn't exist
    output_folder = 'output_figures'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the figure with a unique name in the output folder
    output_path = os.path.join(output_folder, f'figure_{i + 1}.png')
    plt.savefig(output_path)

    # Clear the current figure for the next iteration
    plt.clf()

    plt.plot(a, b)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(f'Figure {i + 1}')

    # Create a folder to save the figures if it doesn't exist
    output_folder = 'output_figures'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the figure with a unique name in the output folder
    output_path = os.path.join(output_folder, f'xxx{i + 1}.png')
    plt.savefig(output_path)

    # Clear the current figure for the next iteration
    plt.clf()

# Optionally, display the figures if needed
plt.show()