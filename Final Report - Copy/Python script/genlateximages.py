import os

# Define the directory containing the images
folder = r"D:\FYPSeagullClassification01\Final Report - Copy\images\interpretability\vgg"

# Supported image extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(image_extensions)])


# Generate LaTeX code for a 4-column grid with smaller images
def escape_latex(s):
    return s.replace('_', '\\_')


print("\\begin{figure}[htbp]")
print("    \\centering")

# Get sorted list of image files
image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(image_extensions)])

# Set number of columns
cols = 4
total_images = len(image_files)

for i, fname in enumerate(image_files):
    safe_fname = escape_latex(fname)

    # Start a new row
    if i % cols == 0 and i > 0:
        print("    ")
        print("    \\vspace{0.3em}")
        print("    ")

    # Print subfigure environment with smaller width
    print(f"    \\begin{{subfigure}}[b]{{0.22\\textwidth}}")
    print(f"        \\includegraphics[width=\\textwidth]{{images/interpretability/vgg/{safe_fname}}}")
    print(f"        \\caption{{}}")
    print(f"    \\end{{subfigure}}")

    # Add spacing between images (except for end of row)
    if (i + 1) % cols != 0 and i < total_images - 1:
        print("    \\hfill")

print("    \\caption{Bird interpretation visualizations.}")
print("    \\label{fig:bird-interpretability}")
print("\\end{figure}")

print("\n# Remember to include these packages in your preamble:")
print("\\usepackage{graphicx}")
print("\\usepackage{subcaption}")