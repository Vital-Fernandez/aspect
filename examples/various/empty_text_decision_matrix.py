# Adding three spaces between the vertically aligned text columns to match the matrix width

# List of entries to append to each row
labels = [
    "undefined",
    "white-noise",
    "continuum",
    "emission",
    "cosmic-ray",
    "pixel-line",
    "broad",
    "doublet",
    "peak",
    "absorption",
    "dead-pixel",
    "trough",
    "Hbeta_OIII-doublet",
    "phl293B"]

# Create the 11x11 matrix of zeros
matrix_11x11 = [[0 for _ in range(len(labels))] for _ in range(len(labels))]

# Format the matrix as a string with each row in square brackets and separated by commas and spaces
formatted_matrix_11x11 = '\n'.join(
    ['[{}], #{}'.format(',  '.join(map(str, row)), labels[i]) for i, row in enumerate(matrix_11x11)]
)


max_len = len(labels)

# Create a 2D grid where each column corresponds to one string in the entry_list with 3 spaces separation
vertical_text_rows_with_spacing = []

# Iterate through each row position (up to max_len)
for i in range(max_len):
    row = []
    for entry in labels:
        if i < len(entry):
            row.append(entry[i])
        else:
            row.append(' ')  # Add space for shorter entries
    vertical_text_rows_with_spacing.append('   '.join(row))  # Join characters with three spaces

# Append the vertical text with spacing to the bottom of the matrix
formatted_matrix_with_spaced_vertical_text = f"{formatted_matrix_11x11}\n\n" + '\n'.join(vertical_text_rows_with_spacing)

# Writing this new format to a text file
file_path_matrix_with_spaced_vertical_text = './11x11_matrix_with_spaced_vertical_text.txt'
with open(file_path_matrix_with_spaced_vertical_text, 'w') as file:
    file.write(formatted_matrix_with_spaced_vertical_text)

