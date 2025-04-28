import csv
import os


file_path = "8-to-1_Multiplexer.csv"
if not os.path.exists(file_path):
    data = [
        ["Control 2 (S2)", "Control 1 (S1)", "Control 0 (S0)", "Input 0 (I0)", "Input 1 (I1)", "Input 2 (I2)", "Input 3 (I3)", "Input 4 (I4)", "Input 5 (I5)", "Input 6 (I6)", "Input 7 (I7)", "Output (Y)"],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    ]

    
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"File created and data successfully written to {file_path}")
else:
    print(f"File already exists at {file_path}")