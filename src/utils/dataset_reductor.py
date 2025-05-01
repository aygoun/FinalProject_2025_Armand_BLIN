import csv
import sys


def reduce_csv(input_path, output_path, num_lines):
    """
    Reduces a CSV file to the first N lines.
    """
    try:
        with open(input_path, "r") as infile, open(
            output_path, "w", newline=""
        ) as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for i, row in enumerate(reader):
                if i >= num_lines:
                    break
                writer.writerow(row)

        print(f"Reduced CSV saved to {output_path} with {num_lines} lines.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python dataset_reductor.py <input_csv_path> <output_csv_path> <num_lines>"
        )
    else:
        input_csv_path = sys.argv[1]
        output_csv_path = sys.argv[2]
        num_lines = int(sys.argv[3])
        reduce_csv(input_csv_path, output_csv_path, num_lines)
