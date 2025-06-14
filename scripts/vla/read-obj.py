
# read the .pt file and print the contents
import sys
import torch
def main():
    if len(sys.argv) != 2:
        print("Usage: python read_pt.py <file.pt>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        data = torch.load(file_path, map_location='cpu')
        print(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()