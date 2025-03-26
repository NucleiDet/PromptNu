import sys
import re
from tensorboardX import SummaryWriter

def extract_and_log_data(log_file, keys, log_dir):
    pattern_str = r'Epoch \[(\d+)\]'
    for key in keys:
        pattern_str += r'.*?%s: (\d+\.?\d*)' % key
    pattern = re.compile(pattern_str)

    writer = SummaryWriter(log_dir=log_dir)

    with open(log_file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                for i, key in enumerate(keys, 2):
                    value = match.group(i)
                    if value is not None:
                        writer.add_scalar(key, float(value), epoch)

    writer.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <log_file_path> <keys_comma_separated>")
        sys.exit(1)

    log_file = sys.argv[1]
    keys = sys.argv[2].split(',')

    extract_and_log_data(log_file, keys, log_dir='runs/loss_log')

if __name__ == "__main__":
    main()
