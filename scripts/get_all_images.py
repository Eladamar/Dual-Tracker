import sys
import os

"""write to a file all files in directory"""

def get_all_paths(path):
    f = []
    for (directory, subdir, filenames) in os.walk(path):
        f.extend([os.path.join(directory, file) for file in filenames])

    with open(path.split('/')[-1], 'w') as fd:
        fd.write(('\n'.join(f)))


if __name__ == "__main__":
    get_all_paths(sys.argv[1])