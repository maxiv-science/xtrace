import os
import sys
directory_path = os.path.abspath(os.path.join('lib'))
if directory_path not in sys.path:
    sys.path.append(directory_path)