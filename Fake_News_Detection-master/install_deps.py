import subprocess
import sys

packages = ['flask', 'pandas', 'scikit-learn', 'nltk', 'seaborn', 'matplotlib']
subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
print('All packages installed successfully')
