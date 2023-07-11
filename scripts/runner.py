import os
import sys

#python3 runner.py  /path/to/directory/  /path/to/file.jar /path/to/results directory/

dir = sys.argv[1]
jar_path = sys.argv[2]
memory_limit = 13 * 1024

results = []

for root, dirs, files in os.walk(sys.argv[3]):
    for file in files:
        if file.endswith('.csv'):
            results.append(file)

repositories = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

for repository in repositories:
    if repository+'.csv' not in results:
        command = f"java -Xmx{memory_limit}m -jar {jar_path} -d {dir} -p {repository} -id '01-01-2012'"
        os.system(command)