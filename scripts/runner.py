import os
import sys

#python3 runner.py  /path/to/directory/  /path/to/file.jar

dir = sys.argv[1]
jar_path = sys.argv[2]
memory_limit = 13 * 1024

repositories = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

for repository in repositories:
    command = f"java -Xmx{memory_limit}m -jar {jar_path} -d {dir} -p {repository}"
    os.system(command)