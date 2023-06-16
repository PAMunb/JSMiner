import csv
import git
import os
import stat
import sys

#python3 runner.py /path/to/directory/ /path/to/file.csv

cwd = sys.argv[1]
clear = []
count = 1
with open(sys.argv[2],newline='') as f:
    projects = csv.reader(f, delimiter=',')
    for project in projects:
        if project[1] == "repository_name":
            continue
        print(count)
        path = cwd+project[1].strip()
        try:
            if os.path.isdir(path):
                if not os.access(path, os.W_OK):
                    os.chmod(path, stat.S_IWUSR)
                    os.system("rmdir /s /q "+path)
                    print("cloning: " + project[1].strip())
                    git.Git(cwd).clone(project[2].strip())
                else:
                    os.system("rmdir /s /q "+path)
                    print("cloning: " + project[1].strip())
                    git.Git(cwd).clone(project[2].strip())
            else:
                print("cloning: " + project[1].strip())
                git.Git(cwd).clone(project[2].strip())
            count = count + 1
        except Exception as e:
            print(e)
            print(path)
            clear.append(project[1].strip())
            if os.path.isdir(path):
                os.chmod(path, stat.S_IWUSR)
                os.system("rmdir /s /q "+path)
    print(clear)
