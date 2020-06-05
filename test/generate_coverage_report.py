import os

'''
Generates code coverage report using the coverage module in python.
Disclaimer: This code was ran and tested on OSX, any other OS may have dependencies that need to be resolved before running the code.
'''
os.system('coverage run -m pytest'+ " >/dev/null ")
os.system('coverage report -m -i')
