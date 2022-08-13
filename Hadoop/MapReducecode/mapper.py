import sys
# input comes from standard input STDIN
for line in sys.stdin:
    # remove leading and trailing whitespaces
    line = line.strip()
    print(line)
    # split the line into words and returns as a list
    words = line.split()
for word in words:
    print('%s    %s' % (word, 1))  # Reults written to STDOUT
