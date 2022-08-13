# pylint: disable=missing-module-docstring
import sys
current_word = None
current_count = 0
word = None
# input comes from standard input STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # parse the output from mapper.py
    word, count = line.split(' ', 1)
    # convert count to int
    try:
        count = int(count)
    except ValueError:
        continue
    # enter when Hadoop sorts map output by key
    # before it is passed to the reducer
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # reults written to STDOUT
            print('%s   %s' % (current_word, current_count))
        current_count = count
        current_word = word
if current_word == word:
    print('%s   %s' % (current_word, current_count))
