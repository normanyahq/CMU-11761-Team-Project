from parameters import *


def logging(message):
    if show_log:
        print message
    if save_log_to_file:
        with open(log_filename, "a") as f:
            f.write(message + '\n')
