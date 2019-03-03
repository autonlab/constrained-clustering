def print_verbose(string, verbose, level = 0, **args):
    if verbose > level:
        print(string, **args)