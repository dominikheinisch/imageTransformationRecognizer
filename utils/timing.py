import time


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took: %0.3f s' % (f.__name__, (time2 - time1)))
        return ret

    return wrap
