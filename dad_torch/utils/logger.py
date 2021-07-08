import math as _math
import datetime, time


def error(msg, debug=True):
    if debug:
        print(f"[ERROR]! {msg}")


def warn(msg, debug=True):
    if debug:
        print(f"[WARNING]! {msg}")


def info(msg, debug=True):
    if debug: print(f"{msg}")


def success(msg, debug=True):
    if debug:
        print(f"[SUCCESS]! {msg}")


def lazy_debug(x, add=0):
    _scale = int(_math.log(max(x, 1)) * _math.log(max(add, 1)))
    return x % (_scale + 1) == 0


def duration(cache: dict, begin, key=None, t_del=None):
    if t_del is None:
        t_del = datetime.datetime.fromtimestamp(time.time()) - datetime.datetime.fromtimestamp(begin)

    if key is not None:
        if cache.get(key) is None:
            cache[key] = [t_del.total_seconds() * 1000]  # Millis
        else:
            cache[key].append(t_del.total_seconds() * 1000)  # Millis
    return t_del
