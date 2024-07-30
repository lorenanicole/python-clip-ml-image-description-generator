from collections import OrderedDict
from functools import wraps

class Cache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.cache_values = OrderedDict()
    
    def get(self, key):
        # import pdb; pdb.set_trace()
        if key not in self.cache_values:
            self.misses += 1
            return -1
        else:
            self.hits += 1
            self.cache_values.move_to_end(key)
            return self.cache_values[key]
    
    def put(self, key, value):
        self.cache_values[key] = value
        self.cache_values.move_to_end(key)

        if len(self.cache_values) > self.max_size:
            self.cache_values.popitem(last=False)
    
    def __repr__(self):
        return f"Cache: {self.cache_values}, Misses: {self.misses}, Hits: {self.hits}"

def my_lru_cache(func, max_size=128):
    cache = Cache(max_size)
    @wraps(func)
    def wrapped(*args, **kwargs):
        value = cache.get(*args, **kwargs)
        if value == -1:
            # import pdb; pdb.set_trace()
            value = func(*args, **kwargs)
            cache.put(value, *args, **kwargs)
        return value
    
    def cache_info():
        return str(cache)

    wrapped.cache_info = cache_info
    return wrapped