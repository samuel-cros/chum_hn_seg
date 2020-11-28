

min_value = -1000.0
max_value = 3071.0

# Takes an input value/array/.. and applies min/max normalization
def standardize(value):
    value -= min_value
    value /= (max_value - min_value)
    return value
