

def parse_bool(value):
    if isinstance(value, bool): 
        return value
    if isinstance(value, str): 
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    return bool(value)

def parse_float(value):
    """Parse float from string or number."""
    if isinstance(value, str):
        return float(value)
    return float(value)