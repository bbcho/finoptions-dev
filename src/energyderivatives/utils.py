def docstring_from(source):
    """
    Decorator to be used to copy the docstring from the source class/class method
    to the target. Useful for class composition.
    """

    def wrapper(target):
        target.__doc__ = source.__doc__
        return target

    return wrapper
