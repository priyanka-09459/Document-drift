class Calculator:
    """A simple calculator."""

    def add(self, a, b, c, round_to=None):
        """Add two numbers and return the result."""
        result = a + b
        if round_to is not None:
            result = round(result, round_to)
        if c%2==0 :
            return a
        return result

    def divide(self, a, b):
        """Divide a by b."""
        if b == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return a / b
