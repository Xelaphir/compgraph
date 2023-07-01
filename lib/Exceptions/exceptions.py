
class EngineException(Exception):
    """Basic class of custom exeptions."""
    def __init__(self):
        pass
 
    def __str__(self):
        pass

class DivisionByZeroException(EngineException):
    """Division by zero."""
    def __init__(self):
        self.message = "Division by zero"
 
    def __str__(self):
        return self.message
    
class RestictException(EngineException):
    """Restricted operation."""
    def __init__(self):
        self.message = "Restricted operation"
 
    def __str__(self):
        return self.message
    
class InitializationException(EngineException):
    """Wrong initialization."""
    def __init__(self):
        self.message = "Wrong initialization"
 
    def __str__(self):
        return self.message
    
class ArgumentsException(EngineException):
    """Wrong arguments."""
    def __init__(self):
        self.message = "Wrong arguments"
 
    def __str__(self):
        return self.message

class DimensionArgumentsException(EngineException):
    """Dimensions of arguments should be suitable."""
    def __init__(self):
        self.message = "Dimensions of arguments should be suitable"
 
    def __str__(self):
        return self.message
    
class ClassArgumentsException(EngineException):
    """Wrong type of argument."""
    def __init__(self, arg):
        self.message = f"Wrong class of argument: {arg.__class__}"
 
    def __str__(self):
        return self.message

class NonSquareMatrixException(EngineException):
    """Matrix is non-square."""
    def __init__(self, matrix):
        mes = f"Matrix is non-square. Has dimension {matrix.n} x {matrix.m}"
        self.message = mes
 
    def __str__(self):
        return self.message

class ZeroDeterminantMatrixException(EngineException):
    """Determinant of matrix equals zero."""
    def __init__(self):
        self.message = "Determinant of matrix equals zero"
 
    def __str__(self):
        return self.message
