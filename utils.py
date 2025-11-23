from sympy import Symbol, sympify, diff, simplify


def find_polynomial_derivative(polynomial, variable='x'):
    """
    Finds the symbolic derivative of a polynomial.
    
    Args:
        polynomial: Can be:
            - A string representation (e.g., "3*x**2 + 2*x + 1" or "3x^2 + 2x + 1")
            - A sympy expression
            - A list of coefficients [a_n, a_{n-1}, ..., a_0] representing 
              a_n*x^n + a_{n-1}*x^(n-1) + ... + a_0
              Coefficients can be numbers, symbols, or strings representing symbols
        variable: The variable to differentiate with respect to (default: 'x')
    
    Returns:
        The symbolic derivative as a sympy expression
    
    Examples:
        >>> find_polynomial_derivative("3*x**2 + 2*x + 1")
        6*x + 2
        
        >>> find_polynomial_derivative([3, 2, 1])
        6*x + 2
        
        >>> find_polynomial_derivative("x**3 - 5*x**2 + 7*x - 2")
        3*x**2 - 10*x + 7
        
        >>> find_polynomial_derivative("a*x**2 + b*x + c")
        2*a*x + b
        
        >>> find_polynomial_derivative(["a", "b", "c"])
        2*a*x + b
    """
    x = Symbol(variable)
    
    # Handle list of coefficients
    if isinstance(polynomial, list):
        expr = 0
        for i, coeff in enumerate(polynomial):
            power = len(polynomial) - 1 - i
            # Convert string coefficients to symbols if needed
            if isinstance(coeff, str):
                coeff = sympify(coeff)
            expr += coeff * x**power
        return simplify(diff(expr, x))
    
    # Handle string or sympy expression
    if isinstance(polynomial, str):
        # Replace ^ with ** for Python syntax
        polynomial = polynomial.replace('^', '**')
        expr = sympify(polynomial)
    else:
        expr = polynomial
    
    return simplify(diff(expr, x))


def find_partial_derivative(polynomial, variable='x', other_variable='y'):
    """
    Finds the partial derivative of a polynomial with respect to one variable.
    
    Args:
        polynomial: Can be:
            - A string representation (e.g., "3*x**2*y + 2*x*y**2 + x*y")
            - A sympy expression
        variable: The variable to differentiate with respect to (default: 'x')
        other_variable: The other variable in the polynomial (default: 'y')
    
    Returns:
        The partial derivative as a sympy expression
    
    Examples:
        >>> find_partial_derivative("3*x**2*y + 2*x*y**2", 'x')
        6*x*y + 2*y**2
        
        >>> find_partial_derivative("3*x**2*y + 2*x*y**2", 'y')
        3*x**2 + 4*x*y
        
        >>> find_partial_derivative("a*x**2*y + b*x*y**2 + c*x*y", 'x')
        2*a*x*y + b*y**2 + c*y
    """
    var = Symbol(variable)
    
    # Handle string or sympy expression
    if isinstance(polynomial, str):
        # Replace ^ with ** for Python syntax
        polynomial = polynomial.replace('^', '**')
        expr = sympify(polynomial)
    else:
        expr = polynomial
    
    return simplify(diff(expr, var))
