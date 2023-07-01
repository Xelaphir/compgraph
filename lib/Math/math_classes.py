
import math
from typing import Union

from lib.Exceptions.exceptions import *

PRECISION = 10


def set_precision(prec: int = 10):

    if not isinstance(prec, int):
        raise ArgumentsException()
    
    global PRECISION
    PRECISION = prec


def bilinear_form(matrix: 'Matrix',
                  v1: 'Vector',
                  v2: 'Vector') -> Union[float, int]:
    """Bilinear form of two vectors and given matrix.

    Args:
        matrix (Matrix): Matrix for matrix multiplication.
        v1 (Vector): Vector number one.
        v2 (Vector): Vector number two.

    Raises:
        ClassArgumentsException: Wrong type of argument for method.

    Returns:
        Union[float, int]: Value of bilinear form.

    """
    if not isinstance(matrix, Matrix):
        raise ClassArgumentsException(matrix)
    
    if not isinstance(v1, Vector):
        raise ClassArgumentsException(v1)
    
    if not isinstance(v2, Vector):
        raise ClassArgumentsException(v2)

    result = v1.transpose() * matrix * Matrix(v2.matrix)

    return round(result[0, 0], PRECISION)


class Matrix:
    """Basic class of matrix.

    Note:
        Initialize ``Matrix`` object using list of lists with elements of matrix.

    """
    def __init__(self, matrix: list[list[Union[int, float, 'Vector']]]):

        with_vector = False

        if not isinstance(matrix, list):
            raise InitializationException()
        
        if not isinstance(matrix[0], list):
            raise InitializationException()
        
        n = len(matrix)
        m = len(matrix[0])
        
        for row in range(n):

            if not isinstance(matrix[row], list):
                raise InitializationException()
            
            if len(matrix[row]) != m:
                raise InitializationException()

            for item in range(m):

                if not isinstance(matrix[row][item], (int, float, Vector)):
                    raise InitializationException()
                
                if isinstance(matrix[row][item], Vector):
                    with_vector = True
                
                if isinstance(matrix[row][item], (int, float)):
                    matrix[row][item] = round(matrix[row][item], PRECISION)

        self.n = n
        self.m = m
        self.matrix = matrix
        self.with_vector = with_vector

    @classmethod
    def zero(cls, n: int, m: int = None) -> 'Matrix':
        """Creating square or non-square matrix with zero elements.

        Note:
            Using one argument matrix will be square.

        Args:
            n (int): Number of rows
            m (int, optional): Number of columns. Defaults to None. 

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            ArgumentsException: Wrong argument for method.

        Returns:
            Matrix: Zero matrix.

        """
        if not isinstance(n, int):
            raise ClassArgumentsException(n)
        
        if m is None:
            m = n
        
        if not isinstance(m, int):
            raise ClassArgumentsException(m)

        if n <= 0 or m <= 0:
            raise ArgumentsException()

        matrix = [[0 for _ in range(m)] for _ in range(n)]

        return cls(matrix)

    @classmethod
    def identity(cls, n: int) -> 'Matrix':
        """Creating square matrix with ones in main diagonal line.

        Args:
            n (int): Dimension of the identity matrix.

        Returns:
            Matrix: Identity matrix.

        """
        matrix = cls.zero(n)

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1

        return matrix

    @classmethod
    def gram(cls, n: int, list_of_vector: list["Vector"]) -> 'Matrix':
        """Creating square matrix with elements which are pairwise scalar 
        product of vectors from the list.

        Args:
            n (int): Dimension of gram matrix.
            list_of_vector (list[Vector]): List of vectors to 
                do scalar product.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimensions of arguments.

        Returns:
            Matrix: Gram matrix.

        """
        if not isinstance(list_of_vector, list):
            raise ClassArgumentsException(list_of_vector)
        
        if len(list_of_vector) != n:
            raise DimensionArgumentsException()
        
        for item in list_of_vector:

            if not isinstance(item, Vector):
                raise ClassArgumentsException(item)
            
            if item.n != n:
                raise DimensionArgumentsException()


        matrix = cls.zero(n)

        for i in range(n):
            for j in range(n):
                result = list_of_vector[i].scalar_product(list_of_vector[j])
                matrix[i, j] = result

        return matrix

    @classmethod
    def rotation_matrix(cls, 
                        plain: tuple[int, int], 
                        angle: Union[int, float], 
                        n: int = None) -> 'Matrix':
        """Creating square matrix which rotate vector on given plain given 
        angle.

        Args:
            plain (tuple[int]): Tuple of pair axis.
            angle (Union[int, float]): Angle in **radians**.
            n (int): Dimension of space.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            ArgumentsException: Wrong argument for method.

        Returns:
            Matrix: Rotation matrix.

        """
        if not isinstance(plain, tuple):
            raise ClassArgumentsException(plain)
        
        if len(plain) != 2:
            raise ArgumentsException()
        
        for axis in plain:
            if not isinstance(axis, int):
                raise ArgumentsException()
        
        if not isinstance(angle, (float, int)):
            raise ClassArgumentsException(angle)
        
        if (n is not None) and not isinstance(n, int):
            raise ClassArgumentsException(n)
        
        if n is None:
            n = 3
        
        axis1 = plain[0]
        axis2 = plain[1]

        if (
            (n < 2 or axis1 < 0 or axis2 < 0) 
            or (axis1 == axis2) 
            or (axis1 >= n or axis2 >= n)
        ):
                raise ArgumentsException()

        matrix = cls.identity(n)

        matrix[axis1, axis1] = round(math.cos(angle), PRECISION)
        matrix[axis2, axis2] = round(math.cos(angle), PRECISION)
        matrix[axis1, axis2] = round((-1)**(axis1 + axis2) * \
                                     math.sin(angle), PRECISION)
        matrix[axis2, axis1] = round((-1)**(axis1 + axis2 + 1) * \
                                     math.sin(angle), PRECISION)
        
        return matrix

    @classmethod
    def tait_bryan_matrix(cls, *angles) -> 'Matrix':
        """Creating square matrix which rotate vector on three axis given 
        angle.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            ArgumentsException: Wrong argument for method.

        Returns:
            Matrix: Matrix for rotating vector three dimensions.

        """
        if not isinstance(angles, tuple):
            raise ClassArgumentsException(angles)
        
        if len(angles) != 3:
            raise ArgumentsException()
        
        for angle in angles:
            if not isinstance(angle, (int, float)):
                raise ArgumentsException()

        tait_bryan = Matrix.rotation_matrix((1, 2), angles[0], 3) * \
                     Matrix.rotation_matrix((2, 0), angles[1], 3) * \
                     Matrix.rotation_matrix((0, 1), angles[2], 3)
        
        return tait_bryan


    def __getitem__(self,
                    keys: Union[int, tuple[int]]) -> Union[int,
                                                           float, 
                                                           'Vector']:
        """Getting row of Matrix as Vector object or single element from the 
        Matrix.

        Note:
            To get single element from ``MatrixObject`` use 
            ``MatrixObject[index1, index2]``, instead of 
            ``MatrixObject[index1][index2]``.

        Args:
            keys (Union[int, tuple[int]]): Indexes of getting element or row.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Union[int, float, Vector]: Vector object when argument has one
            index, and int or float when has two indexes.

        """
        if not isinstance(keys, tuple) and not isinstance(keys, int):     
            raise ClassArgumentsException(keys)

        if isinstance(keys, int):

            if keys >= self.n or keys < 0:
                raise ArgumentsException()

            key = keys

            vector = Vector(self.matrix[key])

            return vector
        
        if isinstance(keys, tuple):

            if len(keys) != 2:
                raise ArgumentsException()
            
            for key in keys:
                if not isinstance(key, int):
                    raise ClassArgumentsException(key)
                
            if not 0 <= keys[0] < self.n or not 0 <= keys[1] < self.m:
                raise ArgumentsException()

            row = keys[0]
            column = keys[1]

            return self.matrix[row][column]


    def __setitem__(self,
                    keys: Union[int, tuple[int]], 
                    value: Union[int, float, 'Vector', list]):
        """Setting row as Vector object or list of elements, or setting single 
        element.

        Note:
            To set single element from ``MatrixObject`` use 
            ``MatrixObject.[index1, index2] = value``, instead of 
            ``MatrixObject.[index1][index2] = value``.

        Args:
            keys (Union[int, tuple[int]]): Indexes of setting single element 
                or row.
            value (Union[int, float, Vector, list]): Setting value of single 
                element or row.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            ArgumentsException: Wrong argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        """
        if not isinstance(keys, tuple) and not isinstance(keys, int):
            raise ClassArgumentsException(keys)
        
        if isinstance(keys, int):

            if keys >= self.n or keys < 0:
                raise ArgumentsException()

            if not isinstance(value, (Vector, list)):
                raise ClassArgumentsException(value)
            
            if isinstance(value, Vector):

                if value.n != self.m:
                    raise DimensionArgumentsException()
                
                key = keys
            
                result = []

                for item_in_list in value.matrix:
                    for item in item_in_list:
                        result.append(item)
            
            if isinstance(value, list):

                if len(value) != self.m:
                    raise DimensionArgumentsException()
                
                key = keys
            
                result = []

                for item in value:

                    if not isinstance(item, (int, float, Vector)):
                        raise ClassArgumentsException(item)

                    if isinstance(item, (int, float)):
                        result.append(round(item, PRECISION))
                    
                    else:
                        result.append(item)

            self.matrix[key] = result

        
        if isinstance(keys, tuple):

            if not isinstance(value, (int, float, Vector)):
                raise ClassArgumentsException(value)

            if len(keys) != 2:
                raise ArgumentsException()
            
            for key in keys:

                if not isinstance(key, int):
                    raise ClassArgumentsException(key)
                
            if not 0 <= keys[0] < self.n or not 0 <= keys[1] < self.m:
                raise ArgumentsException()

            row = keys[0]
            column = keys[1]

            if isinstance(value, (int, float)):
                self.matrix[row][column] = round(value, PRECISION)
            
            else:
                self.matrix[row][column] = value

    
    def addition(self, other: 'Matrix') -> 'Matrix':
        """Adding matrixes with suitable dimensions.

        Note:
            Alternative way of call this function use ``+`` as binary operator.

        Args:
            other (Matrix): Adding matrix.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Matrix: Result matrix after addition.

        """
        if not isinstance(other, Matrix):
            raise ClassArgumentsException(other)

        if self.n != other.n or self.m != other.m:
            raise DimensionArgumentsException()
        
        else:

            matrix = Matrix.zero(self.n, self.m)

            for i in range(self.n):
                for j in range(self.m):
                    matrix[i, j] = round(self[i, j] + other[i, j], PRECISION)

            return matrix 

    
    def multiplication(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        """Multipling matrixes with suitable dimensions or multipling matrix 
        and number.

        Note:
            Alternative way of call this function use ``*`` as binary operator.

        Args:
            other (Union[float , int , Matrix]): Multipling matrix or number.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Matrix: Result matrix after multiplication.

        """
        if not isinstance(other, (float, int, Matrix)):
            raise ClassArgumentsException(other)

        if isinstance(other, Matrix) and self.m != other.n:
            raise DimensionArgumentsException()

        if isinstance(other, (float, int)):

            matrix = Matrix.zero(self.n, self.m)

            for i in range(self.n):
                for j in range(self.m):
                    matrix[i, j] = round(self[i, j] * other, PRECISION)

        elif isinstance(other, Matrix):

            matrix = Matrix.zero(self.n, other.m)

            for i in range(self.n):
                for j in range(other.m):
                    for k in range(self.m):
                        matrix[i, j] = round(matrix[i, j] + self[i, k] * \
                                             other[k, j], PRECISION)

        return matrix
    
    
    def subtraction(self, other: 'Matrix') -> 'Matrix':
        """Subtracting matrixes with suitable dimensions, based on adding and 
        multipling.

        Note:
            Alternative way of call this function use ``-`` as binary operator.

        Args:
            other (Matrix): Subtracting matrix.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Matrix: Result matrix after subtraction.

        """
        return self + other*(-1)
    
    
    def division(self, num: Union[float, int]) -> 'Matrix':
        """Dividing matrix and number. 

        Note:
            Alternative way of call this function use ``/`` as binary operator.

        Args:
            other (Union[float , int]): Dividing matrix or number.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.
            DivisionByZeroException: Division by zero.

        Returns:
            Matrix: Result matrix after division.

        """
        if not isinstance(num, (float, int)):
            raise ClassArgumentsException(num)

        if num == 0:
            raise DivisionByZeroException()
        
        return self * round((1 / num), PRECISION)
    

    def get_minor(self, rows: list[int], columns: list[int]) -> 'Matrix':
        """Creating matrix from the origin matrix deleting rows and columns.

        Args:
            rows (list[int]): List of numbers of rows to delete.
            columns (list[int]): List of numbers of columns to delete.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            ArgumentsException: Wrong argument for method.

        Returns:
            Matrix: Minor of the matrix.

        """
        if not isinstance(rows, list):
            raise ClassArgumentsException(rows)
        
        if self.n < len(rows):
            raise ArgumentsException()
        
        for row in rows:
            if not isinstance(row, int):
                raise ArgumentsException()
            
            if row >= self.n:
                raise ArgumentsException()
        
        if not isinstance(columns, list):
            raise ClassArgumentsException(columns)
        
        if self.m < len(columns):
            raise ArgumentsException()
        
        for column in columns:
            if not isinstance(column, int):
                raise ArgumentsException()
            
            if column >= self.m:
                raise ArgumentsException()
            
        minor = Matrix.zero(self.n - len(rows), self.m - len(columns))

        delta_row = 0
        
        for i in range(self.n):

            if i in rows:
                delta_row += 1
                continue
            
            delta_column = 0
            for j in range(self.m):

                if j in columns:
                    delta_column += 1
                    continue

                minor[i - delta_row, j - delta_column] = self[i, j]

        return minor
  

    def determinant(self) -> Union[float, int, 'Vector']:
        """Method returning determinant of square matrix.

        Raises:
            NonSquareMatrixException: Matrix is non-square.

        Returns:
            Union[float, int, Vector]: Number or Vector object if matrix has 
            Vector elements.

        """
        if self.n != self.m:
            raise NonSquareMatrixException(self)
        
        if self.with_vector:

            determinant = Vector([[0], [0], [0]])

        else:

            determinant = 0

        if self.n == 1:
            return round(self[0, 0], PRECISION)

        if self.n == 2:
            return round(self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0], \
                         PRECISION)

        for c in range(self.n):
            minor = self.get_minor([0], [c])
            determinant = determinant + ((-1) ** c) * self[0, c] * \
                Matrix(minor.matrix).determinant()

        return determinant
    
    
    def inverse(self) -> 'Matrix':
        """Inverting matrix.

        Note:
            Alternative way of call this function use ``~`` as unary operator.

        Raises:
            ZeroDeterminantMatrixException: Determinant equals zero.

        Returns:
            Matrix: Result matrix after inverting.

        """
        det = self.determinant()

        if det == 0:
            raise ZeroDeterminantMatrixException()

        if self.n == 1:
            return Matrix([[round(1 / self[0, 0], PRECISION)]])

        if self.n == 2:
            return Matrix([[round(self[1, 1] / det, PRECISION),
                            round(-1 * self[0, 1] / det, PRECISION)],
                           [round(-1 * self[1, 0] / det, PRECISION), 
                            round(self[0, 0] / det, PRECISION)]])

        cofactors = Matrix.zero(self.n, self.m)

        for i in range(self.n):
            for j in range(self.m):

                minor = [row[:j] + row[j + 1:] 
                         for row in (self.matrix[:i] + self.matrix[i + 1:])]

                minor_det = Matrix(minor).determinant()
                cofactors[i, j] = round(((-1) ** (i + j)) * minor_det / det,
                                        PRECISION)
        
        return cofactors
    
    
    def transpose(self) -> 'Matrix':
        """Transposing matrix.

        Returns:
            Matrix: Result matrix after transposing.

        """
        return Matrix(list(map(list, zip(*self.matrix))))
    
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        return self.addition(other)
    
    
    def __mul__(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        return self.multiplication(other)
    

    def __rmul__(self, other: Union[float, int, 'Matrix']) -> 'Matrix':
        if isinstance(other, Matrix):
            return other.multiplication(self)
        
        return self.multiplication(other)

  
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        return self.subtraction(other)
    
  
    def __truediv__(self, num: Union[float, int]) -> 'Matrix':
        return self.division(num)

   
    def __invert__(self) -> 'Matrix':
        return self.inverse()

    def __eq__(self, other: 'Matrix') -> bool:

        if isinstance(other, Matrix) and \
            self.__class__ == other.__class__ and \
            self.n == other.n and \
            self.m == other.m:
                is_not_eq = False

                for i in range(self.n):
                    for j in range(self.m):
                        if abs(self[i, j] - other[i, j]) > 10**(-PRECISION):
                            is_not_eq = True

                if is_not_eq:
                    return False
                
                else:
                    return True
        
        return False
    

    def __str__(self) -> str:
        string = "".join(map(str, self.matrix))
        description = f"Matrix{self.n}x{self.m}({string})"
        return description


    def __repr__(self) -> str:
        return self.__str__()
    
    
class Vector(Matrix):
    """Basic class of matrix

    Note:
        Initialize Vector object using **coordinates** or **list[coordinates]**
        or **list[list[coordinates]]**
    
    Example:
        There are different ways of initialization of same vector: 
        ``Vector(1, 2, 3)`` or ``Vector([1, 2, 3])`` or 
        ``Vector([[1], [2], [3]])``
    
    """
    def __init__(self, *args: Union[float,
                                    int, 
                                    'Vector', 
                                    list, 
                                    list[list]]) -> 'Vector':

        if not isinstance(args, (tuple, list)):
            raise InitializationException()
        
        first_arg = args[0]

        if not isinstance(first_arg, (float, int, list, Vector)):
            raise InitializationException()
        
        if isinstance(first_arg, (float, int, Vector)):

            for item in args:

                if not isinstance(item, type(first_arg)):
                    raise InitializationException()
                
            state = "coord"
        
        if isinstance(first_arg, list):

            if len(args) > 1:
                raise InitializationException()
            
            first_item = first_arg[0]
            
            for item in first_arg:

                if not isinstance(item, type(first_item)):
                    raise ClassArgumentsException(item)
            
            if isinstance(first_item, list):

                state = "column"

            if isinstance(first_item, (float, int, Vector)):

                state = "row"

        if state == "column":
            result = first_arg

        if state == "row":
            result = [[i] for i in first_arg]
        
        if state == "coord":
            result = [[i] for i in args]
        
        super().__init__(result)

    @staticmethod
    def _get_default_orthonormal_basis():

        basis_i = Vector(1, 0, 0)
        basis_j = Vector(0, 1, 0)
        basis_k = Vector(0, 0, 1)

        return [basis_i, basis_j, basis_k]


    def inverse(self):
        """"""
        raise RestictException()


    def determinant(self):
        """"""
        raise RestictException()
     

    def __getitem__(self, keys: Union[int,
                                tuple[int]]) -> Union[float, int, 'Vector']:
        """Getting coordinate of Vector object, based on getting element from 
        Matrix object.

        Note:
            To get coordinate from ``VectorObject`` use both of 
            ``VectorObject.[index]`` or ``VectorObject.[index, 0]``.

        Args:
            keys (Union[int, tuple[int]]): Indexes of getting element.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Union[int, float, Vector]: Coordinate of Vector object.

        """
        result = super().__getitem__(keys)

        if isinstance(result, Vector) and result.n == 1:
            result = result.matrix[0][0]
        
        return result


    def __setitem__(self, 
                    keys: Union[int, tuple[int]], 
                    value: Union[int, float, 'Vector']):
        """Setting coordinate of Vector object, based on setting element from 
        Matrix object.

        Note:
            To set coordinate from ``VectorObject`` use both of 
            ``VectorObject.[index] = coordinate`` or 
            ``VectorObject.[index, 0] = coordinate``.

        Args:
            keys (Union[int, tuple[int]]): Indexes of setting coordinate.
            value (Union[int, float, Vector]): Setting coordinate.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            ArgumentsException: Wrong argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.
            
        """
        if isinstance(keys, int):
            keys = (keys, 0)

        super().__setitem__(keys, value)


    def addition(self, other: 'Vector') -> 'Vector':
        """Adding vectors with same dimensions.

        Note:
            Alternative way of call this function use ``+`` as binary operator.

        Args:
            other (Vector): Adding vector.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Matrix: Result vector after addition.

        """
        if not isinstance(other, Vector):
            raise ClassArgumentsException(other)

        result = super().addition(other)

        return Vector(result.matrix)


    def multiplication(self, other: Union[float, int]) -> 'Vector':
        """Multipling vector and number.

        Note:
            Alternative way of call this function use ``*`` as binary operator.

        Args:
            other (Union[float, int]): Multipling number.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Vector: Result vector after multiplication.

        """
        if not isinstance(other, (float, int)):
            raise ClassArgumentsException(other)
        
        result = super().multiplication(other)

        return Vector(result.matrix)


    def subtraction(self, other: 'Vector') -> 'Vector':
        """Subtracting vector with same dimensions, based on adding and 
        multipling.

        Note:
            Alternative way of call this function use ``-`` as binary operator.

        Args:
            other (Vector): Subtracting vector.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Matrix: Result vector after subtraction.

        """
        if not isinstance(other, Vector):
            raise ClassArgumentsException(other)

        vector = self + (-1) * other

        return vector


    def scalar_product(self, other: 'Vector') -> float:
        """Scalar product of vectors with same dimensions in orthonormal basis.

        Note:
            Alternative way of call this function use ``%`` as binary operator.

        Args:
            other (Vector): Second vector for product.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            float: Result of scalar product.

        """
        if not isinstance(other, Vector):
            raise ClassArgumentsException(other)
        
        if self.n != other.n:
            raise DimensionArgumentsException()
        
        e_matrix = Matrix.identity(self.n)
        
        result = self.transpose() * e_matrix * Matrix(other.matrix)

        return round(result[0, 0], PRECISION)


    def vector_product(self,
                       other: 'Vector', 
                       basis: list['Vector'] = None) -> 'Vector':
        """Vector product of vectors with same dimensions in given basis, by 
        default in orthonormal.

        Note:
            Alternative way of call this function use ``**`` as binary operator.

        Args:
            other (Vector): Second vector for product.
            basis (list[Vector]): Basis for vector product. Default is None.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Vector: Resulting vector.

        """
        if basis is None:
            basis = Vector._get_default_orthonormal_basis()

        if not isinstance(other, Vector):
            raise ClassArgumentsException(other)
        
        if self.n != other.n != 3:
            raise DimensionArgumentsException()
        
        vector1_coord = self.transpose()[0].matrix
        vector2_coord = other.transpose()[0].matrix

        vector = Matrix([
            basis,
            vector1_coord,
            vector2_coord
        ]).determinant()
        
        return vector


    def transpose(self) -> 'Matrix':
        """Transposing vector.

        Returns:
            Matrix: Result matrix after transposing.

        """
        return Matrix(list(map(list, zip(*self.matrix))))


    def length(self) -> float: 
        """Returns length of the vector."""    
        return round(math.sqrt(self.scalar_product(self)), PRECISION)


    def normalize(self) -> 'Vector':
        """Returns vector with same direction and length equals one.

        Raises:
            ArgumentsException: Wrong argument for method.

        Returns:
            Vector: Resulting vector.

        """
        v = Vector(self.matrix)
        lenght = v.length()

        if lenght == 0:
            raise ArgumentsException()

        matrix = [[0] for _ in range(self.n)]
        for i in range(self.n):
            matrix[i][0] = v.matrix[i][0]
            matrix[i][0] = matrix[i][0] / lenght
        
        vector = Vector(matrix)
        
        return vector


    def dim(self)-> int:
        """Returns dimension of vector"""
        return self.n
    

    def rotate(self, *angles) -> 'Vector':
        """Rotating vector in three dimensions using tait Bryan Matrix.

        Returns:
            Vector: Resulting vector

        """
        tait = Matrix.tait_bryan_matrix(*angles)

        result = tait * Matrix(self.matrix)

        return Vector(result.matrix)


    def __add__(self, other):
        return self.addition(other)


    def __radd__(self, other):
        return self.addition(other)


    def __mul__(self, other):
        return self.multiplication(other)


    def __rmul__(self, other):
        return self.multiplication(other)


    def __sub__(self, other):
        return self.subtraction(other)


    def __rsub__(self, other):
        return other.subtraction(self)


    def __mod__(self, other):
        return self.scalar_product(other)


    def __rmod__(self, other):
        return other.scalar_product(self)


    def __pow__(self, other):
        return self.vector_product(other)


    def __rpow__(self, other):
        return other.vector_product(self)
    

    def __eq__(self, other: 'Vector') -> bool:

        if isinstance(other, Vector) and self.__class__ == other.__class__:
            return self.matrix == other.matrix
        
        return False
    
    def __str__(self) -> str:

        string = ", ".join([str(i[0]) for i in self.matrix])
        description = f"Vector{self.n}({string})"
        return description


class VectorSpace:
    """Basic class of vector space
    
    Note:
        Initializing VectorSpace object using list of basic ``Vector``s
    
    """
    def __init__(self, basis: list[Vector]):

        if not isinstance(basis, list):
            raise InitializationException()
        
        if not isinstance(basis[0], Vector):
            raise InitializationException()
        
        first_dim = basis[0].dim()
        
        for item in basis:

            if not isinstance(item, Vector):
                raise InitializationException()
            
            if item.dim() != first_dim:
                raise InitializationException()

        self.n = len(basis)
        self.basis = basis


    def scalar_product(self, v1: Vector, v2: Vector) -> float:
        """Scalar product of vectors with same dimensions in VectorSpace basis.

        Args:
            v1 (Vector): First vector for product.
            v2 (Vector): Second vector for product.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            float: Result of scalar product.

        """
        if not isinstance(v1, Vector):
            raise ClassArgumentsException(v1)
        
        if not isinstance(v2, Vector):
            raise ClassArgumentsException(v2)
        
        if v1.n != v2.n != self.n:
            raise DimensionArgumentsException()
        
        gram = Matrix.gram(self.n, self.basis) 
        
        result = v1.transpose() * gram * Matrix(v2.matrix)

        return float(result[0, 0])


    def vector_product(self, v1: Vector, v2: Vector) -> Vector:
        """Vector product of three dimensional vectors in VectorSpace basis.

        Args:
            v1 (Vector): First vector for product.
            v2 (Vector): Second vector for product.

        Raises:
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Vector: Resulting vector.

        """
        if self.n != 3:
            raise DimensionArgumentsException()
        
        if v1.n != v2.n != 3:
            raise DimensionArgumentsException()
        
        v1_coord = v1.transpose()[0].matrix
        v2_coord = v2.transpose()[0].matrix
        
        result = Matrix([
            [self.basis[1]**self.basis[2],
             self.basis[2]**self.basis[0], 
             self.basis[0]**self.basis[1]],
            v1_coord,
            v2_coord

        ]).determinant()

        return result


    def as_vector(self, point: 'Point') -> Vector:
        """Transforming point into radius vector.

        Args:
            point (Point): Point for transform.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Vector: Resulting vector

        """
        if not isinstance(point, Point):
            raise ClassArgumentsException(point)
        
        if point.n != self.n:
            raise DimensionArgumentsException()

        result = [[Vector(point.matrix) % e] for e in self.basis]

        return Vector(result)
    

    def __str__(self) -> str:

        string = ", ".join([v.__str__() for v in self.basis])
        description = f"VectorSpace{self.n}({string})"
        return description


class Point(Vector):
    """Basic class of point
    
    Note:
        Initializing the same way as ``Vector``

    """
    def multiplication(self):
        """"""
        raise RestictException()


    def scalar_product(self):
        """"""
        raise RestictException()


    def vector_product(self):
        """"""
        raise RestictException()


    def length(self):
        """"""
        raise RestictException()


    def determinant(self):
        """"""
        raise RestictException()


    def addition(self, other: 'Vector') -> 'Point':
        """Translating point on given vector with positive direction.

        Note:
            Alternative way of call this function use ``+`` as binary operator.

        Args:
            other (Vector): Translating vector.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Point: Translated point

        """
        if not isinstance(other, Vector):
            raise ClassArgumentsException(other)
        
        if self.n != other.n:
            raise DimensionArgumentsException()
        
        result = super().addition(other)

        return Point(result.matrix)


    def subtraction(self, other: 'Vector') -> 'Point':
        """Translating point on given vector with negative direction.

        Note:
            Alternative way of call this function use ``-`` as binary operator.

        Args:
            other (Vector): Translating vector.

        Raises:
            ClassArgumentsException: Wrong type of argument for method.
            DimensionArgumentsException: Wrong dimension of arguments.

        Returns:
            Point: Translated point

        """
        if not isinstance(other, Vector):
            raise ClassArgumentsException(other)
        
        if self.n != other.n:
            raise DimensionArgumentsException()
        
        result = super().subtraction(other)

        return Point(result.matrix)
    

    def __add__(self, other):
        return self.addition(other)


    def __sub__(self, other):
        return self.subtraction(other)


    def __str__(self) -> str:

        string = ", ".join([str(i[0]) for i in self.matrix])
        description = f"Point{self.n}({string})"
        return description


class CoordinateSystem:
    """Basic class of coordinate system.
    
    Note:
        Initializing using ``VectorSpace`` (as basis) and ``Point`` 
        (as initial point)

    """
    def __init__(self, basis: VectorSpace, init_point: Point):

        if not isinstance(basis, VectorSpace): 
            raise InitializationException()
        
        if not isinstance(init_point, Point):
            raise InitializationException()
        
        if basis.n != init_point.n:
            raise InitializationException()

        self.basis = basis
        self.init_point = init_point
    

    def __str__(self):
        string = ", ".join([self.init_point.__str__(), self.basis.__str__()])
        description = f"CoordinateSystem({string})"
        return description
