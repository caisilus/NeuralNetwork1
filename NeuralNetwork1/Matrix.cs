using System;

namespace NeuralNetwork1
{
    public class Matrix
    {
        public double[,] Data;
        public int RowsCount { get; }
        public int ColumnsCount { get; }
        
        public Matrix(double[,] d)
        {
            Data = d;
            RowsCount = d.GetLength(0);
            ColumnsCount = d.GetLength(1);
        }

        public Matrix(int rowsCount, int columnsCount)
        {
            Data = new double[rowsCount, columnsCount];
            RowsCount = rowsCount;
            ColumnsCount = columnsCount;
        }

        public static Matrix FromJaggedArray(double[][] jaggedArray)
        {
            Matrix m = new Matrix(jaggedArray.Length, jaggedArray[0].Length);
            for (int i = 0; i < jaggedArray.Length; i++)
            {
                if (jaggedArray[i].Length != m.ColumnsCount)
                {
                    throw new ArgumentException("Jagged array isn't matrix: inconsistent row length");
                }

                for (int j = 0; j < jaggedArray[i].Length; j++)
                {
                    m[i, j] = jaggedArray[i][j];
                }
            }

            return m;
        }
        
        public Vector[] Rows()
        {
            Vector[] rows = new Vector[RowsCount];
            for (int i = 0; i < RowsCount; i++)
            {
                rows[i] = Row(i);
            }

            return rows;
        }

        public Vector[] Columns()
        {
            Vector[] columns = new Vector[ColumnsCount];
            for (int i = 0; i < ColumnsCount; i++)
            {
                columns[i] = Column(i);
            }

            return columns;
        }

        public Vector Row(int i)
        {
            Vector row = new Vector(ColumnsCount);
            for (int j = 0; j < ColumnsCount; j++)
            {
                row[j] = Data[i, j];
            }

            return row;
        }
        
        public Vector Column(int i)
        {
            Vector column = new Vector(RowsCount);
            for (int j = 0; j < RowsCount; j++)
            {
                column[j] = Data[j, i];
            }

            return column;
        }

        public double this[int i, int j]
        {
            get => Data[i, j];
            set => Data[i, j] = value;
        }
        
        public Matrix Dot(Matrix other)
        {
            if (ColumnsCount!= other.RowsCount)
            {
                throw new Exception($"Dimensions does not agree for dot: ({RowsCount},{ColumnsCount}) " +
                                    $"and ({other.RowsCount},{other.ColumnsCount})");
            }
            
            Matrix dotResult = new Matrix(RowsCount, other.ColumnsCount);
            
            for (int i = 0; i < RowsCount; i++)
            {
                for (int j = 0; j < other.ColumnsCount; j++)
                {
                    for (int k = 0; k < ColumnsCount; k++)
                    {
                        dotResult[i, j] += Data[i, k] * other[k, j];
                    }
                }
            }
            return dotResult;
        }

        public Matrix Transpose()
        {
            Matrix transposed = new Matrix(ColumnsCount, RowsCount);
            for (int i = 0; i < RowsCount; i++)
            {
                for (int j = 0; j < ColumnsCount; j++)
                {
                    transposed[j, i] = Data[i, j];
                }
            }
            return transposed;
        }

        public Vector SumRows()
        {
            Vector result = new Vector(ColumnsCount);
            for (int i = 0; i < ColumnsCount; i++)
            {
                for (int j = 0; j < RowsCount; j++)
                {
                    result[i] += Data[j, i];
                }
            }
            return result;
        }

        public Vector SumColumns()
        {
            Vector result = new Vector(RowsCount);
            for (int i = 0; i < RowsCount; i++)
            {
                for (int j = 0; j < ColumnsCount; j++)
                {
                    result[i] += Data[i, j];
                }
            }
            return result;
        }
        
        public static Matrix operator +(Matrix leftOperand, Matrix rightOperand)
        {
            if (leftOperand.RowsCount != rightOperand.RowsCount || leftOperand.ColumnsCount != rightOperand.ColumnsCount)
            {
                throw new Exception("Dimensions does not agree for addition: " +
                                    $"({leftOperand.RowsCount},{leftOperand.ColumnsCount}) " +
                                    $"and ({rightOperand.RowsCount},{rightOperand.ColumnsCount})");
            }
            Matrix result = new Matrix(leftOperand.RowsCount, leftOperand.ColumnsCount);
            for (int i = 0; i < leftOperand.RowsCount; i++)
            {
                for (int j = 0; j < leftOperand.ColumnsCount; j++)
                {
                    result[i, j] = leftOperand.Data[i, j] + rightOperand.Data[i, j];
                }
            }
            
            return result;
        }

        public static Matrix operator -(Matrix leftOperand, Matrix rightOperand)
        {
            if (leftOperand.RowsCount != rightOperand.RowsCount || leftOperand.ColumnsCount != rightOperand.ColumnsCount)
            {
                throw new Exception("Dimensions does not agree for subtraction" +
                                    $"({leftOperand.RowsCount},{leftOperand.ColumnsCount}) " +
                                    $"and ({rightOperand.RowsCount},{rightOperand.ColumnsCount})");
            }
            Matrix result = new Matrix(leftOperand.RowsCount, leftOperand.ColumnsCount);
            for (int i = 0; i < leftOperand.RowsCount; i++)
            {
                for (int j = 0; j < leftOperand.ColumnsCount; j++)
                {
                    result[i, j] = leftOperand.Data[i, j] - rightOperand.Data[i, j];
                }
            }
            
            return result;
        }

        public static Matrix operator -(Matrix mat)
        {
            return -1 * mat;
        }
        
        public static Matrix operator *(double scalar, Matrix mat)
        {
            Matrix multiplicationResult = new Matrix(mat.RowsCount, mat.ColumnsCount);
            for (int i = 0; i < mat.RowsCount; i++)
            {
                for (int j = 0; j < mat.ColumnsCount; j++)
                {
                    multiplicationResult[i, j] = mat.Data[i, j] * scalar;
                }
            }

            return multiplicationResult;
        }
        
        public static Matrix operator/ (Matrix dividend, double divider)
        {
            Matrix divisionResult = new Matrix(dividend.RowsCount, dividend.ColumnsCount);
            for (int i = 0; i < dividend.RowsCount; i++)
            {
                for (int j = 0; j < dividend.ColumnsCount; j++)
                {
                    divisionResult[i, j] = dividend.Data[i, j] / divider;
                }
            }
            return divisionResult;
        }
    }
}