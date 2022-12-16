using System;

namespace NeuralNetwork1
{
    public class Vector: Matrix
    {
        public Vector(double[] vectorData) : base(VectorDataToMatrixData(vectorData))
        {
        }

        public Vector(int size) : base(1, size)
        {
        }

        private static double[,] VectorDataToMatrixData(double[] vectorData)
        {
            double[,] matrixData = new double[1, vectorData.Length];
            for (int i = 0; i < vectorData.Length; i++)
            {
                matrixData[0, i] = vectorData[i];
            }

            return matrixData;
        }

        public double[] ToDoubleArray()
        {
            double[] result = new double[ColumnsCount];
            for (int i = 0; i < ColumnsCount; i++)
            {
                result[i] = this[i];
            }

            return result;
        }
        
        public double this[int i]
        {
            get => Data[0, i];
            set => Data[0, i] = value;
        }

        public Matrix StackToAgreeOnSum(Matrix matrixToAgreeWith)
        {
            if (matrixToAgreeWith.ColumnsCount == ColumnsCount)
            {
                return StackRows(matrixToAgreeWith.RowsCount);
            }

            if (matrixToAgreeWith.RowsCount == ColumnsCount)
            {
                return StackColumns(matrixToAgreeWith.ColumnsCount);
            }

            throw new ArgumentException("There is no axis matrix and vector agree");
        }
        
        public Matrix StackRows(int count)
        {
            if (count < 1)
            {
                throw new ArgumentException("count cannot be less than 1");
            }
            
            if (count == 1)
                return this;
            
            Matrix result = new Matrix(count, ColumnsCount);
            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < ColumnsCount; j++)
                {
                    result[i, j] = this[0, j];
                }
            }

            return result;
        }
        
        public Matrix StackColumns(int count)
        {
            if (count < 1)
            {
                throw new ArgumentException("count cannot be less than 1");
            }
            
            if (count == 1)
                return this;
            
            Matrix result = new Matrix(ColumnsCount, count);
            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < ColumnsCount; j++)
                {
                    result[j, i] = this[0, j];
                }
            }

            return result;
        }
    }
}