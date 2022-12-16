using System;

namespace NeuralNetwork1
{
    public class SigmoidLayer: NetworkLayer
    {
        private double _alpha;
        
        public SigmoidLayer(NetworkLayer previous, NetworkLayer next, double alpha = 1.0) : base(previous, next)
        {
            _alpha = alpha;
        }
        
        protected override Matrix LayerResult(Matrix layerInput)
        {
            Matrix sigmoidResult = new Matrix(layerInput.RowsCount, layerInput.ColumnsCount);
            for (int i = 0; i < layerInput.RowsCount; i++)
            {
                for (int j = 0; j < layerInput.ColumnsCount; j++)
                {
                    sigmoidResult[i, j] = Sigmoid(layerInput[i, j]);
                }
            }

            return sigmoidResult;
        }

        private double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-2 * _alpha * x));
        }
        
        protected override Matrix DifferentialOverX(Matrix dOutput)
        {
            if (X == null)
            {
                throw new Exception("X is not cached after forward pass");
            }
            Matrix differentialResult = new Matrix(dOutput.RowsCount, dOutput.ColumnsCount);
            for (int i = 0; i < dOutput.RowsCount; i++)
            {
                for (int j = 0; j < dOutput.ColumnsCount; j++)
                {
                    differentialResult[i, j] = 2 * _alpha * Sigmoid(X[i, j]) * (1 - Sigmoid(X[i, j])) * dOutput[i, j];
                }
            }

            return differentialResult;
        }
    }
}