using System;

namespace NeuralNetwork1
{
    public class LinearLayer: NetworkLayer
    {
        private int _inNeuronsCount;
        private int _outNeuronsCount;

        private Matrix _w;
        private Vector _b;
        
        public LinearLayer(int inNeuronsCount, int outNeuronsCount, NetworkLayer previous, NetworkLayer next): 
            base(previous, next)
        {
            _inNeuronsCount = inNeuronsCount;
            _outNeuronsCount = outNeuronsCount;
            InitWeights(-0.005, 0.005);
            InitBiased();
        }

        private void InitWeights(double from, double to)
        {
            Random random = new Random();
            _w = new Matrix(_inNeuronsCount, _outNeuronsCount);
            _b = new Vector(_outNeuronsCount);
            for (int i = 0; i < _inNeuronsCount; i++)
            {
                for (int j = 0; j < _outNeuronsCount; j++)
                {
                    _w[i, j] = random.NextDouble() * (to - from) + from;
                }
            }
        }

        private void InitBiased()
        {
            for (int i = 0; i < _outNeuronsCount; i++)
            {
                _b[i] = 0;
            }
        }

        protected override Matrix LayerResult(Matrix layerInput)
        {
            Matrix aX = layerInput.Dot(_w);
            return aX + _b.StackToAgreeOnSum(aX);
        }
        
        protected override Matrix DifferentialOverX(Matrix dOutput)
        {
            return dOutput.Dot(_w.Transpose());
        }
        
        protected override void UpdateWeights(Matrix dOutput, double learningRate)
        {
            Vector dB = dOutput.SumRows();
            Matrix dW = X.Transpose().Dot(dOutput);
            _w = _w - (learningRate * dW);
            _b = (_b - (learningRate * dB)).Row(0);
        }
    }
}