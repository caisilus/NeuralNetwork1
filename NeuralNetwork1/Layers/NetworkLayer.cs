using System;
using Accord;

namespace NeuralNetwork1
{
    public class NetworkLayer
    {
        public NetworkLayer Previous { get; set; }
        public NetworkLayer Next { get; set; }
        
        protected Matrix X;

        public NetworkLayer(NetworkLayer previous, NetworkLayer next)
        {
            Previous = previous;
            Next = next;
            X = null;
        }

        public Matrix Forward(Matrix layerInput)
        {
            X = layerInput;
            Matrix layerResult = LayerResult(layerInput);
            if (Next != null)
            {
                return Next.Forward(layerResult);
            }

            return layerResult;
        }

        protected virtual Matrix LayerResult(Matrix layerInput)
        {
            throw new NotImplementedException("Layer result not implemented");
        }
        
        public void Backward(Matrix dOutput, double learningRate)
        {
            Matrix dX = DifferentialOverX(dOutput);
            UpdateWeights(dOutput, learningRate);
            Previous?.Backward(dX, learningRate);
        }

        protected virtual Matrix DifferentialOverX(Matrix dOutput)
        {
            throw new NotImplementedException("Differential Over X is not implemented");
        }

        // Does nothing for layers without weights
        protected virtual void UpdateWeights(Matrix dOutput, double learningRate)
        {
        }
    }
}