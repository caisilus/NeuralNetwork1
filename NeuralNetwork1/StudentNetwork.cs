using System;
using System.Diagnostics;
using System.IO;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private NetworkLayer _firstLayer = null;
        private NetworkLayer _lastLayer = null;
        private double _learningRate = 0.5;
        
        //  Секундомер спортивный, завода «Агат», измеряет время пробегания стометровки, ну и время затраченное на обучение тоже умеет
        public Stopwatch stopWatch = new Stopwatch();
        
        public StudentNetwork(int[] structure)
        {
            for (int i = 0; i < structure.Length - 1; i++)
            {
                var linearLayer = new LinearLayer(structure[i], structure[i + 1], _lastLayer, null);
                if (_lastLayer != null)
                {
                    _lastLayer.Next = linearLayer;
                } 
                var sigmoidLayer = new SigmoidLayer(linearLayer, null);
                linearLayer.Next = sigmoidLayer;
                if (_firstLayer == null)
                {
                    _firstLayer = linearLayer;
                }

                _lastLayer = sigmoidLayer;
            }
        }

        double TrainStep(Sample sample, int countSamples=1)
        {
            double[] prediction = Compute(sample.Input);
            sample.ProcessPrediction(prediction);
            double[] error = sample.Error;
            double loss = sample.EstimatedError() / sample.Output.Length /* / countSamples */;
            _lastLayer.Backward((2.0 / sample.Output.Length / countSamples) * new Vector(error), _learningRate);
            return loss;
        }
        
        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int countIterations = 0;
            double loss = TrainStep(sample);
            while (loss > acceptableError)
            {
                loss = TrainStep(sample); 
                countIterations++;
            }

            return countIterations;
        }
        
        public double SamplesSetTrainStep(SamplesSet samplesSet)
        {
            double loss = 0;
            for (int i = 0; i < samplesSet.Count; i++)
            {
                Sample sample = samplesSet[i];
                double sampleLoss = TrainStep(sample, samplesSet.Count);
                loss += sampleLoss;
            }

            return loss;
        }
        
        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            int currentEpoch = 0;
            double loss = double.PositiveInfinity;
            
            StreamWriter errorsFile = File.CreateText("errors.csv");

            stopWatch.Restart();

            while (currentEpoch < epochsCount && loss > acceptableError)
            {
                currentEpoch++;

                loss = SamplesSetTrainStep(samplesSet);
#if DEBUG
                errorsFile.WriteLine(loss);
#endif
                OnTrainProgress((currentEpoch * 1.0) / epochsCount, loss, stopWatch.Elapsed);
            }

#if DEBUG
            errorsFile.Close();
#endif
            OnTrainProgress(1.0, loss, stopWatch.Elapsed);

            stopWatch.Stop();

            return loss;
        }

        protected override double[] Compute(double[] input)
        {
            Vector inputVector = new Vector(input); 
            Vector result = _firstLayer.Forward(inputVector).Row(0);
            return result.ToDoubleArray();
        }
    }
}