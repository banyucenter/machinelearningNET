using System;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace MachineApp
{
    class Program
    {
        public class CarData
        {
            [LoadColumn(0)]
            public float Mpg;

            [LoadColumn(1)]
            public float Cylinders;

            [LoadColumn(2)]
            public float Displacement;

            [LoadColumn(3)]
            public float Horsepower;

            [LoadColumn(4)]
            public float Weight;

            [LoadColumn(5)]
            public float Acceleration;

            [LoadColumn(6)]
            public float Modelyear;

            [LoadColumn(7)]
            public float Origin;

            [LoadColumn(8)]
            public string Label;


        }

        // IrisPrediction is the result returned from prediction operations
        public class CarPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<CarData>(path: "auto-mpg.txt", hasHeader: false, separatorChar: ',');
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features","Mpg", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Modelyear", "Origin"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train your model based on the data set  
            var model = pipeline.Fit(trainingDataView);

            // STEP 5: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.CreatePredictionEngine<CarData, CarPrediction>(mlContext).Predict(
                new CarData()
                {
                    Mpg = 18.0f,
                    Cylinders = 8f,
                    Displacement = 307.0f,
                    Horsepower = 165.0f,
                    Weight = 3704f,
                    Acceleration = 12.0f,
                    Modelyear = 70f,
                    Origin= 1f,

                });

            Console.WriteLine($"Predicted Car Name is: {prediction.PredictedLabels}");

            Console.WriteLine("Press any key to exit....");
            Console.ReadLine();
        }


    }
}
