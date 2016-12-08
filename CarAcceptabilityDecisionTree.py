import os
import sys


# Path for spark source folder
os.environ['SPARK_HOME']="Spark Home Directory"

# Append pyspark  to Python Path
sys.path.append("~/spark-2.0.2-bin-hadoop2.7/python")

#Importing Spark Related packages
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer , VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Sklearn Related packages
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt


#Creating Spark Session
spark = SparkSession.builder.master("local").appName("CarAcceptability Model").getOrCreate()


#Read input file
#Header Information
#buying,maint,doors,persons,lug_boot,safety,acceptance

inputData = spark.read.csv("Input DataSet", header=True)

#inputData.printSchema()

#Filtered input good and v-good labels to make it useful for binary classification
filteredInputData = inputData.where((inputData.acceptance == "unacc") | (inputData.acceptance == "acc"))

print filteredInputData.count()
#Creating Indexers
buyingIndexer = StringIndexer(inputCol="buying", outputCol="buyingIndexer").fit(filteredInputData).transform(filteredInputData)
maintIndexer = StringIndexer(inputCol="maint", outputCol="maintIndexer").fit(buyingIndexer).transform(buyingIndexer)
lugBootIndexer = StringIndexer(inputCol="lug_boot", outputCol="lugBootIndexer").fit(maintIndexer).transform(maintIndexer)
safetyIndexer = StringIndexer(inputCol="safety", outputCol="safetyIndexer").fit(lugBootIndexer).transform(lugBootIndexer)
doorsIndexer = StringIndexer(inputCol="doors", outputCol="doorsIndexer").fit(safetyIndexer).transform(safetyIndexer)
finalIndexer = StringIndexer(inputCol="persons", outputCol="personsIndexer").fit(doorsIndexer).transform(doorsIndexer)


#finalIndexer.show()

#Adding all the columns to the vector assembler
assembler = VectorAssembler(inputCols=["buyingIndexer", "maintIndexer", "doorsIndexer" ,
                                       "personsIndexer", "lugBootIndexer", "safetyIndexer"], outputCol="features")


#New DataDf with new feature names
newDfInputData = assembler.transform(finalIndexer)


#Indexing the label column
labelIndexer = StringIndexer(inputCol="acceptance", outputCol="indexedacceptance").fit(newDfInputData)


#Splitting the dataset into training and test for the model
(trainingData, testData) = newDfInputData.randomSplit([0.75, 0.25])

trainingData.show()

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedacceptance", featuresCol="features")


# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, dt])


# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedacceptance").show(5)




#PLotting ROC Curves
actual = []
prediction = []


for acc, predict in predictions.select("prediction", "indexedacceptance").collect():
    actual.append(acc)
    prediction.append(predict)



false_positive_rate, true_positive_rate, thresholds = roc_curve(actual,prediction)
roc_auc = auc(false_positive_rate, true_positive_rate)

print confusion_matrix(actual, prediction)

#Plotting the Decision trees using matplotlib
plt.title('Receiver Operating Characteristic Decision Trees')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#Evaluating the ROC
evaluator = BinaryClassificationEvaluator(
    labelCol="indexedacceptance", rawPredictionCol="prediction", metricName="areaUnderROC")

areaUnderROC = evaluator.evaluate(predictions)

print("Area Under ROC  = %g " % (areaUnderROC))

