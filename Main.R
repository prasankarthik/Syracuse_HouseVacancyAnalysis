#Loading packages and Data
install.packages("data.table")
library("data.table")
library("e1071")
library("caret")
library("randomforest")
library("mlbench")

#Reading the data
Data <- read.csv("finalData1.csv")

#Factorizing the variables

Data$VacantBuilding <- as.factor(Data$VacantBuilding)
Data$Num.Open.Violations <- as.factor(Data$Num.Open.Violations)
Data$AssessedValue <- as.factor(Data$AssessedValue)
Data$LandUse <- as.factor(Data$LandUse)
Data$YearBuilt <- as.factor(Data$YearBuilt)

#Variable description
#VacantBuilding - Varaible denoting whether a building is vacant or not
#Num.Open.Violations - Number of open violations on the building
#AssessedValue - Estimated value of the building
#LandUse - The land used up by the building
#YearBuilt - The year on which the building was buildt
#Owner Occupied - The probability that the owner itself is occupying the house(It is in probability since this data is not available and we calculated the probability from census data)
#Total taxes owed - The taxes owed by the building
#H1-3 - The probability that the house may be occupied by 1-3 people(It is in probability since this data is not available and we calculated the probability from census data)
#H4- The probability that the house may be occupied by more than 3 people(It is in probability since this data is not available and we calculated the probability from census data)
#Total crimes- The total number of crimes that happened over a period of three years in the street of the house

--------------------------------------------------------------------------------------------------------------------------------------------------------
#Feature Engineering - Manipulating the variables

#Bucketing owner occupied column

Vector1 <- Data$Owner_Occupied
Vector1 <- as.numeric(Vector1)

for(i in 1:14907)
{
  if(Vector1[i]<0.25)
  {
     Vector1[i] <- "Very Low"
  }
  else if(Vector1[i]<0.5)
  {
    Vector1[i] <- "Low"
  }
  else if(Vector1[i]<0.75)
  {
    Vector1[i] <- "High"
  }
  else if(Vector1[i]>0.74)
  {
    Vector1[i] <- "Very High"
  }
    
}
Owner_occupied_Cat <- as.factor(Vector1)

#Bucketinzing H1-3 variable

Vector2 <- Data$H_.1.3.P
Vector2 <- as.numeric(Vector2)

for(i in 1:14907)
{
  if(Vector2[i]<0.25)
  {
    Vector2[i] <- "Very Low"
  }
  else if(Vector2[i]<0.5)
  {
    Vector2[i] <- "Low"
  }
  else if(Vector2[i]<0.75)
  {
    Vector2[i] <- "High"
  }
  else if(Vector2[i]>0.74)
  {
    Vector2[i] <- "Very High"
  }
  
}
H1_3_Cat <- as.factor(Vector2)

#Bucketizing H4

Vector3 <- Data$H.4..P
Vector3 <- as.numeric(Vector3)

for(i in 1:14907)
{
  if(Vector3[i]<0.25)
  {
    Vector3[i] <- "Very Low"
  }
  else if(Vector3[i]<0.5)
  {
    Vector3[i] <- "Low"
  }
  else if(Vector3[i]<0.75)
  {
    Vector3[i] <- "High"
  }
  else if(Vector3[i]>0.74)
  {
    Vector3[i] <- "Very High"
  }
  
}
H4_Cat <- as.factor(Vector3)

#Bucketizing Total taxes owed
Vector4 <- Data$Total.Taxes.Owed
Vector4 <- as.numeric(Vector4)
for(i in 1:14907)
{
  if(Vector4[i]<2000)
  {
    Vector4[i] <- 0
  }
  else if(Vector4[i]<4000)
  {
    Vector4[i] <- 2000
  }
  else if(Vector4[i]<6000)
  {
    Vector4[i] <- 4000
  }
  else if(Vector4[i]<8000)
  {
    Vector4[i] <- 6000
  }
  else if(Vector4[i]<10000)
  {
    Vector4[i] <- 8000
  }
  else if(Vector4[i]<20000)
  {
    Vector4[i] <- 10000
  }
  else if(Vector4[i]<30000)
  {
    Vector4[i] <- 20000
  }
  else if(Vector4[i]<40000)
  {
    Vector4[i] <- 30000
  }
  else if(Vector4[i]<50000)
  {
    Vector4[i] <- 40000
  }
  else if(Vector4[i]<100000)
  {
    Vector4[i] <- 50000
  }
  else if(Vector4[i]>99999)
  {
    Vector4[i] <- 100000
  }

}

Total_taxes_Cat <- as.factor(Vector4)

#Bucketizing crimes

Vector5 <- Data$Total_crimes
Vector5 <- as.numeric(Vector5)
Vector5[11759]

for(i in 1:14907)
{
  if(Vector5[i]<6)
  {
    Vector5[i] <- 5
  }
  else if(Vector5[i]<16)
  {
    Vector5[i] <- 15
  }
  else if(Vector5[i]<30)
  {
    Vector5[i] <- 30
  }
  else if(Vector5[i]>29)
  {
    Vector5[i] <- 45
  }
  
}
Total_crimes_Cat <- as.factor(Vector5)

#Bukcketizing Num.of.open Violations

for(i in 1:nrow(Data))
{
  if(Data$Num.Open.Violations[i]==0)
  {
    Data$Num.Open.Violations[i] <- 0 
  }
  else if(Data$Num.Open.Violations[i]<6)
  {
    Data$Num.Open.Violations[i] <- 5
  }
  else if(Data$Num.Open.Violations[i]<11)
  {
    Data$Num.Open.Violations[i] <- 10
  }
  else 
  {
    Data$Num.Open.Violations[i] <- 25
  }
}
Data$Num.Open.Violations <- as.factor(Data$Num.Open.Violations)

------------------------------------------------------------------------------------------------------------------------------------------------------

# Combining the bucketized  fields with the dataframe
Data <- data.frame(Data, Total_taxes_Cat,Total_crimes_Cat, H1_3_Cat, H4_Cat,Owner_occupied_Cat)

# Preparing the training data by taking equal number of yes and no values for the predicted variable Vacant Building so that the model is not biased

data_yes <- Data[Data$VacantBuilding== "Y",]
data_no <- Data[Data$VacantBuilding == "N",]


#Randomizing the rows
data_no <- data_no[sample(nrow(data_no)),]
data_no_equal <- data_no[1:746,]

data_final <- rbind(data_yes,data_no_equal)
data_final <- data_final[sample(nrow(data_final)),]

-------------------------------------------------------------------------------------------------------------------------------------------------------

# FEATURE SELECTION

# Ordering the attributes based on the importance level

# For the model to be efficient - variables with near zero variances have to be removed
data_final <- data_final[,-c(1,8,11)]



# Deciding on the number of features to keep using Recursive Feature elimination
Prediction <- data_final$VacantBuilding
data_final <- data_final[,-3]

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run the RFE algorithm
results <- rfe(data_final, Prediction, sizes=c(1:8), rfeControl=control)

# summarize the results
print(results)

# list the chosen features
predictors(results)

# plot the results
plot(results, type=c("g", "o"))

#RFE suggests that we keep only "Num.Open.Violations" and "Water Service"

# Allocating weight to each feature using svm
# Prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(VacantBuilding~., data=data_final, method="svmLinear", gamma=0.05, preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=TRUE)
# summarize importance
print(importance)
# plot importance
plot(importance)

#The order of important features after scaling from svmLinear model were

#Importance
#Num.Open.Violations     100.00
#WaterService             76.47
#AssessedValue            27.53
#YearBuilt                21.18
#ZIP                      12.92
#H1_3_Cat                  7.22
#Total_crimes_Cat          2.66
#Owner_occupied_Cat        0.00

#This proves that Num.open violations and Water Service are the important variables

--------------------------------------------------------------------------------------------------------------------------------------------------------

#Building SVM model using train data and validating it on the test data
#Splitting Train and test data 
data_test <- data_final[1:1268,]
data_train <- data_final[1269:1492,]

#Building the svm model using only Num.Open.Violation and Water Service
svm_model <- svm(VacantBuilding ~ Num.Open.Violations + WaterService , gamma=0.01, data= data_train)
nbd.pred <- predict(svm_model, data_test)

#Calculating accuracy
accuracy <- data.frame(table(data_test$VacantBuilding,nbd.pred))
accuracy_percent <- ((accuracy[1,3]+accuracy[4,3])/nrow(data_test))*100

#Using the data split accuracy indicator the accuracy is 93.5%

-----------------------------------------------------------------------------------------------------------------------------------------------

#Now using the repeated k fold cross validation to build the model
#and predict accuracy


# define training control
train_control <- trainControl(method="repeatedcv", number= 5, repeats= 3)
# By varying the "number" - the number of data points for each sample
# and "repeats" - the number of times the whole process has to get repeated
# we can play around and find the best model

# train the model
model <- train(VacantBuilding~Num.Open.Violations+ WaterService, data=data_final, trControl=train_control, method="nb")

# summarize results
print(model)

# Using the repeated k fold cross validation approach 
# the accuracy of svm model was 92.4%

