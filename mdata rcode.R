# Multinomial Logit Model
# packages required
require('mlogit')
require('nnet')

#In built dataset

Mdata <- read_csv("D:/Modules/Module 10 - Multinominal regression/mdata.csv")
head(Mdata)
tail(Mdata)
View(Mdata)
Mdata1 <- Mdata[,c(6:10)]
View(Mdata1)
table(Mdata1$prog) # tabular representation of the Y categories



Mode.prog <- multinom(prog ~ read+write+math+science, data=Mdata1)
summary(Mode.prog)

Mode$choice  <- relevel(Mode$choice, ref= "carpool")  # change the baseline level

##### Significance of Regression Coefficients###
z <- summary(Mode.prog)$coefficients / summary(Mode.prog)$standard.errors
p_value <- (1-pnorm(abs(z),0,1))*2

summary(Mode.prog)$coefficients
p_value

# odds ratio 
exp(coef(Mode.prog))

# predict probabilities
prob <- fitted(Mode.prog)
prob

# Find the accuracy of the model

class(prob)
prob <- data.frame(prob)
View(prob)
prob["pred"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

pred_name <- apply(prob,1,get_names)
?apply
prob$pred <- pred_name
View(prob)

# Confusion matrix
table(pred_name,Mdata$prog)

# confusion matrix visualization
barplot(table(pred_name,Mdata$prog),beside = T,col=c("red","lightgreen","blue","orange"),legend=c("academic","general","vocation"),main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")


# Accuracy 
mean(pred_name==Mdata$prog) 
