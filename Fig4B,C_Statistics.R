rm(list=ls())
ls()

#Load libraries
library(glmmTMB)
library(lsmeans)

## Fig. 4B
# Load dataframe
df <- read.csv("https://raw.githubusercontent.com//casey-e/Casey-et-al-2022-b/main/Fig4/Fig4B_Statistics.csv")
attach(df)
View(df)
#Transform to factor
df$Treatment<-factor(df$Treatment,levels=c("Vehicle","Cocaine"))
df$Region<-factor(df$Region,levels=c("CeCfd","CeCfv","CeCc", "CeLf","CeLc","CeM"))
df$MouseId<-factor(df$MouseId)

#Fit generalized linear model with poisson distribution
model_poisson <- glmmTMB(cFos ~  Treatment*Region + (1|MouseId) + offset(log(Sections)),data=df, family="poisson") 
summary(model_poisson)

## Check assumptions
#Graphs of residuals
windows()
e2 <- resid(model_poisson, type = "pearson")
F2 <- fitted(model_poisson, type ="response")
par(mfrow = c(2,2))
plot(x = F2, 
     y = e2, 
     xlab = "predicted", 
     ylab = "Pearson residuals")
abline(h = 0, lty = 2)

plot(x = df$Treatment:df$Region, y = e2,
     xlab = "Treatment", 
     ylab = "Pearson residuals")
abline(h = 0, lty = 2)

boxplot(e2 ~ MouseId, data = df, 
        xlab = "Mouse", 
        ylab = "Pearson residuals") 
abline(h = 0, lty = 2)

#Validation
#Observed values vs predicted values
plot(x = fitted(model_poisson, type ="response"), 
     y = df$cFos,  
     xlab = "predicted", 
     ylab = "observed")
abline(0,1)
par(mfrow = c(1,1))



disp<-sum(e2^2)/df.residual(model_poisson)

if (disp>1.5){
  print(paste("dispersion parameter = ",as.character( disp),", the model has overdispersion-> Try a negative binomial distribution"))
}else if (disp<0.7){
  print(paste("dispersion parameter = ",as.character( disp ),", the model has subdispersion"))
}else{
  print(paste("dispersion parameter = ",as.character( disp),", the model doesn't have overdispersion nor subdispersion"))
}

#Since model has overdispersion, use a model with negative binomial distribution 

#Fit generalized linear model with negative binomial distribution
model_neg_bin <- glmmTMB(cFos ~  Treatment*Region + (1|MouseId) + offset(log(Sections)),data=df, family="nbinom2") 
summary(model_neg_bin)

## Check assumptions
#Graphs of residuals
windows()
e2 <- resid(model_neg_bin, type = "pearson")
F2 <- fitted(model_neg_bin, type ="response")
par(mfrow = c(2,2))
plot(x = F2, 
     y = e2, 
     xlab = "predicted", 
     ylab = "Pearson residuals")
abline(h = 0, lty = 2)

plot(x = df$Treatment:df$Region, y = e2,
     xlab = "Treatment", 
     ylab = "Pearson residuals")
abline(h = 0, lty = 2)

boxplot(e2 ~ MouseId, data = df, 
        xlab = "Mouse", 
        ylab = "Pearson residuals") 
abline(h = 0, lty = 2)

#Validation
#Observed values vs predicted values
plot(x = fitted(model_neg_bin, type ="response"), 
     y = df$cFos,  
     xlab = "predicted", 
     ylab = "observed")
abline(0,1)
par(mfrow = c(1,1))



disp<-sum(e2^2)/df.residual(model_neg_bin)
if (disp>1.5){
  print(paste("dispersion parameter = ",as.character( disp),", the model has overdispersion"))
}else if (disp<0.7){
  print(paste("dispersion parameter = ",as.character( disp ),", the model has subdispersion"))
}else{
  print(paste("dispersion parameter = ",as.character( disp),", the model doesn't have overdispersion nor subdispersion"))
}

#Since assumptions are achieved, perform likelihood ratio test (LRT)

#Make nested models
null<-glmmTMB(cFos ~  1 + (1|MouseId), data=df, family="nbinom2") #Null model (only random effects variable)
region<-glmmTMB(cFos ~  Region + (1|MouseId), data=df, family="nbinom2") #Model with Region as explanatory variable (EV)
region_treatment<-glmmTMB(cFos ~  Region+Treatment + (1|MouseId), data=df, family="nbinom2") #Model with Region and Treatment as EVs, without interaction

#LRT
lrt=anova(null, region, region_treatment, model_neg_bin, test="Chisq")

#Scheffé post-hoc test 
lsm<-lsmeans(model_neg_bin, ~ Treatment*Region) ## Tukey 
post_hoc=pairs(lsm, by = c("Region" ),adjust="scheffe")
print('Results of statistics for Fig. 4B')
print('Likelihood ratio test:')
print(lrt)
print('Post-hoc multiple comparissons')
print(post_hoc)



##############################################################################3






## Fig. 4D

rm(list=ls())
ls()

#Load libraries
library(glmmTMB)
library(lsmeans)

# Load dataframe
df <- read.csv("https://raw.githubusercontent.com//casey-e/Casey-et-al-2022-b/main/Fig4/Fig4D_Statistics.csv")
attach(df)
View(df)
#Transform to factor
df$Treatment<-factor(df$Treatment,levels=c("Vehicle","SKF","Quinpirole", "Haloperidol"))
df$Region<-factor(df$Region,levels=c("CeCfd","CeCfv","CeCc", "CeLf","CeLc","CeM"))
df$MouseId<-factor(df$MouseId)

#Fit generalized linear model with poisson distribution
model_poisson <- glmmTMB(cFos ~  Treatment*Region + (1|MouseId) + offset(log(Sections)),data=df, family="poisson") 
summary(model_poisson)

## Check assumptions
#Graphs of residuals
windows()
e2 <- resid(model_poisson, type = "pearson")
F2 <- fitted(model_poisson, type ="response")
par(mfrow = c(2,2))
plot(x = F2, 
     y = e2, 
     xlab = "predicted", 
     ylab = "Pearson residuals")
abline(h = 0, lty = 2)

plot(x = df$Treatment:df$Region, y = e2,
     xlab = "Treatment", 
     ylab = "Pearson residuals")
abline(h = 0, lty = 2)

boxplot(e2 ~ MouseId, data = df, 
        xlab = "Mouse", 
        ylab = "Pearson residuals") 
abline(h = 0, lty = 2)

#Validation
#Observed values vs predicted values
plot(x = fitted(model_poisson, type ="response"), 
     y = df$cFos,  
     xlab = "predicted", 
     ylab = "observed")
abline(0,1)
par(mfrow = c(1,1))



disp<-sum(e2^2)/df.residual(model_poisson)

if (disp>1.5){
  print(paste("dispersion parameter = ",as.character( disp),", the model has overdispersion"))
}else if (disp<0.7){
  print(paste("dispersion parameter = ",as.character( disp ),", the model has subdispersion"))
}else{
  print(paste("dispersion parameter = ",as.character( disp),", the model doesn't have overdispersion nor subdispersion"))
}


#Since assumptions are achieved, perform likelihood ratio test (LRT)

#Make nested models
null<-glmmTMB(cFos ~  1 + (1|MouseId), data=df, family="poisson") #Null model (only random effects variable)
region<-glmmTMB(cFos ~  Region + (1|MouseId), data=df, family="poisson") #Model with Region as explanatory variable (EV)
region_treatment<-glmmTMB(cFos ~  Region+Treatment + (1|MouseId), data=df, family="poisson") #Model with Region and Treatment as EVs, without interaction

#LRT
lrt=anova(null, region, region_treatment, model_poisson, test="Chisq")

#Scheffé post-hoc test 
lsm<-lsmeans(model_poisson, ~ Treatment*Region) ## Tukey 
post_hoc=pairs(lsm, by = c("Region" ),adjust="scheffe")
print('Results of statistics for Fig. 4D')
print('Likelihood ratio test:')
print(lrt)
print('Post-hoc multiple comparissons')
print(post_hoc)

