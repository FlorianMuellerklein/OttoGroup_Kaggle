require(xgboost)
require(methods)

train = read.csv('data/train_tfidf.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('data/test_tfidf.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
#param <- list("objective" = "multi:softprob",
#              "eval_metric" = "mlogloss",
#			  "eta" = 0.05,
#			  "min_child_weight" = 6,
#			  "max_depth" = 18,
#              "num_class" = 9,
#              "nthread" = 8)

# graphlab params from kaggle forums  
param <- list("objective" = "multi:softprob",
		"eval_metric" = "mlogloss",
		"eta" = 0.05,
		"max_depth" = 10,
		"min_child_weight" = 4,
		"subsample" = .8,
		"min_loss_reduction" = 1,
		"colsample_bytree" = .7,
		"num_class" = 9)
		#"booster" = 'gblinear')


# Run Cross Valication
cv.nround = 700
xgb.cv(param = param, data = x[trind,], label = y, 
                nfold = 3, nrounds = cv.nround)

# Train the model
nround = 500
bst = xgboost(param = param, data = x[trind,], label = y, nrounds = nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='Preds/xgboost_tuned_softprob_tfidf_datoparams.csv', quote=FALSE,row.names=FALSE)
