#Classifier Lib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression    
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


Datapath='/content/AllData'

####Load target
target=[]

for img_filename in os.listdir(Datapath):
    imgtype=img_filename.split("_",1)[0]
    if (imgtype=='glioma'):
        target.append('0')
    if (imgtype=='meningioma'):
        target.append('1')
    if (imgtype=='pituitary'):
        target.append('2')
    if (imgtype=='no'):
        target.append('3')


#print(target)        
####Load Data        

arrayofdata_=[]
arrayofdata=[]

for filename in glob.glob('/content/AllData/*.jpg'):
    img=cv2.imread(filename)
    img=cv2.resize(img, (80, 80), interpolation=cv2.INTER_LINEAR) 
    img=np.array(img)    
    inputdata = np.reshape(img, (img.shape[0],img.shape[1],img.shape[2]))
    inputdata=np.array(inputdata)
    arrayofdata.append(inputdata.tolist())
    arrayofdata_=arrayofdata

arrayofdata_ = np.array(arrayofdata_)
one_hot_labels = to_categorical(target, num_classes=4)

#print(one_hot_labels)
    
x_train, x_test, y_train, y_test = train_test_split(arrayofdata_,
                                                          one_hot_labels,
                                                          test_size=0.1,shuffle=True,
                                                          random_state=42)

print(x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],x_train.shape[2],3))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],x_test.shape[2],3))



#Reshape Data for Confusion_matrix
x_train1 = np.reshape(x_train, (x_train.shape[0],x_train.shape[1]*x_train.shape[2]*3))
x_test1 = np.reshape(x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]*3))
y_train1 = np.argmax(y_train, axis=1)
y_test1 = np.argmax(y_test, axis=1)

#################################################Machine Learning#########################################
print('Machine Learning')
#HC:Healthy Control Adn PD:Parkinson Disease
my_tags=["Glioma","Meningioma","Pituitary","Healthy"]

#SVM Classifier

logreg = Pipeline([('clf', SVC(kernel = 'rbf', random_state = 42)),])               
logreg.fit(x_train1, y_train1)    
y_pred_svm = logreg.predict(x_test1)
print('SVM accuracy is %s' % accuracy_score(y_pred_svm, y_test1))
print(classification_report(y_test1, y_pred_svm,target_names=my_tags))


#SGD Classifier

from sklearn.linear_model import SGDClassifier
sgd = Pipeline([
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(x_train1, y_train1)

y_pred = sgd.predict(x_test1)
print('SGD accuracy is %s' % accuracy_score(y_pred, y_test1))
print(classification_report(y_test1, y_pred,target_names=my_tags))


#LR Classifier
from sklearn.linear_model import LogisticRegression
logreg = Pipeline([('clf', LogisticRegression(n_jobs=1, C=1e5)), ])              
logreg.fit(x_train1, y_train1)
y_pred_lr = logreg.predict(x_test1)
print('LR accuracy is %s' % accuracy_score(y_pred_lr, y_test1))
print(classification_report(y_test1, y_pred_lr,target_names=my_tags))


#K-Nearest Neighbours Classifier    
 
    

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train1, y_train1)
y_pred_Knn = clf.predict(x_test1)
cm = confusion_matrix(y_test1, y_pred_Knn) 
print('KNN accuracy is %s' %accuracy_score(y_pred_Knn,y_test1 ))
print(classification_report(y_test1, y_pred_Knn,target_names=my_tags))

#RandomForest Classifier

RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_train1, y_train1)  
y_pred_rf=RF.predict(x_test1)  
print('RF accuracy is %s' % accuracy_score(y_pred_rf, y_test1))
print(classification_report(y_test1, y_pred_rf,target_names=my_tags))


my_tags=["Glioma","Meningioma","Pituitary","Healthy"]
#MLP Classifier
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(x_train1, y_train1)  
y_pred_mlp=NN.predict(x_test1)  
print('MLP accuracy is %s' % accuracy_score(y_pred_mlp, y_test1))
print(classification_report(y_test1, y_pred_mlp,target_names=my_tags))
