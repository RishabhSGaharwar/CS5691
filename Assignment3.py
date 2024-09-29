from datasets import load_dataset
import nltk
import sklearn
from sklearn import svm
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
#################################

# Use the Enron_Spam dataset which has 31000 training mails and 33000 testing mails
dataset = load_dataset("SetFit/enron_spam")
train_dataset = dataset["train"]
test_dataset = dataset["test"]


 
## DATASET DESCRIPTION
# Text has both - subject as well as message
# Label - 0 ==> HAM
# Label - 1 ==> SPAM


 
stopword_set = set(stopwords.words('english'))
tokenizer = nltk.tokenize.TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()


 
withLemmatize = True
withStopWordRemoval = True

def PreProcess_Mails(mail, processed_mails, bag_of_words, with_stopword_removal=True, with_lemmatize=True):
    mail_dict = {}
    mail_dict["id"] = mail["message_id"]
    tokens = tokenizer.tokenize(mail["text"])
    words = []
    for token in tokens:
        if token == '$':
            token = 'dollar'
        if token.isalpha() and len(token) > 1:
            word = None
            if with_stopword_removal:
                if with_lemmatize and token not in stopword_set:
                    word = lemmatizer.lemmatize(token.lower())
                elif token not in stopword_set:
                    word = token.lower()
            else:
                if with_lemmatize:
                    word = lemmatizer.lemmatize(token.lower())
                else:
                    word = token.lower()
            if word is not None:
                words.append(word)
                if word not in bag_of_words:
                    bag_of_words[word] = 1
                else:
                    bag_of_words[word] += 1
    mail_dict["processed_text"] = words
    mail_dict["text"] = mail["text"]
    mail_dict["label"] = mail["label"]
    processed_mails.append(mail_dict)

processed_mails = []
bag_of_words = {}

for i in range(len(train_dataset)):
    PreProcess_Mails(train_dataset[i], processed_mails, bag_of_words)


 
threshold = 5000
sorted_items = sorted(bag_of_words.items(), key = lambda x : x[1], reverse=True)
bagOfWords_Frequent= sorted_items[:threshold]


 
word_index_map = {}
for i in range(len(bagOfWords_Frequent)):
    word_index_map[bagOfWords_Frequent[i][0]] = i


 
def VectorizeMails(mail, word_index_map):
    mail_vector_freq = np.zeros((len(word_index_map), 1), dtype='int32')
    for word in mail["processed_text"]:
        if(word in word_index_map):
            mail_vector_freq[word_index_map[word]] += 1
    mail_vector_01 = np.zeros((len(word_index_map)), dtype='int32')
    for i in range(len(word_index_map)):
        if(mail_vector_freq[i]>0):
            mail_vector_01[i] = 1
    mail["frequency_vector"] = mail_vector_freq
    mail["presence_vector"] = mail_vector_01


 
for i in range(len(processed_mails)):
    VectorizeMails(processed_mails[i], word_index_map)
    # print(f"Mail {i} vectorized")

# def read_test_emails():
#     test_folder = "test"
#     test_emails = []
    
#     if not os.path.exists(test_folder):
#         print("Test folder not found.")
#         return test_emails
    
#     for filename in os.listdir(test_folder):
#         if filename.startswith("email") and filename.endswith(".txt"):
#             with open(os.path.join(test_folder, filename), "r") as file:
#                 email_content = file.read()
#                 test_emails.append(email_content)
    
#     return test_emails

# test_dataset = read_test_emails()
# # print(test_emails)
 
processed_mails_test = []
bag_of_words_test = {}
for i in range(len(test_dataset)):
    PreProcess_Mails(test_dataset[i], processed_mails_test, bag_of_words_test)
for i in range(len(test_dataset)):
    VectorizeMails(processed_mails_test[i], word_index_map)


 
def getConfusionMatrix(y_predict, processed_mails_test, description):
    confusion_matrix = np.zeros((2, 2), dtype='int32')
    for i in range(len(processed_mails_test)):
        if processed_mails_test[i]["label"]==1:
            if y_predict[i]==1:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][0] += 1
        else:
            if y_predict[i]==1:
                confusion_matrix[0][1] += 1
            else:
                confusion_matrix[1][1] += 1
    print(f"{description}")
    print(confusion_matrix)
    print(f"Accuracy is {(100.0*(confusion_matrix[0][0]+confusion_matrix[1][1]))/(1.0*(confusion_matrix[0][0]+confusion_matrix[1][1]+confusion_matrix[0][1]+confusion_matrix[1][0]))}%")
    print(f"Precision is {(100.0*(confusion_matrix[0][0]))/(1.0*(confusion_matrix[0][0]+confusion_matrix[0][1]))}%")
    print(f"Recall is {(100.0*(confusion_matrix[0][0]))/(1.0*(confusion_matrix[0][0]+confusion_matrix[1][0]))}%")
    print(f"F1-Score is {(100.0*(2*confusion_matrix[0][0]))/(1.0*(2*confusion_matrix[0][0]+confusion_matrix[1][0]+confusion_matrix[0][1]))}%")
    return confusion_matrix


 
svm_classifier = svm.LinearSVC(loss='hinge', dual='auto', fit_intercept=True)
svm_classifier.fit(([mail["frequency_vector"].reshape(5000,) for mail in processed_mails]), ([mail["label"] for mail in processed_mails]))


 
y_predict_test = svm_classifier.predict([mail["frequency_vector"].reshape(5000) for mail in processed_mails_test])
confusion_matrix_SVM_hinge = getConfusionMatrix(y_predict_test, processed_mails_test, "For SVM with Hinge Loss and Frequency vector as Inputs")
y_predict_train = svm_classifier.predict([mail["frequency_vector"].reshape(5000) for mail in processed_mails])
confusion_matrix_SVM_hinge = getConfusionMatrix(y_predict_train, processed_mails, "For SVM with Hinge Loss and Frequency vector as Inputs")


 
svm_classifier_2 = svm.LinearSVC(loss='hinge', dual='auto', fit_intercept=True)
svm_classifier_2.fit(([mail["presence_vector"].reshape(5000,) for mail in processed_mails]), ([mail["label"] for mail in processed_mails]))


 
y_predict_test_presence = svm_classifier_2.predict([mail["presence_vector"].reshape(5000) for mail in processed_mails_test])
confusion_matrix_SVM_presence = getConfusionMatrix(y_predict_test_presence, processed_mails_test, "For SVM with Hinge Loss and Presence vector as Inputs")
y_predict_train_presence = svm_classifier_2.predict([mail["presence_vector"].reshape(5000) for mail in processed_mails])
confusion_matrix_SVM_hinge = getConfusionMatrix(y_predict_train_presence, processed_mails, "For SVM with Hinge Loss and Presence vector as Inputs")


 
svm_classifier = svm.LinearSVC(loss='squared_hinge', dual='auto', fit_intercept=True, max_iter = 10000)
svm_classifier.fit(([mail["frequency_vector"].reshape(5000,) for mail in processed_mails]), ([mail["label"] for mail in processed_mails]))


 
y_predict_test = svm_classifier.predict([mail["frequency_vector"].reshape(5000) for mail in processed_mails_test])
confusion_matrix_SVM_hinge = getConfusionMatrix(y_predict_test, processed_mails_test, "For SVM with Hinge Loss and Frequency vector as Inputs")
y_predict_train = svm_classifier.predict([mail["frequency_vector"].reshape(5000) for mail in processed_mails])
confusion_matrix_SVM_hinge = getConfusionMatrix(y_predict_train, processed_mails, "For SVM with Hinge Loss and Frequency vector as Inputs")


 
class NaiveBayes:
    def __init__(self, processed_mails_training):
        self.data = processed_mails_training
        self.X = np.array([mail["presence_vector"].flatten() for mail in processed_mails_training], dtype='int32')
        self.Y = np.array([mail["label"] for mail in processed_mails_training], dtype='int32')
        self.ones = np.array([mail["presence_vector"].flatten() for mail in processed_mails_training if mail["label"]==1], dtype='int32')
        self.zeros = np.array([mail["presence_vector"].flatten() for mail in processed_mails_training if mail["label"]==0], dtype='int32')
    
    def parameterTuning(self):
        num_ones = np.sum(self.Y)
        num_zeros = self.Y.shape[0]-num_ones
        self.probability1 = 1.0*(np.sum(self.ones, axis=0)+1)/(num_ones+2)
        self.probability0 = 1.0*(np.sum(self.zeros, axis=0)+1)/(num_zeros+2)
        self.p = (1.0*np.sum(self.Y)+1)/(1.0*self.Y.shape[0]+2)
    
    def predict(self, mail):
        log_probab_1 = np.log(self.p)
        log_probab_0 = np.log(1-self.p)
        for i in range(mail["presence_vector"].shape[0]):
            log_probab_1 += mail["presence_vector"][i]*(np.log(self.probability1[i]))
            log_probab_1 += (1- mail["presence_vector"][i])*(np.log(1-self.probability1[i]))
            log_probab_0 += mail["presence_vector"][i]*(np.log(self.probability0[i]))
            log_probab_0 += (1- mail["presence_vector"][i])*(np.log(1-self.probability0[i]))
        if log_probab_0 > log_probab_1:
            return 0
        return 1


 
def runNaiveBayes(processed_mails):
    nb = NaiveBayes(processed_mails)
    nb.parameterTuning()
    return nb

NaiveBayesModel = runNaiveBayes(processed_mails)


 
y_predict_nb_test = [NaiveBayesModel.predict(mail) for mail in processed_mails_test]
confusion_matrix_nb_test = getConfusionMatrix(y_predict_nb_test, processed_mails_test, "Confusion Matrix for Naive Bayes")


 
y_predict_nb_train = [NaiveBayesModel.predict(mail) for mail in processed_mails]
confusion_matrix_nb_test = getConfusionMatrix(y_predict_nb_train, processed_mails, "Confusion Matrix for Naive Bayes")


 
class LogisticRegression:
    def __init__(self, processed_mails_training):
        self.data = processed_mails_training
        self.X = np.array([mail["frequency_vector"].flatten() for mail in processed_mails_training], dtype='int32')
        self.Y = np.array([mail["label"] for mail in processed_mails_training], dtype='int32')
        self.w = np.zeros_like(self.X[0].flatten())
    
    def linkFunction(self, x):
        dot_product = np.dot(x, self.w)
        return (1/(1+np.exp(-dot_product)))

    def gradient(self):
        predictions = self.linkFunction(self.X)
        residual = self.Y - predictions
        gradient = np.dot(residual, self.X)
        return gradient
    
    def gradientAscent(self, learning_rate, epochs):
        w_next = np.zeros_like(self.w)
        w_prev = self.w
        difference_array = []
        for i in range(int(epochs)):
            w_next = w_prev + (learning_rate*(self.gradient())/(i+1))
            difference_array.append(np.linalg.norm(w_next - w_prev))   
            self.w = w_next
            w_prev = self.w   
        return difference_array, self.w
    
    def predict(self, x_test):
        prediction_probability = self.linkFunction(x_test["frequency_vector"].flatten())
        if prediction_probability >= 0.5:
            return 1
        return 0


 
def runLogisticRegression(processed_mails, processed_mails_test):
    lr = LogisticRegression(processed_mails)
    diff_array, w_lr = lr.gradientAscent(1e-5, 2e2)
    return lr

LogisticRegressionCM = runLogisticRegression(processed_mails, processed_mails_test)


 
y_predict_lr_test = [LogisticRegressionCM.predict(mail) for mail in processed_mails_test]
confusion_matrix = getConfusionMatrix(y_predict_lr_test, processed_mails_test, "Confusion Matrix for Logistic Regression - Testing Data")


 
y_predict_lr_train = [LogisticRegressionCM.predict(mail) for mail in processed_mails]
confusion_matrix_lr_train = getConfusionMatrix(y_predict_lr_train, processed_mails, "Confusion Matrix for Logistic Regression - Training Data")


 
class Node:
    def __init__(self, LeftNodeLink, RightNodeLink, Value, feature, Depth):
        self.Left = LeftNodeLink
        self.Right = RightNodeLink
        self.Value = Value
        self.Depth = Depth
        self.Feature = feature
    
    def __str__(self):
        value = (f"Left child is {'None' if self.Left is None else 'not None'}") + '\n' + (f"Right child is {'None' if self.Right is None else 'not None'}") + '\n' + (f"Feature is {self.Feature}") + '\n' + (f"Value is {self.Value}") + '\n' + (f"Depth is {self.Depth}") + '\n' + ("=======================================")
        return value


class DecisionTree:
    def __init__(self, maxDepth, X_train, Y_train):
        self.X = X_train
        self.Y = Y_train
        self.maxDepth = maxDepth
        self.features = (X_train[0].shape)[0]  

    def entropy(self, num_ones, num_zeros):
        if num_ones==0 or num_zeros==0:
            return 0
        fraction1 = num_ones/(num_ones+num_zeros)
        fraction0 = num_zeros/(num_zeros+num_ones)
        entropy = -fraction0*(np.log2(fraction0))
        entropy += -fraction1*(np.log2(fraction1))
        return entropy

    def constructTree(self, features, indices, depth):
        num_ones = np.sum(self.Y[indices])
        num_zeros = self.Y.shape[0] - num_ones
        majority = 1 if num_ones >= num_zeros else 0
        if depth == self.maxDepth:
            return Node(None, None, majority, None, depth)
        elif majority == self.Y.shape[0]:
            return Node(None, None, majority, None, depth)
        elif len(features)==0:
            return Node(None, None, majority, None, depth)
        min_entropy = 10
        min_entropy_feature = -1
        index = -1
        current_X = self.X[indices]
        yes_indices = None
        no_indices = None
        for i,feature in enumerate(features):
            yes_indices = np.argwhere(current_X[:, feature]==1) [:, 0]
            no_indices = np.argwhere(current_X[:, feature]==0)[:, 0]     
            yes_indices_labels = self.Y[yes_indices]
            no_indices_labels = self.Y[no_indices]
            yes_indices_1 = np.sum(yes_indices_labels)
            yes_indices_0 = yes_indices_labels.shape[0] - yes_indices_1
            no_indices_1 = np.sum(no_indices_labels)
            no_indices_0 = no_indices_labels.shape[0] - no_indices_1
            loss = (yes_indices.shape[0]/(num_ones+num_zeros))*self.entropy(yes_indices_1, yes_indices_0) + (no_indices.shape[0]/(num_ones+num_zeros))*self.entropy(no_indices_1, no_indices_0)
            if loss < min_entropy:
                min_entropy = loss
                min_entropy_feature = feature
                index = i
        new_features = np.delete(features, index)
        leftNode = self.constructTree(new_features, yes_indices, depth+1)
        rightNode = self.constructTree(new_features, no_indices, depth+1) 
        return Node(leftNode, rightNode, 1, min_entropy_feature, depth)   
    
    def rootConstructor(self):
        self.root = self.constructTree(np.arange(self.X[0].shape[0]), np.arange(self.X.shape[0]), 0)
        return self.root

def DecisionTreepredict(root, x_test):
    y_test = -1
    while(root.Left is not None and root.Right is not None):
        if(x_test[root.Feature]==root.Value):
            root = root.Left
        else:
            root = root.Right
    y_test = root.Value
    return y_test


 
X_train_tree = np.array([mail["presence_vector"].flatten() for mail in processed_mails])
Y_train_tree = np.array([mail["label"] for mail in processed_mails])


 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth = 8)
clf.fit(X_train_tree, Y_train_tree)
ysklearn = clf.predict(np.array([mail["presence_vector"] for mail in processed_mails_test]))
cfm = getConfusionMatrix(ysklearn, processed_mails_test, "Confusion Matrix for sklearn decision tree")


 
dt = DecisionTree(12, X_train_tree, Y_train_tree)
root = dt.rootConstructor()


 
y_predict_dt_test = [DecisionTreepredict(root, mail["presence_vector"]) for mail in processed_mails_test]
cfmtx = getConfusionMatrix(y_predict_dt_test, processed_mails_test, "Confusion Matrix for Decision Stump - Testing Data")


 
y_predict_dt_train = [DecisionTreepredict(root, mail["presence_vector"]) for mail in processed_mails]
cfmtx = getConfusionMatrix(y_predict_dt_train, processed_mails, "Confusion Matrix for Decision Stump - Training Data")


 
class AdaBoost:
    def __init__(self, X_train, Y_train, gamma=0.01):
        self.X = X_train
        self.Y = Y_train
        self.D = np.ones_like(self.Y, dtype='float64')
        self.D /= np.sum(self.D)
        self.learners_ = []
        self.weights_ = []
        self.alphas_ = []
        self.T = np.log(2*(X_train.shape[0]))/(2*(gamma**2))
        print(f"Number of iterations is {self.T}")
        print(self.D)
        self.features = np.arange(self.X[0].shape[0])

    def weak_learner(self):
        np.random.shuffle(self.features)
        for feature in self.features:
            yes_indices = np.argwhere(self.X[:, feature]==1) [:, 0]
            no_indices = np.argwhere(self.X[:, feature]==0)[:, 0]    
            yes_indices_labels = self.Y[yes_indices]
            no_indices_labels = self.Y[no_indices]
            error1 = np.sum((yes_indices_labels==0).astype('float')*self.D[yes_indices]) + np.sum((no_indices_labels==1).astype('float')*self.D[no_indices])
            error2 = np.sum((yes_indices_labels==1).astype('float')*self.D[yes_indices]) + np.sum((no_indices_labels==0).astype('float')*self.D[no_indices])
            # print("ERROR", error1, error2)
            if error1<error2:
                root = Node(Node(None, None, 1, None, 0), Node(None, None, 0, None, 0), 1, feature, 1)
                return root, error1
            else:
                root = Node(Node(None, None, 1, None, 0), Node(None, None, 0, None, 0), 0, feature, 1)
                return root, error2
    
    def boosting(self):
        for i in range(int(self.T)):
            # print("=====================================================")
            # print(f"Iteration {i} started")
            stump_root_, error_ = self.weak_learner()
            self.learners_.append(stump_root_)
            prediction = np.array([DecisionTreepredict(stump_root_, self.X[i]) for i in range(self.X.shape[0])])
            predict_result = prediction==self.Y
            # print(predict_result)
            correct_label = predict_result.astype('int')
            incorrect_label = 1-correct_label
            alpha = 0.5*np.log((1-error_)/error_)
            # print(f"Alpha is {alpha}")
            correct_label = np.exp(-alpha*correct_label)
            incorrect_label = np.exp(alpha*incorrect_label)
            # print(correct_label)
            # print(incorrect_label)
            # print(correct_label*incorrect_label)
            self.D = self.D*correct_label*incorrect_label
            self.D /= np.sum(self.D)
            self.alphas_.append(alpha)
            # print(self.D)
            # print(f"Iteration {i} completed")
            # print("=====================================================")

    def predict(self, mail):
        H = 0.0
        for i in range(int(self.T)):
            H += self.alphas_[i]*(1 if DecisionTreepredict(self.learners_[i], mail)==1 else -1)
        H = H>=0
        H = int(H)
        return H


 
def runAdaBoost(X_train_tree, Y_train_tree):
    adaboost = AdaBoost(X_train_tree, Y_train_tree)
    adaboost.boosting()
    return adaboost


 
ADABOOST_MODEL = runAdaBoost(X_train_tree, Y_train_tree)


 
def testAdaBoost(adaboost_model, processed_mails_test, label):
    y_predict_boosting = [adaboost_model.predict(mail["presence_vector"]) for mail in processed_mails_test]
    cnfmtx_boost = getConfusionMatrix(y_predict_boosting, processed_mails_test, label)
    return cnfmtx_boost


 
confusion_matrix_AdaBoost_testing = testAdaBoost(ADABOOST_MODEL, processed_mails_test, "Confusion Matrix for Boosting (Testing Data)")


 
confusion_matrix_AdaBoost_testing = testAdaBoost(ADABOOST_MODEL, processed_mails, "Confusion Matrix for Boosting (Training Data)")


