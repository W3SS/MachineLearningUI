print 'from os import system'
from os import system
system('cls')
print "-----IMPORTING LIBS-----"
print "import matplotlib.pyplot as plt"
import matplotlib.pyplot as plt
print "from sklearn.datasets import load_digits"
from sklearn.datasets import load_digits
print "from sklearn.metrics import accuracy_score"
from sklearn.metrics import accuracy_score
print "from sklearn import svm"
from sklearn import svm
print "from sklearn import ensemble"
from sklearn import ensemble
print "from sklearn import naive_bayes"
from sklearn import naive_bayes
print "import numpy as np"
import numpy as np
print "import matplotlib.mlab as mlab"
import matplotlib.mlab as mlab
print "from scipy.stats import norm"
from scipy.stats import norm
print "import msvcrt as m"
import msvcrt as m
print "import numbers"
import numbers
print 'from pprint import pprint'
from pprint import pprint
system('cls')

digits = load_digits()

colorss = ['#000000', '#FF0000', '#FFFF00', '#00FF00', '#00FFFF', '#000080', '#FF00FF', '#800080', '#FF69B4', '#BC8F8F']
globVars = {'datasets': {'digits': digits}, 'digits1DPic': digits['data'], 'target': digits['target']}

def plotdata(f1=[], f2=[], mode='dot'):
    if(mode == 'dot'):
        print "-----DOT MODE-----"
        for i in range(len(f1)):
                label = digits.target[i]
                x = f1[i]
                y = f2[i]
                plt.plot(x, y, color=colorss[label], alpha=0.5, label=str(label), marker='x')
    elif(mode == 'hist'):
        print "-----HISTOGRAM MODE-----"
        hist = [[] for y in range(max(digits.target)+1)]
        for i, a in enumerate(f1):
            hist[digits.target[i]].append(a)
        for i, dataclass in enumerate(hist):
                plt.hist(dataclass, histtype='step', color=colorss[i], alpha=0.5, label=str(i))
    elif(mode == 'histlin'):#BUGGY
        print "-----HISTOGRAM MODE-----"
        hist = [[] for y in range(max(digits.target)+1)]
        for i, a in enumerate(f1):
            hist[digits.target[i]].append(a)
        for i, dataclass in enumerate(hist):
            (mu, sigma) = norm.fit(i)
            n, bins, patches = plt.hist(i, 60, normed=1, facecolor='green', alpha=0.75)
            y = mlab.normpdf( bins, mu, sigma)
            plt.plot(bins, y, color=colorss[i], alpha=0.5, label=str(i))
    plt.legend(loc='upper right')
    plt.show()

def pltUsFr():
    # TODO: make it use the function below too. press to go to next frame etc.
    print """<<-----PLOT-----
-1) BACK
1) 2D map

    """
    choice = input('Chouce? ')
    if choice == -1:
        return -1
    elif choice == 1:
        plotDat = printGlobVars('chose 2D arr to plot: ', ret=True)
        plotDig(plotDat)
    else:
        return -1
    print 'press any key to contine'
    m.getch()
    system('cls')
    home()
#     print "-----Plot-----"
#     print """
# # 1) Set X
# # 2) Set Y
# # 3) Plot
# """
#     x = []
#     y = []
#     if choice == 1:
#         system('cls')
#         print 

def plotDig(dig):
    #dig is something thst looks like digits.image[x]
    plt.figure(1, figsize=(3, 3))
    plt.imshow(dig, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

def diffmirror(threeDarr):
    # TODO: check this over. might not actually mirrir
    # consider matrices
    rev = list(reversed(threeDarr))
    ret = []
    for i, d1 in enumerate(threeDarr):
        temp2d = []
        for j, d2 in enumerate(d1):
            rev = list(reversed(d2))
            temp = []
            for k, d3 in enumerate(d2):
                temp.append(d3-rev[k])
            temp2d.append(temp)
        ret.append(temp2d)
    return ret

def readto3d(twoDarr, stride=8):
    ret = []
    for i in range(len(twoDarr)):
        temp2d = []
        while(len(temp2d) < len(twoDarr[i])/stride):
            temp = []
            globpos = 0
            for j in range(stride):
                temp.append(twoDarr[i][j+globpos])
            temp2d.append(temp)
        ret.append(temp2d)
    return ret

def readto2d(threeDarr):
    ret = []
    for i in threeDarr:
        temp = []
        for j in i:
            for k in j:
                temp.append(k)
        ret.append(temp)
    return ret

# for use in subsequent function
def expand(editDat, message):
    if isinstance(editDat, (list, np.ndarray)):
        print 'Indicies are: ' + str(np.shape(editDat))
        index = input('index? ')
        ret = editDat[index]
    elif isinstance(editDat, dict):
        for key in editDat.keys():
            print '\t|---' + str(key)
        key = input('key? ')
        ret = editDat[key]
    else:
        ret = editDat
        # can't expand any more
    print '-----'
    pprint(ret)
    print '-----'
    return ret

def printGlobVars(message, ret=False):
    system('cls')
    print '-2) Expand Var'
    print '-1) HOME'
    for key in globVars:
        print str(globVars.keys().index(key)) + ') ' + str(key) + '   --- Dimmensions: ' + str(np.shape(globVars[key])) + '---' + str(len(np.shape(globVars[key])))
    if ret:
        choice = input('----------\n'+str(message))
        if choice == -1:
            return -1
            # go home, break functionns with -1
        elif choice == -2:
            choice = input('chosea dataset to expand: ')
            editDat = globVars[globVars.keys()[choice]]
            cont = True
            while cont:
                editDat = expand(editDat, message)
                a = 'a'
                while a not in 'yYnN':
                     print 'Continue to expand? (y/n)'
                     a = m.getch()
                if a in 'yY':
                    cont = True
                elif a in 'nN':
                    cont = False
                    break
            system('cls')
            return editDat
        else:
            editDat = globVars[globVars.keys()[choice]]
            system('cls')
            return editDat
    else:
        choice = input('Choice: ')
        if choice == -2:
            choice = input('chosea dataset to expand: ')
            editDat = globVars[globVars.keys()[choice]]
            cont = True
            while cont:
                editDat = expand(editDat, message)
                a = 'a'
                while a not in 'yYnN':
                    print 'Continue to expand? (y/n)'
                    a = m.getch()
                    if a in 'yY':
                        cont = True
                    elif a in 'nN':
                        cont = False
                        break
    print 'Press any key to continue'
    m.getch()
    system('cls')

def chop():
    system('cls')
    print "<<-----CHOP-----"
    editDat = printGlobVars('What do you want to CHOP? ', ret=True)
    if isinstance(editDat, int):
        if editDat == -1:
            return -1
    dim = len(np.shape(editDat))
    system('cls')

    chopRet = []
    if dim == 0:
        print "You chose a single number"
    elif dim == 1:
        print "Enter an expression for which integer terms at x will be included in the return(Ex: 2*x + 1 will include every other term starting at 1)"
# changed f(x) to string with eval statement - Think is dons
# changed range to input tuple and pass each as catsted int to range - think is done
# changed domain.append(fx) - think is done
        f = input("***surround with quotes***\nf(x) = ")
        domain = []
        print "Domains can be overshot"
        print "\tshape: ", np.shape(editDat)
        rang = input("Domain(lower, upper, incriment): ")
        for i, x in enumerate(range(int(rang[0]), int(rang[1]), int(rang[2]))):
            domain.append(eval(f))
            #populate domain of f(x)
        for i, x in enumerate(domain):
            if x > len(editDat)-1:
                # gotten through all of the dataset
                break
            elif domain[i]%1 == 0:
                chopRet.append( editDat[int(x)] )
                #gets values for average from domain values
    elif dim == 2:
#TODO: make right
# [[None] * int(np.shape(digits.data)[1]) for x in xrange(int(np.shape(digits.data)[0]))]
# changed append to avarr[iy]
# check everything many more times and add more error catching
        print "Enter expressions for which integer terms at x, y will be included in the return(Ex: f(x) = 2*x + 1, f(y) = 3*y will include every other term starting at 1 in every third column for every third row)"
        fx = input("***surround with quotes***\nf(x) = ")
        fy = input("f(y) = ")
        domainx = [] # conatnis values at which array of data will be accessed
        domainy = []
        print "\tshape: ", np.shape(editDat)
        rang = input("Domain x(lower, upper, incriment): ")
        for i, x in enumerate(range(int(rang[0]), int(rang[1]), int(rang[2]))):
            domainx.append(eval(fx))
            #populate domain of f(x)
        rang = input("Domain y(lower, upper, incriment): ")
        for i, y in enumerate(range(int(rang[0]), int(rang[1]), int(rang[2]))):
            domainy.append(eval(fy))
            #populate domain of f(y)
        domainx = list(set(domainx))
        domainx.sort()
        domainy = list(set(domainy))
        domainy.sort()

        for iy, y in enumerate(domainy):
            if y > len(editDat)-1:
                break
                # no dataponit
            elif y%1 == 0: #its a whole number
                for ix, x in enumerate(domainx):
                    if x > len(editDat[y])-1:
                        # gotten through all of the dataset / value to access array is out of range
                        break
                    elif y%1 == 0: #its a whole number
                        try:
                            chopRet[iy].insert(ix, editDat[int(y)][int(x)] )
                        except IndexError:
                            chopRet.insert(iy, [])
                            chopRet[iy].insert(ix, editDat[int(y)][int(x)] )
    elif dim == 3:
#TODO: make 3d
# UNTESTED
        print "Enter expressions for which integer terms at x, y will be included in the return(Ex: f(x) = 2*x + 1, f(y) = 3*y will include every other term starting at 1 in every third column for every third row)"
        fx = input("***surround with quotes***\nf(x) = ")
        fy = input("f(y) = ")
        fz = input("f(z) = ")
        domainx = [] # conatnis values at which array of data will be accessed
        domainy = []
        domainz = []
        print "\tshape: ", np.shape(editDat)
        rang = input("Domain x(lower, upper, incriment): ")
        for i, x in enumerate(range(int(rang[0]), int(rang[1]), int(rang[2]))):
            domainx.append(eval(fx))
            #populate domain of f(x)
        rang = input("Domain y(lower, upper, incriment): ")
        for i, y in enumerate(range(int(rang[0]), int(rang[1]), int(rang[2]))):
            domainy.append(eval(fy))
            #populate domain of f(y)
        rang = input("Domain z(lower, upper, incriment): ")
        for i, z in enumerate(range(int(rang[0]), int(rang[1]), int(rang[2]))):
            domainy.append(eval(fz))
            #populate domain of f(z)
        domainx = list(set(domainx))
        domainx.sort()
        domainy = list(set(domainy))
        domainy.sort()
        domainz = list(set(domainz))
        domainz.sort()

        for iz, z in enumerate(domainz):
            if z > len(editDat):
                break
                # no more data
            elif z%1 == 0:
                for iy, y in enumerate(domainy):
                    if y > len(editDat[z])-1:
                        break
                        # no dataponit
                    elif y%1 == 0: #its a whole number
                        for ix, x in enumerate(domainx):
                            if x > len(editDat[z][y])-1:
                                # gotten through all of the dataset / value to access array is out of range
                                break
                            elif y%1 == 0: #its a whole number
                                try:
                                    chopRet[iz][iy].insert(ix, editDat[int(z)][int(y)][int(x)] )
                                except IndexError:
                                    chopRet[ix].insert(iy, [])
                                    try:
                                        chopRet.insert(iz, [])   
                                        chopRet[iz][iy].insert(ix, editDat[int(z)][int(y)][int(x)] )
                                    except IndexError:
                                        chopRet[iz][iy].insert(ix, editDat[int(z)][int(y)][int(x)] )
    else:
        return -1
    # DONE: average chopRet

    return chopRet

def summ():
    system('cls')
    print "<<-----SUM-----"
    sumArr = chop()
    if isinstance(sumArr, int):
        if sumArr == -1:
            return -1
    a = 'a'
    kepDim = False
    while a not in 'yYnN':
         print 'Keep origional dimmensions? (y/n)'
         a = m.getch()
    if a in 'yY':
        kepDim = True
    SUM = np.sum(sumArr, axis=1, keepdims=kepDim)
    print "Shape Sum: ", np.shape(SUM)
    print "Sum: ", SUM
    return SUM

def divide():
    system('cls')
    print "<<-----DIVIDE-----"
    dividend = chop()
    divisor = chop()
    if isinstance(dividend, int):
        if dividend == -1:
            return -1
    if isinstance(divisor, int):
        if divisor == -1:
            return -1
    a = 'a'
    kepDim = False
    while a not in 'yYnN':
         print 'Keep origional dimmensions? (y/n)'
         a = m.getch()
    if a in 'yY':
        kepDim = True
    try:
        QUOTIENT = np.divide(dividend, divisor)
        print "Shape Quotient: ", np.shape(QUOTIENT)
        print "Quotient: ", QUOTIENT
        if kepDim == False:
            QUOTIENT = np.squeeze(QUOTIENT)
        return QUOTIENT
    except Exception as e:
        print "somthing didnt work. chenck to make sure both arrays are the same dimmensions"
        print 'Error:\n\n' + str(e)
        m.getch()
        return -1

def multiply():
    system('cls')
    print "<<-----MULTIPLY-----"
    m1 = chop()
    m2 = chop()
    if isinstance(m1, int):
        if m1 == -1:
            return -1
    if isinstance(m2, int):
        if m2 == -1:
            return -1
    a = 'a'
    kepDim = False
    while a not in 'yYnN':
         print 'Keep origional dimmensions? (y/n)'
         a = m.getch()
    if a in 'yY':
        kepDim = True
    try:
        PRODUCT = np.multiply(m1, m2)
        print "Shape Product: ", np.shape(PRODUCT)
        print "Product: ", PRODUCT
        if kepDim == False:
            PRODUCT = np.squeeze(PRODUCT)
        return PRODUCT
    except Exception as e:
        print "somthing didnt work. chenck to make sure both arrays are the same dimmensions"
        print 'Error:\n\n' + str(e)
        m.getch()
        return -1

def average():
    system('cls')
    print "<<-----AVERAGE-----"
    avArr = chop()
    if isinstance(avArr, int):
        if avArr == -1:
            return -1
    a = 'a'
    kepDim = False
    while a not in 'yYnN':
         print 'Keep origional dimmensions? (y/n)'
         a = m.getch()
    if a in 'yY':
        kepDim = True
    AVERAGE = np.mean(avArr, axis=1, keepdims=kepDim)
    print "Shape Avg: ", np.shape(AVERAGE)
    print "Average: ", AVERAGE
    return AVERAGE

def importDat():
    system('cls')
    print "<<-----IMPORT DATASETS-----"
    print "import(1) or craete(2)?"
    choice = input('-> ')
    ret = -1
    if choice == 1:
        ret = input('sklearn.datasets import expression (Ex: load_digits()): ')
        # TODO: verify. make option to select subdivision of dataset
    elif choice == 2:
        system('cls')
        print "highest to lowest ex: (rows (y), columns (x))"
        shape = []
        shape.extend(input('what is the shape (Ex: [5(y), 6(x)] )? '))
        value = input('Fill array with what value?')
        ret = np.full(shape, value)
        # TODO: verify

    print "Press any key to continue"
    m.getch()
    return ret

def SVMFit(data, value):
    system('cls')
    gamma = input("gamma? ")
    C = input("C? ")
    cutRat = input("What ratio to cut data at for training and scoring?")
    print "-----Training SVM-----"
    cut = int(len(data)-cutRat*len(data))
    clf = svm.SVC(gamma=gamma, C=C)
    clf.fit(data[:cut], value[:cut])
    pred = clf.predict(data[cut:])
    print "\tSVM accurate to " + str(accuracy_score(value[cut:], pred))
    print "press any key to continue"
    m.getch()
    return clf

def RandForstFit(data, value):
    system('cls')
    n_estimators = input("n_estimators (defult=10)? ")
    max_depth = input("max_depth (defaule=None)? ")
    min_samples_split = input("min_samples_split (default=2)? ")
    cutRat = input("What ratio to cut data at for training and scoring? ")
    print "-----Training Random Forrest-----"
    cut = int(len(data)-cutRat*len(data))
    clf = ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(data[:cut], value[:cut])
    pred = clf.predict(data[cut:])
    print "\tRandom Forrest accurate to " + str(accuracy_score(value[cut:], pred))
    print "press any key to continue"
    m.getch()
    return clf

def NaiveBayesFit(data, value):
    system('cls')
    cutRat = input("What ratio to cut data at for training and scoring? ")
    print "-----Training Naive Bayes-----"
    cut = int(len(data)-cutRat*len(data))
    clf = naive_bayes.GaussianNB()
    clf.fit(data[:cut], value[:cut])
    pred = clf.predict(data[cut:])
    print "\tNaive Bayes accurate to " + str(accuracy_score(value[cut:], pred))
    print "press any key to continue"
    m.getch()
    return clf

def dat():
    system('cls')
    print "<<-----DATA-----"
    print """
1) Plot
2) View global variables
3) Add global variable
4) BACK
"""
    choice = input('Choice: ')
    if choice == 1:
        pltUsFr()
        editDat = -1
    elif choice ==2:
        printGlobVars('-> ')
        system('cls')
        print "\n\nPress any key to continue"
        m.getch()
        editDat = -1
    elif choice == 3:
        datasets = globVars['datasets']
        choice = input("Import/Create new dataset(1) or use existing dataset(2): ")
        if choice == 1:
            editDat = importDat()
            if isinstance(editDat, int):
                if editDat == -1:
                    return -1
        elif choice == 2:
            system('cls')
            print "<<-----OPERATIONS-----"
            print """
1) Leave the same
2) Chop
3) Average
4) Sum #TODO
5) Divide
6) Multiply
            """
            # TODO: add more operations like std dev, med, sum, divide
            choice = input('choice? ')
            if choice == 1:
                print "Choose dataset to use"
                for key in datasets:
                    print str(datasets.keys().index(key)) + ') ' + str(key)
                choice = input('Choice? ')
                editDat = datasets[datasets.keys()[choice]]
                print 'Leaving data unaltered'
            elif choice == 2:
                editDat = chop()
                if isinstance(editDat, int):
                    if editDat == -1:
                        return -1
            elif choice == 3:
                editDat = average()
                if isinstance(editDat, int):
                    if editDat == -1:
                        return -1
            elif choice == 4:
                editDat = summ()
                if isinstance(editDat, int):
                    if editDat == -1:
                        return -1
            elif choice == 5:
                editDat = divide()
                if isinstance(editDat, int):
                    if editDat == -1:
                        return -1
            elif choice == 6:
                editDat = multiply()
                if isinstance(editDat, int):
                    if editDat == -1:
                        return -1
            else:
                dat()
        globVars[input('Name? (surround with quotation marks) ')] = editDat
        print "Press any key to continue"
        m.getch()
        return -1
    elif choice == 4:
        return
    else:
        print "Try again\nPress any button"
        m.getch()
        dat()


def variabs():
    system('cls')
    print "<<-----VARIABLES-----"
    print """
1) Print data
2) Print target
3) Print algorithm
4) Set data
5) Set target
6) Set algorithm
7) BACK
"""
    choice = input('Choice: ')
    if choice == 1:
        print "Data\n\n"
        print runnable['data']
        print "\nPress any key to continue"
        m.getch()
        variabs()
    elif choice == 2:
        print "Target\n\n"
        print runnable['target']
        print "\nPress any key to continue"
        m.getch()
        variabs()
    elif choice == 3:
        print "Algorithm\n\n"
        print runnable['alg']
        print "\nPress any key to continue"
        m.getch()
        variabs()
    elif choice == 4:
        runnable['data'] = printGlobVars('Set Data to be classified - ', True)
    elif choice == 5:
        runnable['Target'] = printGlobVars('Set Data to be classified - ', True)
    elif choice == 6:
        algs()
    elif choice == 7:
        return
    else:
        print "Try again\nPress any button"
        m.getch()
        variabs()

def algs():
    system('cls')
    print "<<-----ALGORITHMS-----"
    print """
1) SVM
2) Random Forest
3) Naive Bayes
"""
    choice = input("Choice: ")
    if choice == 1:
        runnable['alg'] = SVMFit
    elif choice == 2:
        runnable['alg'] = RandForstFit
    elif choice == 3:
        runnable['alg'] = NaiveBayesFit
    else:
        print "Try again\nPress any button"
        m.getch()
        algs()

def run():
    system('cls')
    print "-----RUNNING-----"
    print "confirm: "
    for key in runnable:
        print '\t', key, ': ', runnable[key]
    m.getch()
    clf = runnable['alg'](runnable['data'], runnable['target'])
    return

def home():
    system('cls')
    print 'NOTE: shape appears in (z, y, x), (y, x), (x) and ()'
    print """-----HOME-----

1) Data
2) Runnable variables
3) RUN
4) Exit

--------------"""
    choice = input('Your choice: ')
    if choice == 1:
        dat()
    elif choice == 2:
        variabs()
    elif choice == 3:
        run()
    elif choice == 4:
        return
    home()

# digits = load_digits()
# data2d = readto3d(digits.data)
# mirdif3d = diffmirror(data2d)
# smirdif2d = readto2d(mirdif3d)
# plotDig(digits.images[1])
# plotdata(avg, mode='hist')
runnable = {'alg': SVMFit, 'data': digits.data, 'target': digits.target}
home()
system('cls')


"""
-SUM - easy, migt even do to demonstrate modularity
-DIVIDE/MULTIPLY #might work actually (there is code)
-PLOT - harder. only include rudimentary things
"""