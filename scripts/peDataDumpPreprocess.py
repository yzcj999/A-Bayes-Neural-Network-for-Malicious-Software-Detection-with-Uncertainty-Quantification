#install the pefile module using !pip install pefile
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import pickle as pck
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def createDataframeFromPEdump(pe):
    dosHeaders = ['e_magic', 'e_cblp', 'e_cp', 'e_crlc', 'e_cparhdr',
                  'e_minalloc', 'e_maxalloc', 'e_ss', 'e_sp', 'e_csum', 'e_ip', 'e_cs',
                  'e_lfarlc', 'e_ovno', 'e_oemid', 'e_oeminfo', 'e_lfanew']
    fileHeaders = ['Machine',
                   'NumberOfSections', 'TimeDateStamp', 'PointerToSymbolTable',
                   'NumberOfSymbols', 'SizeOfOptionalHeader', 'Characteristics']
    optionalHeaders = ['Magic',
                       'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode',
                       'SizeOfInitializedData', 'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode',
                       'ImageBase', 'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion',
                       'MinorOperatingSystemVersion', 'MajorImageVersion', 'MinorImageVersion',
                       'MajorSubsystemVersion', 'MinorSubsystemVersion', 'SizeOfHeaders',
                       'CheckSum', 'SizeOfImage', 'Subsystem', 'DllCharacteristics',
                       'SizeOfStackReserve', 'SizeOfStackCommit', 'SizeOfHeapReserve',
                       'SizeOfHeapCommit', 'LoaderFlags', 'NumberOfRvaAndSizes']
    imageDirectory = ['ImageDirectoryEntryExport', 'ImageDirectoryEntryImport',
                      'ImageDirectoryEntryResource', 'ImageDirectoryEntryException',
                      'ImageDirectoryEntrySecurity']

    dheaders = {}
    fheaders = {}
    oheaders = {}
    imd1 = {}

    for x in dosHeaders:
        dheaders[x] = getattr(pe.DOS_HEADER, x)
    df = pd.DataFrame(dheaders, index=[0])

    for i in fileHeaders:
        fheaders[i] = getattr(pe.FILE_HEADER, i)
    df = pd.concat([df, (pd.DataFrame(fheaders, index=[0]))], axis=1)

    for y in optionalHeaders:
        oheaders[y] = getattr(pe.OPTIONAL_HEADER, y)
    df = pd.concat([df, (pd.DataFrame(oheaders, index=[0]))], axis=1)

    for q in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
        imd1[q.name] = q.VirtualAddress
    imd1 = dict(itertools.islice(imd1.items(), 5))
    df = pd.concat([df, (pd.DataFrame(imd1, index=[0]))], axis=1)

    return df


def getPredictions(df):
    print(df.shape)
    train = pd.read_csv('scripts/dataset_malwares.csv', sep=',')
    test = pd.read_csv('scripts/dataset_test.csv', sep=',')
    # #The target is Malware Column {0=Benign, 1=Malware}

    X = train.drop(['Name', 'Malware'], axis=1)
    y = train['Malware']
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    # Feature Scaling

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_new = pd.DataFrame(X_scaled, columns=X_train.columns)
    skpca = PCA(n_components=55)
    X_pca = skpca.fit_transform(X_new)
    model = LGBMClassifier(n_estimators=100, random_state=0, oob_score=True, max_depth=16, max_features='sqrt')
    model.fit(X_pca, y_train)

    X_test_scaled = scaler.transform(X_test)
    X_new_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    X_test_pca = skpca.transform(X_new_test)
    y_pred = model.predict(X_test_pca)

    pipe = Pipeline([('scale', scaler), ('pca', skpca), ('clf', model)])
    X_testing = test.drop('Name', axis=1)
    print(X_testing.shape)
    X_testing_scaled = pipe.named_steps['scale'].transform(X_testing)
    X_testing_pca = pipe.named_steps['pca'].transform(X_testing_scaled)
    y_testing_pred = pipe.named_steps['clf'].predict_proba(X_testing_pca)


    df = np.array(df)
    df = df.reshape(1, -1)
    results = pipe.predict_proba(df)
    pred = pipe.predict(df)
    return results[0]


