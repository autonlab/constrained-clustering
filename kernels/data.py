from sklearn.preprocessing import StandardScaler

def get_transformation(data):
    """
        Iterates on all the transformation of the data
    
        Arguments:
            data {List of data} -- List of elements
    """
    # Raw data
    yield 'raw', data

    # Scale data
    standdata = StandardScaler().fit_transform(data)
    yield 'scaled', standdata