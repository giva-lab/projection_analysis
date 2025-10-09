from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_linnerud, fetch_openml
# from utility import *
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import os
import pickle

# Utility Functions
def save_metrics(metrics, filepath):
    """Save metrics to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(metrics, f)

def load_metrics(filepath):
    """Load metrics from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_binary_dt(data_path):
    with open(data_path, 'rb') as f:  # Open the file in binary mode
        return pickle.load(f, encoding='latin1')  # Use 'latin1' for compatibility
    

def generate_high_dimension_gaussians(num_dim, n_pts_per_gauss=200, spread_factor=0.1, distance_factor=1.0, distance_factor_2 = 1.0, move_cluster_index=0, random_seed=5):

    if random_seed is None:
        raise ValueError("random_seed cannot be None")

    # breakpoint()
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    

    # breakpoint()
    if num_dim < 1:
        raise ValueError("Number of dimensions must be at least 1.")
    
    centers = np.zeros((4, num_dim))  # Initialize with zeros

    # Second center: All ones except the last dimension is 0
    centers[1, :-1] = 1  # Set all elements except the last one to 1
    
    # Third center: All ones
    centers[2, :] = 1  # Set all elements to 1
    
    # Fourth center: First element is 1, rest are zeros
    centers[3, 0] = 1  # Set first element to 1

    # Move one cluster away dynamically
    moved_tetrahedron = centers.copy()
    moved_tetrahedron[move_cluster_index] *= distance_factor  # Move selected cluster
    moved_tetrahedron[1] *= distance_factor_2  # Move selected cluster

    # Generate Gaussian clusters
    centers = moved_tetrahedron
    cov_matrices = [np.eye(num_dim) * spread_factor for _ in range(4)]

    # Create dataset
    D = np.zeros((n_pts_per_gauss * 4, num_dim))
    c = np.zeros(n_pts_per_gauss * 4)

    for i in range(4):
        # D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
        D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = rng.multivariate_normal(
            centers[i], cov_matrices[i], n_pts_per_gauss
        )
        c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i  

    # Normalize dataset
    D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

    return D, c, centers



def cluster_position(cluster_spacing, mode):

    # if mode == 'cluster_1_far_other_close':
    #     centers = [
    #             [-10 * cluster_spacing, -10 * cluster_spacing, -10 * cluster_spacing],  # Cluster 1 (Far Away)
    #             [cluster_spacing, cluster_spacing, 0],                                 # Cluster 2 (Close to Cluster 3 and 4)
    #             [cluster_spacing + 1, cluster_spacing - 1, 0],                         # Cluster 3 (Close to Cluster 2 and 4)
    #             [cluster_spacing, cluster_spacing - 2, 0]                              # Cluster 4 (Close to Cluster 2 and 3)
    #         ]
    #     overlap_factors = [0.02, 0.02, 0.02, 0.02]

    if mode == 'cluster_1_far_other_close':   ## Cluster 1 is very far away from the other clusters. 
        ###The other three clusters (Clusters 2, 3, and 4) are equidistant from each other but are still equally far from Cluster 1
        # centers = [
        #         [-10 * cluster_spacing, -10 * cluster_spacing, -10 * cluster_spacing],  # Cluster 1 (Far Away)
        #         [cluster_spacing, 0, 0],                                               # Cluster 2
        #         [0, cluster_spacing, 0],                                               # Cluster 3
        #         [0, 0, cluster_spacing]                                                # Cluster 4
        #     ]
        # overlap_factors = [2.5, 2.5, 2.5, 2.5]
        distance_far = 20
        centers = [
                [-cluster_spacing + distance_far, -cluster_spacing + distance_far, -cluster_spacing + distance_far],  # Cluster 1 (Far Away)
                [cluster_spacing, 0, 0],                                               # Cluster 2
                [0, cluster_spacing, 0],                                               # Cluster 3
                [0, 0, cluster_spacing]                                                # Cluster 4
            ]
        over_value = 2.5
        overlap_factors = [over_value, over_value, over_value, over_value]

    # elif mode in ['tetrahedron_eq', 'tetrahedron_close', 'tetrahedron_far', 'tetrahedron_more_far']:
    elif any(mode == m for m in ['tetrahedron_eq_1_far','tetrahedron_eq', 'tetrahedron_eq_1_close', 'tetrahedron_eq_2_close', 'tetrahedron_more_far']):
        centers = np.array([
                [cluster_spacing, cluster_spacing, cluster_spacing],   
                [-cluster_spacing, -cluster_spacing, cluster_spacing],  
                [-cluster_spacing, cluster_spacing, -cluster_spacing],  
                [cluster_spacing, -cluster_spacing, -cluster_spacing]   
            ]) / np.sqrt(3)
        
        overlap_factors = None
     
    elif mode == 'equidistant':   #equidistant_old
        # cluster_spacing = 5* cluster_spacing
        # centers = [
        #         [cluster_spacing, cluster_spacing, cluster_spacing],        # Cluster 1
        #         [-cluster_spacing, cluster_spacing, cluster_spacing],    # Cluster 2
        #         [cluster_spacing, -cluster_spacing, cluster_spacing],    # Cluster 3
        #         [cluster_spacing, cluster_spacing, -cluster_spacing]     # Cluster 4
        #     ]
        # overlap_factors = [0.1, 0.1, 0.1, 0.1]

        centers = [
            [cluster_spacing, cluster_spacing, cluster_spacing],      # Cluster 1
            [-cluster_spacing, -cluster_spacing, cluster_spacing],    # Cluster 2
            [-cluster_spacing, cluster_spacing, -cluster_spacing],    # Cluster 3
            [cluster_spacing, -cluster_spacing, -cluster_spacing]     # Cluster 4
        ]
        over_value = 1.5
        overlap_factors = [over_value, over_value, over_value, over_value]
        
        # cluster_spacing = cluster_spacing           # equidistant_tetrahedron_centers
        # cluster_spacing = 5* cluster_spacing      # equidistant_tetrahedron_centers_more_far
        # centers = [
        #         [0, 0, 0],                    # Cluster 1
        #         [cluster_spacing, 0, 0],                    # Cluster 2
        #         [cluster_spacing / 2, np.sqrt(3) * cluster_spacing / 2, 0],  # Cluster 3
        #         [cluster_spacing / 2, np.sqrt(3) * cluster_spacing / 6, np.sqrt(6) * cluster_spacing / 3]  # Cluster 4
        #     ]
        # overlap_factors = [0.01, 0.01, 0.01, 0.01]       

        
    elif mode == '2_close_pairs':
        # Custom Centers: Clusters 1 & 2 are close, Clusters 3 & 4 are close, but pairs 1-2 and 3-4 are far apart.
        
        # The farthest pairs involve Cluster 1 and Cluster 3 or 4.
        # Cluster 2 and Cluster 3/4 are closer than Cluster 1 and Cluster 3/4
        # centers = [
        #     [-cluster_spacing, -cluster_spacing, 0],  
        #     [-cluster_spacing + 1, -cluster_spacing + 1, 0],  
        #     [cluster_spacing, cluster_spacing, 0],  
        #     [cluster_spacing + 1, cluster_spacing - 1, 0] 
        # ] 
        centers = [
            [-cluster_spacing, -cluster_spacing, 0],  
            [-cluster_spacing + 5, -cluster_spacing - 5, 0],  
            [cluster_spacing, cluster_spacing, 0],  
            [cluster_spacing + 5, cluster_spacing - 5, 0] 
        ] 
        # overlap_factors = [0.02, 0.02, 0.02, 0.02]
        over_value = 2.5
        overlap_factors = [over_value, over_value, over_value, over_value]

    elif mode == '1_close_pairs_1_pair_far':
        # Custom Centers: Clusters 1 & 2 are close, Clusters 3 & 4 are close, but Groups 1-2 and 3-4 are far apart
        # centers = [
        #     [-cluster_spacing, -cluster_spacing, 0],  
        #     [-cluster_spacing + 10, -cluster_spacing + 10, 0],  
        #     [cluster_spacing, cluster_spacing, 0],  
        #     [cluster_spacing + 1, cluster_spacing - 1, 0]  
        # ]
        # overlap_factors = [0.02, 0.02, 0.02, 0.02]
        centers = [
            [-cluster_spacing, -cluster_spacing, 0],  
            [-cluster_spacing + 10, -cluster_spacing + 10, 0],  
            [cluster_spacing - 3, cluster_spacing + 3, 0],  
            [cluster_spacing + 3, cluster_spacing - 3, 0]  
        ]
        over_value = 2
        over_value_2 =0.5
        overlap_factors = [over_value, over_value, over_value, over_value]

    elif mode == '2_10_points_far':
        # Custom Centers: Clusters 1 & 2 are close, Clusters 3 & 4 are close, but Groups 1-2 and 3-4 are far apart
        centers = [
            [-cluster_spacing, -cluster_spacing, 0],  
            [-cluster_spacing + 100, -cluster_spacing + 100, 0],  
            [cluster_spacing, cluster_spacing, 0],  
            [cluster_spacing + 100, cluster_spacing - 100, 0]  
        ]

        # centers = [       ### cluster 1/2 are close and cluster 3/4 are close. Cl 1/2 are far from cl 3/4. cluster 2 is little close to 3 and 4 than 1.
        #     [-cluster_spacing, -cluster_spacing, cluster_spacing],  
        #     [-cluster_spacing + 5, -cluster_spacing + 5, cluster_spacing],  
        #     [cluster_spacing, cluster_spacing, -cluster_spacing],  
        #     [cluster_spacing + 5, cluster_spacing - 5, -cluster_spacing]  
        # ]
        overlap_factors = [0.02, 0.02, 0.02, 0.02]
    
    elif mode == 'non_symmetric':                                                                              #Non-symmetric and unevenly spaced cluster centers simulate real-world scenarios where clusters are irregular. This setup helps test clustering algorithms' adaptability to imbalanced or irregular data.
        centers = [
            [3, 1.5, 0.9],
            [-1.8, -2.5, 2.3],
            [1.2, -2.2, 0.7],
            [-3.0, 2.8, -1.4]
        ]

        # Overlap factors for each cluster
        overlap_factors = [0.03, 0.02, 0.05, 0.04]
    
    elif mode == 'irregular':                                                                              # Mimics real-world Gaussian mixture models, where clusters follow distinct but natural distributions. Ideal for evaluating models like k-means and Gaussian Mixture Models (GMM) 
        centers = [
                [0, 0, 0],    # Dense cluster near origin
                [10, 10, 10], # Farther cluster
                [5, 0, 0],    # Cluster along one axis
                [0, 5, 10]    # Mixed position cluster
        ]
        overlap_factors = [0.5, 5.0, 2.0, 3.0]

    elif mode == 'sparse':
        centers = [
                [10, 0, 0],    
                [0, 10, 0], 
                [0, 0, 10],    
                [0, 0, 10]    
        ]
        overlap_factors = [0.1, 0.1, 0.1, 0.1]
    else:
        raise ValueError(f"Invalid mode '{mode}' provided.")
     
    return centers, overlap_factors

def generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=200, base_tetrahedron = None, spread_factor=0.1, distance_factor=1.0, distance_factor_2 = 1.0, move_cluster_index=0):
    """
    Generate well-separated 3D Gaussian clusters inside a tetrahedron with an option 
    to dynamically move one cluster away from others.

    Parameters:
        n_pts_per_gauss (int): Number of points per Gaussian.
        spread_factor (float): Variance of the Gaussian clusters.
        distance_factor (float): Scaling factor to move one cluster dynamically.
        move_cluster_index (int): Index (0-3) of the cluster to be moved.

    Returns:
        D (ndarray): Dataset of points.
        c (ndarray): Corresponding class labels.
        centers (ndarray): Updated centers of Gaussian clusters.
    """
    # Regular Tetrahedron Vertices (inside unit sphere)
    # base_tetrahedron = np.array([
    #     [1, 1, 1],   
    #     [-1, -1, 1],  
    #     [-1, 1, -1],  
    #     [1, -1, -1]   
    # ]) / np.sqrt(3)

    # Move one cluster away dynamically
    moved_tetrahedron = base_tetrahedron.copy()
    moved_tetrahedron[move_cluster_index] *= distance_factor  # Move selected cluster
    moved_tetrahedron[1] *= distance_factor_2  # Move selected cluster

    # Generate Gaussian clusters
    centers = moved_tetrahedron
    cov_matrices = [np.eye(3) * spread_factor for _ in range(4)]

    # Create dataset
    D = np.zeros((n_pts_per_gauss * 4, 3))
    c = np.zeros(n_pts_per_gauss * 4)

    for i in range(4):
        D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
            centers[i], cov_matrices[i], n_pts_per_gauss
        )
        c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i  

    # Normalize dataset
    D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

    return D, c, centers


def selected_dataset_dt(dataset, num_dim, n_pts_per_gauss, cluster_spacing = 1.0, spread_factor = 0.01):

    if dataset == "gaussian":
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        dim = 3
        output_size = dim
        n_gauss = 6
        D, c, centers = gaussian_dt(n_gauss, n_pts_per_gauss, dim)
    elif dataset == "tetrahedron_eq":
        distance_factor = 1.0
        distance_factor_2 = 1.0
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
        
    elif dataset == "tetrahedron_eq_1_far":
        distance_factor = 3.0
        distance_factor_2 = 1.0
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == "tetrahedron_eq_1_close":
        distance_factor = 0.0
        distance_factor_2 = 1.0
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == "tetrahedron_eq_2_close":
        distance_factor = 0.5
        distance_factor_2 = 0.5
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))

    elif dataset == "iris":
        dim = 4
        output_size = dim
        n_gauss = 3  # number of classes
        D, c = iris_dt()

    elif dataset == 'digits':

        D, c = digits_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
        
    elif dataset == 'har':

        D, c = har_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'covariance':

        D, c = covariance_type()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'wine':

        D, c = wine_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'breast':

        D, c = breast_cancer_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'cifar':

        D, c = cifar_10()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'high_dim':
        D, c, centers = generate_high_dimension_gaussians(num_dim = num_dim, n_pts_per_gauss=200, spread_factor= spread_factor)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'mnist':
        D, c = MNIST()
        dim = D.shape[1]
        output_size = dim
        # breakpoint()
        n_gauss = len(np.unique(c))
    elif dataset == 'linnerud':
        D, c = linnerud()
        dim = D.shape[1]
        output_size = dim
        # breakpoint()
        n_gauss = len(np.unique(c))

    elif dataset == 'bank_superv':
        datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle", "bank_marketing")
        # datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle")
        data_path = os.path.join(datasets_folder, "bank_full.csv")
        dt = pd.read_csv(data_path, sep= ";")
        # dt = pd.read_csv(data_path)
        # breakpoint()
        
        D,label = bank_superv(dt)
        # breakpoint()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label
    elif dataset == 'hcv_superv':
        
        data_path = os.path.join("datasets", "hcvdata.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)        
        D,label = hcv_superv(dt)
        # breakpoint()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label

    elif dataset in ['adessex']:
        
        data_path = os.path.join("datasets", "dataset_new_setup", "AedesSex.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)        
        D,label = spamebase_dt(dt)
        # breakpoint()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label

    elif dataset in ['anurancalls']:
        
        data_path = os.path.join("datasets", "dataset_new_setup", "anuranCalls.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)        
        D,label = spamebase_dt(dt)
        # breakpoint()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label
    elif dataset in ['arabicdigit']:
        
        data_path = os.path.join("datasets", "dataset_new_setup", "ArabicDigit.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)        
        D,label = spamebase_dt(dt)
        # breakpoint()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label
        # breakpoint()
    elif dataset in ['bng']:
        
        data_path = os.path.join("datasets", "dataset_new_setup", "BNG.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)        
        D,label = spamebase_dt(dt)
        # breakpoint()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label
        # breakpoint()
    elif dataset in ['magic']:
        
        data_path = os.path.join("datasets", "dataset_new_setup", "magic.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)        
        D,label = spamebase_dt(dt)
        
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label
    elif dataset in ['mozilla']:
        
        data_path = os.path.join("datasets", "dataset_new_setup", "mozilla.csv")
        dt = pd.read_csv(data_path)
        # dt = pd.read_csv(data_path)        
        D,label = spamebase_dt(dt)
        
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(label))
        c= label

    else:
        raise ValueError("Invalid dataset name.")

    # Normalize dataset
    # breakpoint()
    # D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))
    # breakpoint()
    return D,c, dim, output_size, n_gauss


##################################################################################################################################
# Get the current working directory (main project folder)
project_dir = os.getcwd()

def gaussian_dt(n_gauss, n_pts_per_gauss, dim):
    centers = np.random.uniform(-1, 1, size=(n_gauss, 3))
    cov_m = [np.diag([0.01 for _ in range(dim)]), np.diag([0.01 if i % 2 != 0 else 0.01 for i in range(dim)])]

    D = np.zeros((n_pts_per_gauss * n_gauss, dim))
    c = np.zeros(n_pts_per_gauss * n_gauss)
    for i in range(n_gauss):
        k = np.random.randint(0, 2, 1)[0]
        D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
            centers[i], cov_m[k], n_pts_per_gauss
        )
        c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i
    D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

    return D, c, centers

def iris_dt():
    iris = load_iris()
    data = iris['data']
    c = iris['target']
    target_names = iris['target_names']
    feature_names = iris['feature_names']

    # # Center the data (subtract the mean of each feature)
    # X_normalized = data - np.mean(data, axis=0)

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(data)

    # Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(X_normalized.T)

    return X_normalized, c

def digits_dt():
    # Load the digits dataset
    digits = load_digits()

    # Features and target
    X = digits.data
    y = digits.target
    # breakpoint()
    # # Normalize the features
    # non_zero_columns = ~np.all(X == 0, axis=0)
    # data_cleaned = X[:, non_zero_columns]
    # normalizer = MinMaxScaler()
    # X_normalized = normalizer.fit_transform(data_cleaned)
    # breakpoint()
    # # Standardize the features
    # scaler = StandardScaler()
    # X_stand = scaler.fit_transform(X)
    return X, y


def har_dt(sample_size=None, random_state=5):
    """
    Load and preprocess the HAR dataset, retaining only specific activities.

    Parameters:
        sample_size (int): Number of samples to retain (optional).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Normalized features and filtered labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "UCI_ HAR_ Dataset", "train")

    # File paths
    x_train_path = os.path.join(datasets_folder, "X_train.txt")
    y_train_path = os.path.join(datasets_folder, "y_train.txt")

    # Load the data as NumPy arrays
    X_train = np.loadtxt(x_train_path)
    y_train = np.loadtxt(y_train_path)

    # Filter for specific labels (WALKING: 1, SITTING: 4, STANDING: 5,  Laying: 6)
    desired_labels = [1, 2, 3, 4, 5, 6]
    # desired_labels = [1, 3]# 3, 4, 5, 6]   # C_0 = 1226, C_1= 1073,  C_2= 986
    mask = np.isin(y_train, desired_labels)
    X_filtered = X_train[mask]
    y_filtered = y_train[mask]

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X_filtered)

    # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    y_filtered = np.array([label_mapping[label] for label in y_filtered])
    # If sample_size is specified, perform stratified sampling
    if sample_size is not None and sample_size < len(y_filtered):
        X_normalized, _, y_filtered, _ = train_test_split(
            X_normalized,
            y_filtered,
            train_size=sample_size,
            stratify=y_filtered,
            random_state=random_state
        )
    # breakpoint()
    # Count samples for each class
    unique_labels, counts = np.unique(y_filtered, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    
        # Sort indices based on class labels
    sorted_indices = np.argsort(y_filtered)

    # Reorder dataset and labels
    X_sorted = X_normalized[sorted_indices]
    y_sorted = y_filtered[sorted_indices]
    # return X_normalized, y_filtered
    return X_sorted, y_sorted

def bank_superv(dt):
    data = bank(dt)
    X = data[:, :-1]  # All rows, all columns except the last
    y = data[:, -1] 

    breakpoint()
    return X, y

def hcv_superv(dt):
    data = hcv_dt(dt)
    X = data[:, 1:]  # All rows, all columns except the last
    y = data[:, 0] 

    breakpoint()
    return X, y

def spamebase_dt(dt):
    X = dt.drop(columns=['class'])
    y = dt['class'] 
    # breakpoint()
    return np.array(X), np.array(y)


def har_dt_v2(sample_size=None, random_state=5):
    """
    Load and preprocess the HAR dataset, retaining only specific activities.

    Parameters:
        sample_size (int): Number of samples to retain (optional).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Normalized features and filtered labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "UCI_ HAR_ Dataset", "train")

    # File paths
    x_train_path = os.path.join(datasets_folder, "X_train.txt")
    y_train_path = os.path.join(datasets_folder, "y_train.txt")

    # Load the data as NumPy arrays
    X_train = np.loadtxt(x_train_path)
    y_train = np.loadtxt(y_train_path)

    # # Filter for specific labels (WALKING: 1, SITTING: 4, STANDING: 5,  Laying: 6)
    # desired_labels = [1, 2, 3, 4, 5, 6]
    # # desired_labels = [1, 3]# 3, 4, 5, 6]   # C_0 = 1226, C_1= 1073,  C_2= 986
    # mask = np.isin(y_train, desired_labels)
    # X_filtered = X_train[mask]
    # y_filtered = y_train[mask]

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X_train)
    desired_labels = [1, 2, 3, 4, 5, 6]

    # # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    y_filtered = np.array([label_mapping[label] for label in y_train])
    # # If sample_size is specified, perform stratified sampling
    # if sample_size is not None and sample_size < len(y_filtered):
    #     X_normalized, _, y_filtered, _ = train_test_split(
    #         X_normalized,
    #         y_filtered,
    #         train_size=sample_size,
    #         stratify=y_filtered,
    #         random_state=random_state
    #     )
    # # breakpoint()
    # # Count samples for each class
    # breakpoint()
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    return X_normalized, y_filtered



def covariance_type(sample_size=None, random_state=5):
    """
    Reads the covariance dataset, separates labels, normalizes features,
    and returns a stratified sample if sample_size is provided.

    Args:
    - sample_size (int): Number of samples to return (maintains class ratio).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - normalized_features (ndarray): Normalized feature matrix.
    - labels (ndarray): Corresponding labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "covariance_type")
    
    # File path
    data_path = os.path.join(datasets_folder, "covertype.csv")
    
    # Read the CSV file while skipping the first row (column names)
    data = pd.read_csv(data_path, header=0)
    
    # Separate the label (last column)
    labels = data.iloc[:, -1].values  # Extract the last column as the labels
    
    # Extract features (all columns except the last)
    features = data.iloc[:, :-1].values  # Extract all columns except the last
    # breakpoint()
    # Filter for specific labels
    # desired_labels = [3, 5, 7]
    desired_labels = [1,2,3, 4,5,6, 7]
    mask = np.isin(labels, desired_labels)
    features = features[mask]
    labels = labels[mask]
    
    # Normalize the features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    labels = np.array([label_mapping[label] for label in labels])
    
    if sample_size is not None:
        # Stratified sampling to maintain class ratios
        normalized_features, _, labels, _ = train_test_split(
            normalized_features,
            labels,
            train_size=sample_size,
            stratify=labels,
            random_state=random_state
        )

    # Count samples for each class
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    
    return normalized_features, labels

def cifar_10(sample_size=None, random_state=5):
    """
    Reads the cifar_10 dataset, separates labels, normalizes features,
    and returns a stratified sample if sample_size is provided.

    Args:
    - sample_size (int): Number of samples to return (maintains class ratio).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - normalized_features (ndarray): Normalized feature matrix.
    - labels (ndarray): Corresponding labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "cifar-10-python_cs_totonto", "cifar-10-batches-py")

    # # List of batch file names
    # batch_files = [f"data_batch_{i}" for i in range(1, 6)]  # Adjust if you have more than 4 batches

    # # Initialize empty lists to store data and labels
    # all_data = []
    # all_labels = []

    # # Load each batch and append data & labels
    # for batch_file in batch_files:
    #     data_path = os.path.join(datasets_folder, batch_file)
    #     dt = load_binary_dt(data_path)
        
    #     all_data.append(dt[b'data'])  # Image data
    #     all_labels.append(dt[b'labels'])  # Labels

    # # Convert lists to numpy arrays
    # X = np.concatenate(all_data, axis=0)  # Stack all image data
    # y = np.concatenate(all_labels, axis=0)  # Stack all labels

    
    # File path
    data_path = os.path.join(datasets_folder, "data_batch_1")

    dt = load_binary_dt(data_path)
    
    # Extract label
    labels = dt['labels']
    labels = np.array(labels)
    

    # Extract features 
    features = dt['data']
    breakpoint()
    # # Filter for specific labels
    # desired_labels = [0,2,3]   # ['airplane', automobile, bird, cat, deer, dog, frog, horse, ship, truck]
    desired_labels = [0,1,2,3,4, 5,6,7, 8,9]   # ['airplane', automobile, bird, cat, deer, dog, frog, horse, ship, truck]

    mask = np.isin(labels, desired_labels)
    features = features[mask]
    labels = labels[mask]
    
    # Normalize the features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    labels = np.array([label_mapping[label] for label in labels])
    if sample_size is not None:
        # Stratified sampling to maintain class ratios
        normalized_features, _, labels, _ = train_test_split(
            normalized_features,
            labels,
            train_size=sample_size,
            stratify=labels,
            random_state=random_state
        )

    # Count samples for each class
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    
    return normalized_features, labels


def wine_dt():
    # Load the digits dataset
    digits = load_wine()

    # Features and target
    X = digits.data
    y = digits.target

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    # # Standardize the features
    # scaler = StandardScaler()
    # X_stand = scaler.fit_transform(X)
    return X_normalized, y

def breast_cancer_dt():
    # Load the digits dataset
    digits = load_breast_cancer()

    # Features and target
    X = digits.data
    y = digits.target

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    # # Standardize the features
    # scaler = StandardScaler()
    # X_stand = scaler.fit_transform(X)
    return X_normalized, y

def MNIST():
    mnist = fetch_openml("mnist_784", version=1, cache=True)
    X, y = mnist.data.to_numpy()[::7], mnist.target.to_numpy()[::7]  # X: images, y: labels
    # Normalize the features
    # normalizer = MinMaxScaler()
    # X_normalized = normalizer.fit_transform(X)

    return X, y

def linnerud():
    linnerud = load_linnerud()
    X, y = linnerud.data, linnerud.target  # X: images, y: labels
    
    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    return X_normalized, y


def selected_unsupervised_dt(dataset, a = None, b = None, c = None, freq=None, n_pts = None):

    if dataset == "helix":
        
        dim = 3
        output_size = dim
        data = helix_dt(n_pts, a, b, c, freq)

    elif dataset == "trefoil_knot":
        
        dim = 3
        output_size = dim
        data = Trefoil_knot_dt(n_pts, a, b, c, freq)

    elif dataset == "spiral":
        
        dim = 3
        output_size = dim
        data = spiral_dt(n_pts, a, b, c, freq)

    elif dataset == "lissajous":
        
        dim = 3
        output_size = dim
        data = lissajous_dt(n_pts, a, b, c, freq)
    
    elif dataset == "market":
        project_dir = "thesis_reproduced"
        datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle")
        data_path = os.path.join(datasets_folder, "Customer_Data.csv")
        dt = pd.read_csv(data_path, header=None, skiprows=1)
        breakpoint()
        data = market_segmentation(dt)
        breakpoint()
        dim = data.shape[1]
        output_size = dim

    elif dataset == "campaign":
        project_dir = "thesis_reproduced"
        datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle")
        data_path = os.path.join(datasets_folder, "marketing_campaign.csv")
        dt = pd.read_csv(data_path, sep="\t")
        # breakpoint()
        data = market_campaign(dt)
        dim = data.shape[1]
        output_size = dim

    elif dataset == "bank":
        project_dir = "thesis_reproduced"
        datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle", "bank_marketing")
        data_path = os.path.join(datasets_folder, "bank.csv")
        dt = pd.read_csv(data_path, sep= ";")
        breakpoint()
        data = bank(dt)
        dim = data.shape[1]
        output_size = dim

    elif dataset == "hcv":
        project_dir = "thesis_reproduced"
        # datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle", "bank_marketing")
        data_path = os.path.join("datasets", "hcvdata.csv")
        dt = pd.read_csv(data_path)
        # breakpoint()
        data = hcv_dt(dt)
        dim = data.shape[1]
        output_size = dim

    elif dataset == "medical":
        project_dir = "thesis_reproduced"
        # datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle", "bank_marketing")
        data_path = os.path.join("datasets", "medical_insurance.csv")
        dt = pd.read_csv(data_path)
        # breakpoint()
        data = medical_dt(dt)
        dim = data.shape[1]
        output_size = dim

    elif dataset == "epileptic":
        project_dir = "thesis_reproduced"
        # datasets_folder = os.path.join("datasets", "Market_segmentation_kaggle", "bank_marketing")
        data_path = os.path.join("datasets", "epileptic_seizure_recognition.csv")
        dt = pd.read_csv(data_path)
        # breakpoint()
        data = epileptic_dt(dt)
        dim = data.shape[1]
        output_size = dim
        
        # Apply Min-Max scaling
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data, dim, output_size



def helix_dt(n_pts, a, b, c, freq):
    # helix

    a = a # 1
    b = b # 2
    c = c # 0.5
    n = n_pts
    t = np.linspace(0,2*np.pi,n)
    # freq  = 2 

    xh = a*np.cos(freq*t)+np.random.uniform(-0.3,0.3,n)
    yh = b*np.sin(freq*t)+np.random.uniform(-0.3,0.3,n)
    zh = c*t+np.random.uniform(-0.3,0.3,n)
    data = np.stack((xh, yh, zh), axis=1)

    return data

def Trefoil_knot_dt(n_pts, a, b, c, freq):
    n = n_pts
    t = np.linspace(0,2*np.pi,n)

    # xt = a*np.sin(t)+2*np.sin(2*t)+np.random.uniform(-0.3,0.3,n)
    # yt = b*np.cos(t)-2*np.cos(2*t)+np.random.uniform(-0.3,0.3,n)
    xt = a*np.sin(t)+2*np.sin(freq*t)+np.random.uniform(-0.3,0.3,n)
    yt = b*np.cos(t)-2*np.cos(freq*t)+np.random.uniform(-0.3,0.3,n)
    zt = -c*np.sin(3*t)+np.random.uniform(-0.3,0.3,n)

    data = np.stack((xt, yt, zt), axis=1)

    return data

def spiral_dt(n_pts, a, b, c, freq):
    R = 1
    k = freq
    n = n_pts
    t = np.linspace(0,2*np.pi,n)

    xs = a*np.cos(t)*np.cos(k*t)+np.random.uniform(-0.3,0.3,n)
    ys = b*np.sin(t)*np.cos(k*t)+np.random.uniform(-0.3,0.3,n)
    zs = c*np.sin(k*t)+np.random.uniform(-0.3,0.3,n)
    # xs = R*np.cos(t)*np.cos(k*t)+np.random.uniform(-0.3,0.3,n)
    # ys = R*np.sin(t)*np.cos(k*t)+np.random.uniform(-0.3,0.3,n)
    # zs = R*np.sin(k*t)+np.random.uniform(-0.3,0.3,n)

    data = np.stack((xs, ys, zs), axis=1)

    return data

def lissajous_dt(n_pts, a, b, c, freq):
    # a = 3
    # b = 2
    # c = 1
    dx = 1
    dy = 2
    dz = 1
    n = n_pts
    t = np.linspace(0,2*np.pi,n)

    xl = a*np.sin(a*t+dx)+np.random.uniform(-0.3,0.3,n)
    yl = b*np.sin(b*t+dy)+np.random.uniform(-0.3,0.3,n)
    zl = c*np.sin(c*t+dz)+np.random.uniform(-0.3,0.3,n)

    data = np.stack((xl, yl, zl), axis=1)

    return data

def market_segmentation(data):
    # breakpoint()
    df = data.drop(columns=[0]) # Remove the first column about customer ID
    df = df.dropna()  # Remove rows with any NaN values
    df.reset_index(drop=True, inplace=True)
    # normalizer = Normalizer(norm='l1')
    # #normalize the data
    # normalizer.fit(df)
    # D = normalizer.transform(df)
    # breakpoint()
    # data = data.iloc[:, 1:]  # Remove the first column
    # data = data.dropna()  # Remove rows with any NaN values
    #     # Normalize the features
    scaler = MinMaxScaler()
    D = scaler.fit_transform(df)
    return np.array(D)

def market_campaign(data):
    
    # breakpoint()
    # df = data.drop(columns=[0]) # Remove the first column about customer ID
    # df = df.dropna()  # Remove rows with any NaN values

    data = data.dropna()
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True)

    dates = []
    for i in data["Dt_Customer"]:
        i = i.date()
        dates.append(i) 
    
    #Created a feature "Customer_For"
    days = []
    d1 = max(dates) #taking it to be the newest customer
    for i in dates:
        delta = d1 - i
        days.append(delta)
    data["Customer_For"] = days
    data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")

    #Feature Engineering
    #Age of customer today 
    data["Age"] = 2025-data["Year_Birth"]

        #Total spendings on various items
    data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

    #Deriving living situation by marital status"Alone"
    data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

    #Feature indicating total children living in the household
    data["Children"]=data["Kidhome"]+data["Teenhome"]

    #Feature for total members in the householde
    data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

    #Feature pertaining parenthood
    data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

    #Segmenting education levels in three groups
    data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

    #For clarity
    data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

    #Dropping some of the redundant features
    to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
    data = data.drop(to_drop, axis=1)

    #Dropping the outliers by setting a cap on Age and income. 
    data = data[(data["Age"]<90)]
    data = data[(data["Income"]<600000)]

    #Get list of categorical variables
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)
    #Label Encoding the object dtypes.
    LE=LabelEncoder()
    for i in object_cols:
        data[i]=data[[i]].apply(LE.fit_transform)
    

    #Creating a copy of data
    ds = data.copy()
    # creating a subset of dataframe by dropping the features on deals accepted and promotions
    cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
    ds = ds.drop(cols_del, axis=1)
    #Scaling
    scaler = StandardScaler()
    scaler.fit(ds)
    scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )

    # breakpoint()
    return np.array(scaled_ds)


def bank(dt):
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    # categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']

    label_encoder = LabelEncoder()
    df2 = dt.copy()

    for column in categorical_columns:
        df2[column] = label_encoder.fit_transform(dt[column])
    
    scaler = MinMaxScaler()
    scaler.fit(df2)
    scaled_ds = pd.DataFrame(scaler.transform(df2),columns= df2.columns )

    # breakpoint() 
    return np.array(scaled_ds)

def hcv_dt(data):
    # breakpoint()
    data = data.dropna()
    categorical_columns = ['Category', 'Sex']
    label_encoder = LabelEncoder()
    df2 = data.copy()

    for column in categorical_columns:
        df2[column] = label_encoder.fit_transform(data[column])
    
    df2 = df2.drop(columns=['Unnamed: 0'])
    exclude_col = 'Category'

    # Apply MinMaxScaler to all columns except the excluded one
    scaler = MinMaxScaler()
    df_scaled = df2.copy()
    df_scaled.loc[:, df2.columns != exclude_col] = scaler.fit_transform(df2.loc[:, df2.columns != exclude_col])

    # breakpoint() 
    return np.array(df_scaled)

def medical_dt(data):
    categorical_columns = ['sex', 'smoker', 'region']
    label_encoder = LabelEncoder()
    df2 = data.copy()

    for column in categorical_columns:
        df2[column] = label_encoder.fit_transform(data[column])
    
    breakpoint()
    # Apply MinMaxScaler to all columns except the excluded one
    scaler = MinMaxScaler()
    scaler.fit(df2)
    scaled_ds = pd.DataFrame(scaler.transform(df2),columns= df2.columns )
    # breakpoint() 
    return np.array(scaled_ds)

def epileptic_dt(data):
    
    data = data.drop(columns=['Unnamed'])
    
    # breakpoint()
    # # Apply MinMaxScaler to all columns except the excluded one
    # scaler = MinMaxScaler()
    # scaler.fit(data)
    # data = pd.DataFrame(scaler.transform(data),columns= data.columns )
    # breakpoint() 
    return np.array(data)
