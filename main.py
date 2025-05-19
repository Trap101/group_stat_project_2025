import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from icecream import ic
from tqdm import tqdm
import STOM_higgs_tools
from STOM_higgs_tools import get_B_chi

def exponential_distribution(x: np.ndarray, A: float, lamb: float) -> np.ndarray:
    #Exponential distribution function for background events.

    return A * np.exp(-x/lamb)


def gaussian_distribution(x: np.ndarray, B: float, mu: float, sigma: float) -> np.ndarray:
    #Gaussian distribution function for signal events. 
    return B * np.exp(-(x-mu)**2 / (2*sigma**2))


def combined_distribution(x: np.ndarray, *params) -> np.ndarray:
    #Combined distribution of exponential background and Gaussian signal.
    A, lamb, B, mu, sigma = params
    return exponential_distribution(x, A, lamb) + gaussian_distribution(x, B, mu, sigma)


MASS_RANGE = (104, 155)
NUM_BINS = 30
# Estimate background parameters this is perceived between the two vertical lines in the histogram
SIGNAL_REGION = (120, 130)

def main():
    # Generate and prepare data
    amplitude_values = np.array(STOM_higgs_tools.generate_data())
    
    # Define histogram parameters
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # plt.hist returns bin_heights, bin_edges, patches
    # where bin_heights is the number of events in each bin
    # bin_edges is the coordinates of the bin edges
    bin_heights, bin_edges, _ = plt.hist(
        amplitude_values, 
        bins=NUM_BINS, 
        alpha=0, 
        color='blue', 
        edgecolor='black', 
        range=MASS_RANGE
    )
    
    # Calculate bin properties
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    uncertainties = np.sqrt(bin_heights)  # Poisson uncertainty

    bin_width = bin_edges[1] - bin_edges[0]
    x_uncertainties = bin_width / 2
    
    background_values = np.concatenate((
        amplitude_values[amplitude_values < SIGNAL_REGION[0]],
        amplitude_values[amplitude_values > SIGNAL_REGION[1]]
    ))
    bin_heights_signal = bin_heights[
        (bin_centers >= SIGNAL_REGION[0]) & 
        (bin_centers <= SIGNAL_REGION[1])
    ]
    bin_centers_signal = bin_centers[
        (bin_centers >= SIGNAL_REGION[0]) & 
        (bin_centers <= SIGNAL_REGION[1])
    ]
    # finding the index of the peak of the bin_heights_signal and using that as gaussian mean
    peak_index,_ = find_peaks(bin_heights_signal)
    mu_estimate = bin_centers_signal[peak_index[0]]
    # take sigma as the standard deviation of the histogram within the signal range
    sigma_estimate = np.std(bin_centers_signal)

    # Estimate lambda for exponential background
    lamb_estimate = np.sum(background_values) / len(background_values)
    
    # Calculate amplitude for exponential background
    # by taking an integration of the background distribution over the mass range
    # and dividing by the area under the histogram
    area_under_histogram = np.sum(bin_heights) * bin_width
    A_estimate = area_under_histogram / (
        lamb_estimate * 
        (np.exp(-MASS_RANGE[0]/lamb_estimate) - 
         np.exp(-MASS_RANGE[1]/lamb_estimate))
    )
    # B_estimate has been obtained by taking the height of the peak of the gaussian shape in the histogram
    B_estimate = np.max(bin_heights)


    # Initial parameter guesses for curve fitting
    initial_params = [
        A_estimate,    # A
        lamb_estimate,  # lambda
        B_estimate,                 # B
        mu_estimate,              # mu
        sigma_estimate                  # sigma
    ]
    
    # Perform curve fitting
    fitted_params, _ = curve_fit(
        combined_distribution, 
        bin_centers, 
        bin_heights, 
        p0=initial_params
    )
    
    # Plot results
    x_fine = np.linspace(*MASS_RANGE, 1000)

    

    # Plot combined distribution
    ax.plot(
        x_fine, 
        combined_distribution(x_fine, *fitted_params),
        label='Combined Distribution'
    )
    
    # Plot background distribution
    ax.plot(
        x_fine,
        exponential_distribution(x_fine, *fitted_params[:2]),
        label='Exponential Distribution'
    )
    
    # Plot error bars
    ax.errorbar(
        bin_centers, 
        bin_heights, 
        yerr=uncertainties, 
        xerr=x_uncertainties, 
        fmt='k.', 
        capsize=3, 
        alpha=0.5
    )
    
    # Customize plot
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylabel('Number of Events')
    ax.set_title('Distribution of Signal and Background Events')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save and display plot
    # plt.savefig('hist.png')
    # plt.show()


########################################################
# task3
########################################################
# new plot
    fig, ax2 = plt.subplots(figsize=(10, 6))
    # function to calculate chi-square for a given bin
    def chi_square(bin_edges, amplitude_values, fitted_params):
        # left_bin_edges = bin_edges[bin_edges < 121]
        left_bin_edges = bin_edges[bin_edges < SIGNAL_REGION[0]]
        right_bin_edges = bin_edges[bin_edges > SIGNAL_REGION[1]]
        # -1 because the bin edges are not included in the bin_heights
        N_para_left = len(left_bin_edges)-1
        N_para_right = len(right_bin_edges)-1
        chi_left = get_B_chi(amplitude_values, (MASS_RANGE[0], left_bin_edges[0]), left_bin_edges, fitted_params[0], fitted_params[1])
        chi_right = get_B_chi(amplitude_values, (right_bin_edges[-1], MASS_RANGE[1]), right_bin_edges, fitted_params[0], fitted_params[1])
        total_chi = chi_left + chi_right
        total_dof = N_para_left + N_para_right-4
        # reduce chi-square to a single value
        return total_chi/total_dof
    # calculate chi-square for each bin
    # for lambda +- 10% and A +- 10%, plot the reduced chi-square
    

    def iterate_chi(A, lamb):
        lambda_range = np.linspace(lamb*0.9, lamb*1.1, 10)
        A_range = np.linspace(A*0.9, A*1.1, 10) 
        # Create meshgrid for 3D plotting
        LAMBDA, A = np.meshgrid(A_range, lambda_range)
        chi_square_values = np.zeros_like(LAMBDA) 
        # Calculate chi-square values for each combination
        for i, A_val in enumerate(tqdm(A_range)):
            for j, lambda_ in enumerate(lambda_range):
                chi_square_values[i, j] = chi_square(bin_edges, amplitude_values, 
                                                [A_val, lambda_, fitted_params[2], 
                                                    fitted_params[3], fitted_params[4]]) 
        # Find the chi-square value closest to 1
        min_diff_to_one = np.argmin(np.abs(chi_square_values - 1))
        # Get the corresponding A and lambda values
        A_idx = min_diff_to_one//len(A_range)
        lambda_idx = min_diff_to_one%len(lambda_range)
        A_min = A_range[A_idx]
        lamb_min = lambda_range[lambda_idx]
        chi_min = chi_square_values[A_idx, lambda_idx]
        # save chi_square_values to a csv file
        np.savetxt('chi_square_values.csv', chi_square_values, delimiter=',')
        
        return A_min, lamb_min, chi_min
    A_min, lamb_min, chi_min = iterate_chi(fitted_params[0], fitted_params[1])
    A_min_2, lamb_min_2, chi_min_2 = iterate_chi(A_min, lamb_min)
    ic(A_min, lamb_min, chi_min)
    ic(A_min_2, lamb_min_2, chi_min_2)


    # # Create 3D plot
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Plot the surface
    # surf = ax.plot_surface(LAMBDA, A, chi_square_values, cmap='viridis')
    
    # # Add labels and title
    # ax.set_xlabel('Lambda')
    # ax.set_ylabel('A')
    # ax.set_zlabel('Reduced Chi-square')
    # ax.set_title('Chi-square Surface Plot')
    
    # # Add colorbar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    # plt.show()



if __name__ == "__main__":
    main()
